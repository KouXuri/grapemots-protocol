#!/usr/bin/env python3
"""Analyse the out-of-fold GrapeMOTS configuration experiment.

The input directory must contain ``oof_{A..E}_{arm}.json`` files produced by
``track_grapemots_mot.py``. Each file contains two held-out videos and the
per-frame predicted and ground-truth IDs and boxes required for count-error
decomposition and, when installed, TrackEval's HOTA metric family.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tools.decompose_count_error import HAVE_SCIPY, decompose, iou_matrix
except ModuleNotFoundError as error:  # Support ``python tools/analyse_grapemots_oof.py``.
    if error.name != "tools":
        raise
    from decompose_count_error import HAVE_SCIPY, decompose, iou_matrix


SPLITS = ("A", "B", "C", "D", "E")
ARMS = ("conf055", "conf040", "ios", "botsort", "bytetrack", "reid")
ARM_LABELS = {
    "conf055": "Confidence 0.55",
    "conf040": "Confidence 0.40",
    "ios": "IoS merge",
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "reid": "BoT-SORT + ReID",
}
PRIMARY_SPLITS = ("A", "B", "C")
METRIC_NAMES = ("HOTA", "DetA", "AssA", "LocA")
STANDARD_METRIC_NAMES = ("idf1", "mota", "recall")
_AUTO_HOTA = object()


def _result_path(results: Path, split: str, arm: str) -> Path:
    plain = results / f"oof_{split}_{arm}.json"
    compressed = plain.with_suffix(".json.gz")
    if plain.is_file():
        return plain
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(
        f"Missing expected result file: {plain} (or compressed {compressed.name})"
    )


def _read_json(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        message = getattr(error, "msg", str(error))
        raise ValueError(f"{path}: invalid JSON ({message})") from error


def _validate_video(video: Any, source: Path) -> None:
    if not isinstance(video, dict):
        raise ValueError(f"{source}: each videos entry must be an object")
    name = video.get("video")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{source}: every videos entry needs a non-empty video name")

    keys = (
        "frame_predicted_ids",
        "frame_predicted_boxes",
        "frame_gt_ids",
        "frame_gt_boxes",
    )
    missing = [key for key in keys if key not in video]
    if missing:
        raise ValueError(f"{source}: video {name!r} is missing {', '.join(missing)}")
    arrays = [video[key] for key in keys]
    if any(not isinstance(value, list) for value in arrays):
        raise ValueError(f"{source}: video {name!r} frame dumps must be lists")
    lengths = [len(value) for value in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"{source}: video {name!r} has inconsistent frame-dump lengths {lengths}")
    if "frames" in video and video["frames"] != lengths[0]:
        raise ValueError(
            f"{source}: video {name!r} declares {video['frames']} frames but dumps {lengths[0]}"
        )

    for frame_index, (pred_ids, pred_boxes, gt_ids, gt_boxes) in enumerate(zip(*arrays)):
        for role, ids, boxes in (
            ("predicted", pred_ids, pred_boxes),
            ("ground-truth", gt_ids, gt_boxes),
        ):
            if not isinstance(ids, list) or not isinstance(boxes, list):
                raise ValueError(
                    f"{source}: {name!r} frame {frame_index} {role} IDs and boxes must be lists"
                )
            if len(ids) != len(boxes):
                raise ValueError(
                    f"{source}: {name!r} frame {frame_index} has {len(ids)} {role} IDs "
                    f"but {len(boxes)} boxes"
                )
            if len(set(ids)) != len(ids):
                raise ValueError(
                    f"{source}: {name!r} frame {frame_index} repeats a {role} track ID"
                )
            if any(not isinstance(track_id, int) or isinstance(track_id, bool) for track_id in ids):
                raise ValueError(
                    f"{source}: {name!r} frame {frame_index} has a non-integer {role} track ID"
                )
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(
                        f"{source}: {name!r} frame {frame_index} has an invalid {role} box"
                    )
                if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in box):
                    raise ValueError(
                        f"{source}: {name!r} frame {frame_index} has a non-finite {role} box"
                    )
                if box[2] < box[0] or box[3] < box[1]:
                    raise ValueError(
                        f"{source}: {name!r} frame {frame_index} has an inverted {role} box"
                    )


def _checked_splits(splits: Sequence[str], label: str) -> Tuple[str, ...]:
    values = tuple(split.upper() for split in splits)
    if not values:
        raise ValueError(f"{label} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate split names")
    invalid = [split for split in values if split not in SPLITS]
    if invalid:
        raise ValueError(f"{label} contains unsupported splits: {', '.join(invalid)}")
    return values


def load_oof_results(
    results: Path,
    splits: Sequence[str] = SPLITS,
    primary_splits: Sequence[str] = PRIMARY_SPLITS,
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Load and pair all 30 expected split-arm outputs."""
    splits = _checked_splits(splits, "Splits")
    primary_splits = _checked_splits(primary_splits, "Primary splits")
    missing_primary = [split for split in primary_splits if split not in splits]
    if missing_primary:
        raise ValueError(
            "Primary splits must be included in --splits; missing " + ", ".join(missing_primary)
        )
    loaded: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for split in splits:
        loaded[split] = {}
        expected_names: Optional[set] = None
        reference: Optional[Dict[str, dict]] = None
        for arm in ARMS:
            path = _result_path(results, split, arm)
            payload = _read_json(path)
            videos = payload.get("videos") if isinstance(payload, dict) else None
            if not isinstance(videos, list):
                raise ValueError(f"{path}: top-level 'videos' must be a list")
            if len(videos) != 2:
                raise ValueError(f"{path}: expected two held-out videos, found {len(videos)}")
            for video in videos:
                _validate_video(video, path)
            by_name = {video["video"]: video for video in videos}
            if len(by_name) != len(videos):
                raise ValueError(f"{path}: video names must be unique within a split-arm file")
            names = set(by_name)
            if expected_names is None:
                expected_names = names
                reference = by_name
            elif names != expected_names:
                raise ValueError(
                    f"{path}: held-out videos {sorted(names)} do not match split {split} "
                    f"reference {sorted(expected_names)}"
                )
            assert reference is not None
            for name, video in by_name.items():
                if (
                    video["frame_gt_ids"] != reference[name]["frame_gt_ids"]
                    or video["frame_gt_boxes"] != reference[name]["frame_gt_boxes"]
                ):
                    raise ValueError(
                        f"{path}: ground truth for split {split}, video {name!r} differs across arms"
                    )
            loaded[split][arm] = by_name

    primary_names = [name for split in primary_splits for name in loaded[split][ARMS[0]]]
    expected_primary_cells = 2 * len(primary_splits)
    if len(primary_names) != expected_primary_cells or len(set(primary_names)) != expected_primary_cells:
        raise ValueError(
            f"Primary splits {'/'.join(primary_splits)} must contain {expected_primary_cells} unique "
            "held-out videos; "
            f"found {len(set(primary_names))} unique names across {len(primary_names)} cells"
        )
    return loaded


def build_trackeval_data(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool video cells for TrackEval while making identities cell-local."""
    gt_ids: List[np.ndarray] = []
    tracker_ids: List[np.ndarray] = []
    similarities: List[np.ndarray] = []
    gt_offset = 0
    tracker_offset = 0

    for entry in entries:
        unique_gt = sorted({track_id for frame in entry["frame_gt_ids"] for track_id in frame})
        unique_tracker = sorted(
            {track_id for frame in entry["frame_predicted_ids"] for track_id in frame}
        )
        gt_map = {track_id: gt_offset + index for index, track_id in enumerate(unique_gt)}
        tracker_map = {
            track_id: tracker_offset + index for index, track_id in enumerate(unique_tracker)
        }
        for frame_gt_ids, frame_gt_boxes, frame_pred_ids, frame_pred_boxes in zip(
            entry["frame_gt_ids"],
            entry["frame_gt_boxes"],
            entry["frame_predicted_ids"],
            entry["frame_predicted_boxes"],
        ):
            gt_ids.append(np.asarray([gt_map[track_id] for track_id in frame_gt_ids], dtype=int))
            tracker_ids.append(
                np.asarray([tracker_map[track_id] for track_id in frame_pred_ids], dtype=int)
            )
            similarities.append(
                iou_matrix(
                    np.asarray(frame_gt_boxes, dtype=float),
                    np.asarray(frame_pred_boxes, dtype=float),
                )
            )
        gt_offset += len(unique_gt)
        tracker_offset += len(unique_tracker)

    return {
        "num_timesteps": len(gt_ids),
        "num_gt_ids": gt_offset,
        "num_tracker_ids": tracker_offset,
        "num_gt_dets": int(sum(len(frame) for frame in gt_ids)),
        "num_tracker_dets": int(sum(len(frame) for frame in tracker_ids)),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
    }


def _resolve_hota_factory() -> Tuple[Optional[Callable[[], Any]], Optional[str]]:
    try:
        from trackeval.metrics import HOTA
    except Exception as error:  # Optional dependency may fail while importing a transitive module.
        return None, (
            "TrackEval is unavailable, so pooled HOTA, DetA, AssA and LocA were not "
            f"calculated ({type(error).__name__}: {error})."
        )
    return HOTA, None


def _unavailable_hota(reason: Optional[str]) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason
        or "TrackEval was not supplied, so pooled HOTA, DetA, AssA and LocA were not calculated.",
        **{name: None for name in METRIC_NAMES},
    }


def evaluate_hota(
    entries: Sequence[Mapping[str, Any]],
    hota_factory: Optional[Callable[[], Any]],
    unavailable_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if hota_factory is None:
        return _unavailable_hota(unavailable_reason)
    result = hota_factory().eval_sequence(build_trackeval_data(entries))
    metrics = {}
    for name in METRIC_NAMES:
        if name not in result:
            raise ValueError(f"TrackEval HOTA result is missing {name}")
        value = float(np.mean(result[name]))
        if not math.isfinite(value):
            raise ValueError(f"TrackEval returned a non-finite pooled {name}")
        metrics[name] = value
    return {"available": True, "reason": None, **metrics}


def _median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return float(median(present)) if present else None


def _standard_metrics(video: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    source = video.get("metrics", {})
    metrics = {}
    for name in STANDARD_METRIC_NAMES:
        value = source.get(name) if isinstance(source, Mapping) else None
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{video.get('video', '<unknown>')}: non-finite {name}")
        metrics[name] = value
    return metrics


def _summarise_arm(
    loaded: Mapping[str, Mapping[str, Mapping[str, dict]]],
    splits: Sequence[str],
    arm: str,
    match_iou: float,
    min_track_len: int,
    hota_factory: Optional[Callable[[], Any]],
    hota_reason: Optional[str],
) -> Dict[str, Any]:
    cells = []
    entries = []
    for split in splits:
        for video_name, video in sorted(loaded[split][arm].items()):
            row = decompose(video, threshold=match_iou, tau=min_track_len)
            recovered_fraction = (row["G"] - row["M"]) / row["G"] if row["G"] else None
            per_cell_hota = evaluate_hota([video], hota_factory, hota_reason)
            cells.append(
                {
                    "split": split,
                    "video": video_name,
                    "cell": f"{split}/{video_name}",
                    "P": row["P"],
                    "G": row["G"],
                    "U": row["U"],
                    "D": row["D"],
                    "M": row["M"],
                    "net_signed_error": row["signed_error"],
                    "recovered_fraction": recovered_fraction,
                    "identity_holds": row["identity_holds"],
                    "tracking_metrics": {
                        **_standard_metrics(video),
                        **{name: per_cell_hota[name] for name in METRIC_NAMES},
                    },
                }
            )
            entries.append(video)

    pooled_counts = {name: sum(cell[name] for cell in cells) for name in ("P", "G", "U", "D", "M")}
    pooled = {
        **pooled_counts,
        "net_signed_error": (
            (pooled_counts["P"] - pooled_counts["G"]) / pooled_counts["G"]
            if pooled_counts["G"]
            else None
        ),
        "recovered_trajectories": pooled_counts["G"] - pooled_counts["M"],
        "recovered_fraction": (
            (pooled_counts["G"] - pooled_counts["M"]) / pooled_counts["G"]
            if pooled_counts["G"]
            else None
        ),
        "identity_holds": (
            pooled_counts["U"] + pooled_counts["D"] - pooled_counts["M"]
            == pooled_counts["P"] - pooled_counts["G"]
        ),
    }
    identity_passed = sum(bool(cell["identity_holds"]) for cell in cells)
    return {
        "label": ARM_LABELS[arm],
        "cells": cells,
        "pooled": pooled,
        "macro_medians": {
            "per_cell_net_signed_error": _median_or_none(
                cell["net_signed_error"] for cell in cells
            ),
            "per_cell_absolute_error": _median_or_none(
                abs(cell["net_signed_error"]) if cell["net_signed_error"] is not None else None
                for cell in cells
            ),
            "per_cell_recovered_fraction": _median_or_none(
                cell["recovered_fraction"] for cell in cells
            ),
            **{
                f"per_cell_{name}": _median_or_none(
                    cell["tracking_metrics"][name] for cell in cells
                )
                for name in (*STANDARD_METRIC_NAMES, *METRIC_NAMES)
            },
        },
        "identity_checks": {
            "all_hold": identity_passed == len(cells) and pooled["identity_holds"],
            "passed": identity_passed,
            "total": len(cells),
            "failed_cells": [cell["cell"] for cell in cells if not cell["identity_holds"]],
        },
        "pooled_tracking_metrics": evaluate_hota(entries, hota_factory, hota_reason),
    }


def _winner(first: Optional[float], second: Optional[float], higher: bool) -> str:
    if first is None or second is None:
        return "unavailable"
    if math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12):
        return "tie"
    first_wins = first > second if higher else first < second
    return "conf055" if first_wins else "reid"


def pairwise_inversions(conf055_cells: Sequence[Mapping[str, Any]], reid_cells: Sequence[Mapping[str, Any]]) -> dict:
    """Compare count closeness with recovery on the same split-video cells."""
    conf_by_cell = {cell["cell"]: cell for cell in conf055_cells}
    reid_by_cell = {cell["cell"]: cell for cell in reid_cells}
    if set(conf_by_cell) != set(reid_by_cell):
        raise ValueError("Confidence-0.55 and ReID results do not contain the same paired cells")

    rows = []
    for cell_id in sorted(conf_by_cell):
        conf = conf_by_cell[cell_id]
        reid = reid_by_cell[cell_id]
        conf_error = conf["net_signed_error"]
        reid_error = reid["net_signed_error"]
        absolute_error_winner = _winner(
            abs(conf_error) if conf_error is not None else None,
            abs(reid_error) if reid_error is not None else None,
            higher=False,
        )
        recovery_winner = _winner(
            conf["recovered_fraction"], reid["recovered_fraction"], higher=True
        )
        strict_inversion = (
            absolute_error_winner not in {"tie", "unavailable"}
            and recovery_winner not in {"tie", "unavailable"}
            and absolute_error_winner != recovery_winner
        )
        directional_signed = (
            conf_error is not None
            and reid_error is not None
            and conf["recovered_fraction"] is not None
            and reid["recovered_fraction"] is not None
            and conf_error < reid_error
            and conf["recovered_fraction"] < reid["recovered_fraction"]
        )
        rows.append(
            {
                "split": conf["split"],
                "video": conf["video"],
                "cell": cell_id,
                "conf055": {
                    "net_signed_error": conf_error,
                    "absolute_error": abs(conf_error) if conf_error is not None else None,
                    "recovered_fraction": conf["recovered_fraction"],
                },
                "reid": {
                    "net_signed_error": reid_error,
                    "absolute_error": abs(reid_error) if reid_error is not None else None,
                    "recovered_fraction": reid["recovered_fraction"],
                },
                "lower_absolute_error_arm": absolute_error_winner,
                "higher_recovery_arm": recovery_winner,
                "strict_inversion": strict_inversion,
                "conf055_lower_signed_error_reid_higher_recovery": directional_signed,
            }
        )

    return {
        "definition": (
            "A strict inversion is a paired cell in which the arm with lower absolute "
            "signed count error is not the arm with higher recovered fraction."
        ),
        "total_paired_cells": len(rows),
        "strict_inversion_count": sum(row["strict_inversion"] for row in rows),
        "conf055_lower_absolute_error_reid_higher_recovery_count": sum(
            row["strict_inversion"]
            and row["lower_absolute_error_arm"] == "conf055"
            and row["higher_recovery_arm"] == "reid"
            for row in rows
        ),
        "reid_lower_absolute_error_conf055_higher_recovery_count": sum(
            row["strict_inversion"]
            and row["lower_absolute_error_arm"] == "reid"
            and row["higher_recovery_arm"] == "conf055"
            for row in rows
        ),
        "conf055_lower_signed_error_reid_higher_recovery_count": sum(
            row["conf055_lower_signed_error_reid_higher_recovery"] for row in rows
        ),
        "absolute_error_tie_count": sum(
            row["lower_absolute_error_arm"] == "tie" for row in rows
        ),
        "recovery_tie_count": sum(row["higher_recovery_arm"] == "tie" for row in rows),
        "cells": rows,
    }


def analyse_results(
    loaded: Mapping[str, Mapping[str, Mapping[str, dict]]],
    match_iou: float = 0.5,
    min_track_len: int = 1,
    hota_factory: Any = _AUTO_HOTA,
    splits: Sequence[str] = SPLITS,
    primary_splits: Sequence[str] = PRIMARY_SPLITS,
) -> dict:
    if not 0 <= match_iou <= 1:
        raise ValueError("Match IoU must lie between 0 and 1")
    if min_track_len < 1:
        raise ValueError("Minimum track length must be at least 1")
    splits = _checked_splits(splits, "Splits")
    primary_splits = _checked_splits(primary_splits, "Primary splits")
    if any(split not in splits for split in primary_splits):
        raise ValueError("Every primary split must also be present in the sensitivity splits")
    if any(split not in loaded for split in splits):
        raise ValueError("Loaded results do not contain every requested split")
    cohorts = {"primary": primary_splits, "sensitivity": splits}
    if hota_factory is _AUTO_HOTA:
        hota_factory, hota_reason = _resolve_hota_factory()
    elif hota_factory is None:
        hota_reason = None
    else:
        hota_reason = None

    report = {
        "schema_version": 1,
        "protocol": {
            "splits": list(splits),
            "arms": list(ARMS),
            "match_iou": match_iou,
            "minimum_track_length": min_track_len,
            "assignment": "Hungarian" if HAVE_SCIPY else "greedy fallback",
            "primary_splits": list(primary_splits),
            "sensitivity_splits": list(splits),
        },
        "cohorts": {},
    }
    for cohort_name, cohort_splits in cohorts.items():
        arms = {
            arm: _summarise_arm(
                loaded,
                cohort_splits,
                arm,
                match_iou,
                min_track_len,
                hota_factory,
                hota_reason,
            )
            for arm in ARMS
        }
        failures = [
            {"arm": arm, "cell": cell["cell"]}
            for arm, summary in arms.items()
            for cell in summary["cells"]
            if not cell["identity_holds"]
        ]
        total_checks = sum(summary["identity_checks"]["total"] for summary in arms.values())
        report["cohorts"][cohort_name] = {
            "splits": list(cohort_splits),
            "split_video_cells": len(cohort_splits) * 2,
            "arms": arms,
            "identity_checks": {
                "all_hold": not failures,
                "passed": total_checks - len(failures),
                "total": total_checks,
                "failures": failures,
            },
            "conf055_vs_reid": pairwise_inversions(
                arms["conf055"]["cells"], arms["reid"]["cells"]
            ),
        }
    return report


def _number(value: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(value, spec)


def _markdown_cohort(name: str, cohort: Mapping[str, Any]) -> List[str]:
    label = "Primary cohort" if name == "primary" else "Sensitivity cohort"
    splits = "/".join(cohort["splits"])
    lines = [f"## {label} (splits {splits})", ""]
    lines.extend(
        [
            "| Arm | Cells | P | G | U | D | M | Net error | Recovered | "
            "Median cell error | Median cell recovery | IDF1 | MOTA | Recall | "
            "HOTA | DetA | AssA | LocA | Identity |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        summary = cohort["arms"][arm]
        pooled = summary["pooled"]
        medians = summary["macro_medians"]
        hota = summary["pooled_tracking_metrics"]
        identity = summary["identity_checks"]
        lines.append(
            "| {label} | {cells} | {P} | {G} | {U} | {D} | {M} | {error} | {recovery} | "
            "{median_error} | {median_recovery} | {idf1} | {mota} | {recall} | "
            "{HOTA} | {DetA} | {AssA} | {LocA} | {passed}/{total} |".format(
                label=summary["label"],
                cells=len(summary["cells"]),
                error=_number(pooled["net_signed_error"], signed=True),
                recovery=_number(pooled["recovered_fraction"]),
                median_error=_number(medians["per_cell_net_signed_error"], signed=True),
                median_recovery=_number(medians["per_cell_recovered_fraction"]),
                idf1=_number(medians["per_cell_idf1"]),
                mota=_number(medians["per_cell_mota"]),
                recall=_number(medians["per_cell_recall"]),
                HOTA=_number(hota["HOTA"], digits=4),
                DetA=_number(hota["DetA"], digits=4),
                AssA=_number(hota["AssA"], digits=4),
                LocA=_number(hota["LocA"], digits=4),
                passed=identity["passed"],
                total=identity["total"],
                **pooled,
            )
        )

    checks = cohort["identity_checks"]
    lines.extend(
        [
            "",
            f"Count identity checks: {checks['passed']}/{checks['total']} per-cell checks passed.",
            "",
        ]
    )
    pair = cohort["conf055_vs_reid"]
    lines.append(
        f"Confidence 0.55 versus ReID: {pair['strict_inversion_count']} of "
        f"{pair['total_paired_cells']} paired cells were strict inversions. Of these, "
        f"{pair['conf055_lower_absolute_error_reid_higher_recovery_count']} favoured "
        "confidence 0.55 for absolute count error and ReID for recovery; "
        f"{pair['reid_lower_absolute_error_conf055_higher_recovery_count']} had the reverse direction."
    )
    lines.extend(
        [
            "",
            "| Split | Video | conf055 error | ReID error | conf055 recovery | ReID recovery | "
            "Lower absolute error | Higher recovery | Inversion |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for cell in pair["cells"]:
        video = str(cell["video"]).replace("|", "\\|")
        lines.append(
            f"| {cell['split']} | {video} | "
            f"{_number(cell['conf055']['net_signed_error'], signed=True)} | "
            f"{_number(cell['reid']['net_signed_error'], signed=True)} | "
            f"{_number(cell['conf055']['recovered_fraction'])} | "
            f"{_number(cell['reid']['recovered_fraction'])} | "
            f"{cell['lower_absolute_error_arm']} | {cell['higher_recovery_arm']} | "
            f"{'yes' if cell['strict_inversion'] else 'no'} |"
        )
    lines.append("")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    protocol = report["protocol"]
    lines = [
        "# GrapeMOTS out-of-fold configuration analysis",
        "",
        f"Counts use minimum track length {protocol['minimum_track_length']} and matching IoU "
        f"{protocol['match_iou']:.2f}. The {protocol['assignment']} assignment is used for "
        "the per-frame count decomposition.",
        "",
        "Pooled values weight trajectories; macro values are medians across paired split-video cells.",
        "",
    ]
    first_hota = report["cohorts"]["primary"]["arms"][ARMS[0]]["pooled_tracking_metrics"]
    if not first_hota["available"]:
        lines.extend([first_hota["reason"], ""])
    for name in ("primary", "sensitivity"):
        lines.extend(_markdown_cohort(name, report["cohorts"][name]))
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", "--input-dir", dest="results", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--primary-splits", nargs="+", default=list(PRIMARY_SPLITS))
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSON output path; a Markdown report is written beside it with the same stem",
    )
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--min-track-len", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    loaded = load_oof_results(args.results, args.splits, args.primary_splits)
    report = analyse_results(
        loaded,
        match_iou=args.match_iou,
        min_track_len=args.min_track_len,
        splits=args.splits,
        primary_splits=args.primary_splits,
    )
    markdown = render_markdown(report)
    json_out = args.out if args.out.suffix.lower() == ".json" else Path(f"{args.out}.json")
    markdown_out = json_out.with_suffix(".md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")
    failed = [
        cohort
        for cohort, summary in report["cohorts"].items()
        if not summary["identity_checks"]["all_hold"]
    ]
    if failed:
        raise SystemExit(
            "Count identity failed in cohort(s): " + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
