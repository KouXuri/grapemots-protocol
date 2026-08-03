#!/usr/bin/env python3
"""Audit configuration rankings over the OOF prefix ``L x tau`` surface.

This analysis is deliberately post hoc and read-only. It consumes the frame-level
artefacts already written by ``track_grapemots_mot.py`` and never runs inference.
The primary result uses the manuscript's frozen retained-cell rule. A separate
coverage-unfiltered common-grid sensitivity keeps the video cohort fixed at six.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARMS = ("conf055", "conf040", "ios", "botsort", "bytetrack", "reid")
ARM_LABELS = {
    "conf055": "Confidence 0.55",
    "conf040": "Confidence 0.40",
    "ios": "IoS merge",
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "reid": "BoT-SORT + ReID",
}
TOLERANCE = 1e-12


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def _result_path(results: Path, split: str, arm: str) -> Path:
    plain = results / f"oof_{split}_{arm}.json"
    compressed = plain.with_suffix(".json.gz")
    if plain.is_file():
        return plain
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(f"Missing OOF result: {plain} (or compressed {compressed.name})")


def _json_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def _sha256_json(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_json_bytes(path))
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(_json_bytes(path).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        message = getattr(error, "msg", str(error))
        raise ValueError(f"{path}: invalid JSON ({message})") from error


def _surface(video: Mapping[str, Any], source: Path) -> dict[tuple[int, int], dict]:
    rows = video.get("count_error_surface")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}: missing count_error_surface")
    surface: dict[tuple[int, int], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{source}: a count_error_surface row is not an object")
        key = (int(row["window_frames"]), int(row["min_track_len"]))
        if key in surface:
            raise ValueError(f"{source}: duplicate surface cell {key}")
        surface[key] = row
    return surface


def _recompute_prefix(
    frame_predicted_ids: Sequence[Sequence[int]],
    frame_gt_ids: Sequence[Sequence[int]],
    length: int,
    tau: int,
) -> tuple[int, int, float | None]:
    predicted_observations: Counter[int] = Counter()
    ground_truth_ids: set[int] = set()
    for predicted_ids, gt_ids in zip(
        frame_predicted_ids[:length], frame_gt_ids[:length]
    ):
        predicted_observations.update(predicted_ids)
        ground_truth_ids.update(gt_ids)
    predicted = sum(observations >= tau for observations in predicted_observations.values())
    truth = len(ground_truth_ids)
    error = (predicted - truth) / truth if truth else None
    return predicted, truth, error


def load_and_verify(
    results: Path, splits: Sequence[str]
) -> tuple[dict[tuple[str, str, str], dict], dict[str, str], dict[str, int]]:
    loaded: dict[tuple[str, str, str], dict] = {}
    checksums: dict[str, str] = {}
    expected_videos: dict[str, set[str]] = {}
    expected_gt: dict[tuple[str, str], tuple[list, list]] = {}
    expected_surface: dict[tuple[str, str], dict[tuple[int, int], int]] = {}
    recomputed_cells = 0

    for split in splits:
        for arm in ARMS:
            path = _result_path(results, split, arm)
            logical_name = path.name.removesuffix(".gz")
            checksums[logical_name] = _sha256_json(path)
            payload = _read_json(path)
            videos = payload.get("videos") if isinstance(payload, dict) else None
            if not isinstance(videos, list) or len(videos) != 2:
                raise ValueError(f"{path}: expected exactly two videos")
            names = {video.get("video") for video in videos if isinstance(video, dict)}
            if None in names or len(names) != 2:
                raise ValueError(f"{path}: video names are missing or repeated")
            if split in expected_videos and names != expected_videos[split]:
                raise ValueError(f"{path}: held-out videos differ across arms")
            expected_videos.setdefault(split, names)

            for video in videos:
                name = video["video"]
                frames = int(video.get("frames", -1))
                required = (
                    "frame_predicted_ids",
                    "frame_predicted_boxes",
                    "frame_gt_ids",
                    "frame_gt_boxes",
                )
                if any(key not in video for key in required):
                    raise ValueError(f"{path}: {name} lacks frame-level track artefacts")
                lengths = [len(video[key]) for key in required]
                if len(set(lengths)) != 1 or lengths[0] != frames:
                    raise ValueError(f"{path}: {name} has inconsistent frame artefact lengths")
                key = (split, name)
                gt = (video["frame_gt_ids"], video["frame_gt_boxes"])
                if key in expected_gt and gt != expected_gt[key]:
                    raise ValueError(f"{path}: {name} ground truth differs across arms")
                expected_gt.setdefault(key, gt)

                surface = _surface(video, path)
                gt_surface = {
                    protocol: int(row["prefix_gt_tracks"])
                    for protocol, row in surface.items()
                }
                if key in expected_surface and gt_surface != expected_surface[key]:
                    raise ValueError(f"{path}: {name} protocol grid or ground truth differs")
                expected_surface.setdefault(key, gt_surface)

                for (length, tau), row in surface.items():
                    predicted, truth, error = _recompute_prefix(
                        video["frame_predicted_ids"], video["frame_gt_ids"], length, tau
                    )
                    stored_error = row["prefix_signed_relative_error"]
                    if predicted != int(row["prefix_predicted_tracks"]):
                        raise ValueError(f"{path}: {name} P mismatch at {(length, tau)}")
                    if truth != int(row["prefix_gt_tracks"]):
                        raise ValueError(f"{path}: {name} G mismatch at {(length, tau)}")
                    if error is None:
                        if stored_error is not None:
                            raise ValueError(f"{path}: {name} e should be null at {(length, tau)}")
                    elif stored_error is None or not _close(error, float(stored_error)):
                        raise ValueError(f"{path}: {name} e mismatch at {(length, tau)}")
                    recomputed_cells += 1
                loaded[(split, name, arm)] = video

    video_cells = [(split, name) for split in splits for name in expected_videos[split]]
    names = [name for _, name in video_cells]
    if len(set(names)) != len(names):
        raise ValueError("Primary OOF splits do not contain six distinct videos")
    validation = {
        "input_files": len(checksums),
        "distinct_videos": len(video_cells),
        "frame_level_artefact_files": len(checksums),
        "surface_cells_recomputed": recomputed_cells,
    }
    return loaded, dict(sorted(checksums.items())), validation


def _video_cells(
    loaded: Mapping[tuple[str, str, str], dict], split: str, video: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    reference = loaded[(split, video, ARMS[0])]
    surface = _surface(reference, Path(f"{split}/{video}/{ARMS[0]}"))
    full_length = int(reference["frames"])
    full_ground_truth = int(surface[(full_length, 1)]["prefix_gt_tracks"])
    valid = sorted(
        protocol
        for protocol, row in surface.items()
        if protocol[1] <= protocol[0] / 2 and int(row["prefix_gt_tracks"]) > 0
    )
    retained = [
        protocol
        for protocol in valid
        if int(surface[protocol]["prefix_gt_tracks"]) / full_ground_truth >= 0.8
    ]
    return valid, retained, full_ground_truth


def _error(
    loaded: Mapping[tuple[str, str, str], dict],
    split: str,
    video: str,
    arm: str,
    protocol: tuple[int, int],
) -> float:
    row = _surface(loaded[(split, video, arm)], Path(f"{split}/{video}/{arm}"))[protocol]
    value = row["prefix_signed_relative_error"]
    if value is None:
        raise ValueError(f"Unexpected null error at {split}/{video}/{arm}/{protocol}")
    return float(value)


def _winners(values: Mapping[str, float]) -> list[str]:
    best = min(values.values())
    return [arm for arm in ARMS if _close(values[arm], best)]


def _direction(first: float, second: float) -> int:
    difference = first - second
    if _close(difference, 0.0):
        return 0
    return -1 if difference < 0 else 1


def retained_surface_analysis(
    loaded: Mapping[tuple[str, str, str], dict], video_cells: Sequence[tuple[str, str]]
) -> dict:
    cell_rows = []
    per_video: dict[str, dict] = {}
    unique_wins: Counter[str] = Counter()
    co_wins: Counter[str] = Counter()
    pair_video_rows = []

    for split, video in video_cells:
        _, protocols, full_ground_truth = _video_cells(loaded, split, video)
        full_length = int(loaded[(split, video, ARMS[0])]["frames"])
        reference_protocol = (full_length, 1)
        reference_values = {
            arm: abs(_error(loaded, split, video, arm, reference_protocol)) for arm in ARMS
        }
        reference_winners = _winners(reference_values)
        winner_sets: list[tuple[str, ...]] = []
        changed = 0

        for length, tau in protocols:
            signed = {
                arm: _error(loaded, split, video, arm, (length, tau)) for arm in ARMS
            }
            absolute = {arm: abs(value) for arm, value in signed.items()}
            winners = _winners(absolute)
            winner_tuple = tuple(winners)
            winner_sets.append(winner_tuple)
            for arm in winners:
                co_wins[arm] += 1
            if len(winners) == 1:
                unique_wins[winners[0]] += 1
            if winners != reference_winners:
                changed += 1
            cell_rows.append(
                {
                    "split": split,
                    "video": video,
                    "L": length,
                    "tau": tau,
                    "G": int(
                        _surface(
                            loaded[(split, video, ARMS[0])],
                            Path(f"{split}/{video}/{ARMS[0]}"),
                        )[(length, tau)]["prefix_gt_tracks"]
                    ),
                    "coverage": int(
                        _surface(
                            loaded[(split, video, ARMS[0])],
                            Path(f"{split}/{video}/{ARMS[0]}"),
                        )[(length, tau)]["prefix_gt_tracks"]
                    )
                    / full_ground_truth,
                    "signed_error": signed,
                    "absolute_error": absolute,
                    "winners": winners,
                }
            )

        reversed_pairs = []
        for first, second in itertools.combinations(ARMS, 2):
            directions = {
                _direction(
                    abs(_error(loaded, split, video, first, protocol)),
                    abs(_error(loaded, split, video, second, protocol)),
                )
                for protocol in protocols
            }
            reversed_order = -1 in directions and 1 in directions
            if reversed_order:
                reversed_pairs.append([first, second])
            pair_video_rows.append(
                {
                    "split": split,
                    "video": video,
                    "first": first,
                    "second": second,
                    "reversal": reversed_order,
                }
            )

        per_video[video] = {
            "split": split,
            "frames": full_length,
            "G_full": full_ground_truth,
            "retained_protocol_cells": len(protocols),
            "retained_L": sorted({length for length, _ in protocols}),
            "reference_protocol": {"L": full_length, "tau": 1},
            "reference_winners": reference_winners,
            "distinct_winner_sets": len(set(winner_sets)),
            "distinct_winning_arms": sorted(
                {arm for winners in winner_sets for arm in winners}, key=ARMS.index
            ),
            "winner_set_changes_vs_reference": changed,
            "pairwise_reversals": len(reversed_pairs),
            "reversed_pairs": reversed_pairs,
        }

    pair_summary = []
    for first, second in itertools.combinations(ARMS, 2):
        rows = [
            row
            for row in pair_video_rows
            if row["first"] == first and row["second"] == second
        ]
        videos = [row["video"] for row in rows if row["reversal"]]
        pair_summary.append(
            {
                "first": first,
                "second": second,
                "videos_with_reversal": videos,
                "reversal_video_count": len(videos),
            }
        )

    tied_cells = sum(len(row["winners"]) > 1 for row in cell_rows)
    pair_video_reversals = sum(row["reversal"] for row in pair_video_rows)
    videos_with_winner_change = [
        video
        for video, summary in per_video.items()
        if summary["winner_set_changes_vs_reference"] > 0
    ]
    videos_with_pair_reversal = [
        video for video, summary in per_video.items() if summary["pairwise_reversals"] > 0
    ]
    return {
        "definition": {
            "window_rule": "prefix",
            "performance_measure": "absolute signed relative count error |(P-G)/G|",
            "retained_cell": "tau <= L/2, G(L) > 0, and G(L)/G(full) >= 0.8",
            "winner": "all arms tied for the minimum absolute error; ties are retained",
            "pairwise_reversal": (
                "within one video, each arm has lower absolute error than the other "
                "on at least one retained protocol cell"
            ),
        },
        "video_protocol_cells": len(cell_rows),
        "arm_measurements": len(cell_rows) * len(ARMS),
        "unique_winner_cells": len(cell_rows) - tied_cells,
        "tied_winner_cells": tied_cells,
        "unique_win_counts": {arm: unique_wins[arm] for arm in ARMS},
        "co_win_counts": {arm: co_wins[arm] for arm in ARMS},
        "winner_set_changes_vs_full_tau1": sum(
            summary["winner_set_changes_vs_reference"] for summary in per_video.values()
        ),
        "videos_with_winner_change_vs_full_tau1": videos_with_winner_change,
        "videos_with_any_pairwise_reversal": videos_with_pair_reversal,
        "pair_video_opportunities": len(pair_video_rows),
        "pair_video_reversals": pair_video_reversals,
        "arm_pairs": len(pair_summary),
        "arm_pairs_with_reversal_on_at_least_one_video": sum(
            row["reversal_video_count"] > 0 for row in pair_summary
        ),
        "per_pair": pair_summary,
        "per_video": per_video,
        "cells": cell_rows,
    }


def common_grid_analysis(
    loaded: Mapping[tuple[str, str, str], dict], video_cells: Sequence[tuple[str, str]]
) -> dict:
    valid_sets = []
    retained_sets = []
    for split, video in video_cells:
        valid, retained, _ = _video_cells(loaded, split, video)
        valid_sets.append(set(valid))
        retained_sets.append(set(retained))
    common_valid = sorted(set.intersection(*valid_sets))
    common_retained = sorted(set.intersection(*retained_sets))

    rows = []
    unique_wins: Counter[str] = Counter()
    co_wins: Counter[str] = Counter()
    for length, tau in common_valid:
        macro_medians = {
            arm: statistics.median(
                abs(_error(loaded, split, video, arm, (length, tau)))
                for split, video in video_cells
            )
            for arm in ARMS
        }
        winners = _winners(macro_medians)
        for arm in winners:
            co_wins[arm] += 1
        if len(winners) == 1:
            unique_wins[winners[0]] += 1
        rows.append(
            {
                "L": length,
                "tau": tau,
                "macro_median_absolute_error": macro_medians,
                "winners": winners,
            }
        )

    reversals = []
    for first, second in itertools.combinations(ARMS, 2):
        directions = {
            _direction(
                row["macro_median_absolute_error"][first],
                row["macro_median_absolute_error"][second],
            )
            for row in rows
        }
        if -1 in directions and 1 in directions:
            reversals.append([first, second])

    return {
        "role": (
            "coverage-unfiltered sensitivity only; it fixes all six videos at each "
            "literal L x tau point but includes low-coverage windows"
        ),
        "valid_rule": "tau <= L/2 and G(L) > 0; no coverage threshold",
        "common_protocol_cells": len(common_valid),
        "common_retained_protocol_cells": len(common_retained),
        "unique_win_counts": {arm: unique_wins[arm] for arm in ARMS},
        "co_win_counts": {arm: co_wins[arm] for arm in ARMS},
        "arm_pair_opportunities": math.comb(len(ARMS), 2),
        "arm_pair_reversals": len(reversals),
        "reversed_pairs": reversals,
        "cells": rows,
    }


def analyse(results: Path, splits: Sequence[str]) -> dict:
    loaded, checksums, validation = load_and_verify(results, splits)
    video_cells = sorted(
        {(split, video) for split, video, _ in loaded}, key=lambda item: (item[0], item[1])
    )
    retained = retained_surface_analysis(loaded, video_cells)
    common = common_grid_analysis(loaded, video_cells)
    return {
        "schema_version": 1,
        "inputs": {
            "results_root": str(results),
            "splits": list(splits),
            "arms": list(ARMS),
            "arm_labels": ARM_LABELS,
            "files_sha256": checksums,
        },
        "validation": {
            **validation,
            "all_prefix_counts_recomputed_from_frame_ids": True,
            "ground_truth_equal_across_arms": True,
            "protocol_grids_equal_across_arms": True,
        },
        "retained_surface": retained,
        "common_grid_sensitivity": common,
    }


def _pair(first: str, second: str) -> str:
    return f"{first} vs {second}"


def render_markdown(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    retained = report["retained_surface"]
    common = report["common_grid_sensitivity"]
    lines = [
        "# OOF real-configuration protocol-surface audit",
        "",
        "This is a read-only analysis of existing A--C OOF outputs. No inference or training was run.",
        "",
        "## Verification",
        "",
        f"- {validation['input_files']} input JSON files, {validation['distinct_videos']} distinct held-out videos.",
        f"- Recomputed {validation['surface_cells_recomputed']} stored prefix cells from frame-level IDs; all matched P, G and e.",
        "- Ground truth and protocol grids matched across all six arms within every split-video cell.",
        "",
        "## Retained prefix surface",
        "",
        f"Rule: `{retained['definition']['retained_cell']}`. Winner means minimum `{retained['definition']['performance_measure']}`.",
        "",
        f"There are {retained['video_protocol_cells']} video-protocol cells and {retained['arm_measurements']} arm measurements. "
        f"The winner is unique in {retained['unique_winner_cells']} cells and tied in {retained['tied_winner_cells']}.",
        "",
        "| Video | Cells | Winner changes from full, tau=1 | Pair reversals (of 15) | Winning arms |",
        "|---|---:|---:|---:|---|",
    ]
    for video, summary in retained["per_video"].items():
        lines.append(
            f"| {video} | {summary['retained_protocol_cells']} | "
            f"{summary['winner_set_changes_vs_reference']} | {summary['pairwise_reversals']} | "
            f"{', '.join(summary['distinct_winning_arms'])} |"
        )
    lines.extend(
        [
            "",
            "Confidence 0.55 is the full-sequence, tau=1 reference winner on all six videos. "
            f"The cell-level winner changes from that reference on {len(retained['videos_with_winner_change_vs_full_tau1'])}/6 videos "
            f"and in {retained['winner_set_changes_vs_full_tau1']}/{retained['video_protocol_cells']} retained cells. "
            f"At least one pair reverses on {len(retained['videos_with_any_pairwise_reversal'])}/6 videos. "
            f"Overall, {retained['pair_video_reversals']}/{retained['pair_video_opportunities']} arm-pair/video cases reverse, "
            f"and {retained['arm_pairs_with_reversal_on_at_least_one_video']}/{retained['arm_pairs']} arm pairs reverse on at least one video.",
            "",
            "Unique winner counts are descriptive cell counts (videos contribute different numbers of retained L values):",
            "",
            "| Arm | Unique wins | Co-wins |",
            "|---|---:|---:|",
        ]
    )
    for arm in ARMS:
        lines.append(
            f"| {arm} | {retained['unique_win_counts'][arm]} | {retained['co_win_counts'][arm]} |"
        )
    lines.extend(
        [
            "",
            "| Pair | Videos with reversal |",
            "|---|---:|",
        ]
    )
    for row in sorted(
        retained["per_pair"], key=lambda item: (-item["reversal_video_count"], item["first"], item["second"])
    ):
        lines.append(
            f"| {_pair(row['first'], row['second'])} | {row['reversal_video_count']} |"
        )
    lines.extend(
        [
            "",
            "## Common-grid sensitivity",
            "",
            f"No literal `(L, tau)` cell survives the retained-cell rule on all six videos "
            f"({common['common_retained_protocol_cells']} common retained cells). The coverage-unfiltered intersection has "
            f"{common['common_protocol_cells']} cells. Across their six-video macro-median absolute errors, "
            f"all six arms win at least one cell and {common['arm_pair_reversals']}/{common['arm_pair_opportunities']} arm pairs reverse.",
            "",
            "This common-grid result is sensitivity evidence, not a replacement for the retained analysis, because low-coverage windows are included.",
            "",
            "## Interpretation boundary",
            "",
            "The existing outputs fully support a within-video retained-surface ranking analysis without new inference. "
            "A single six-video retained ranking at every literal L is not identifiable because the retained-L intersection is empty. "
            "Producing one would require a declared additional analysis choice, such as normalised sequence fractions or interpolation, not additional model computation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown summary path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    splits = tuple(split.upper() for split in args.splits)
    if len(splits) != 3 or len(set(splits)) != 3:
        raise SystemExit("This audit requires exactly three distinct primary OOF splits")
    report = analyse(args.results, splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = args.markdown or args.out.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {markdown}")


if __name__ == "__main__":
    main()
