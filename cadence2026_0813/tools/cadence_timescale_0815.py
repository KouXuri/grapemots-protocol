#!/usr/bin/env python3
"""Does the cadence effect survive when the motion model is told the real time?

tools/cadence_arms_0813.py contrasts a released-cadence arm with a source-rate
arm on one decode and one detection pass. A reviewer asked what else that
contrast moves: an implementation advances its Kalman filter once per processed
frame, so the sparse arm treats a displacement of 36 source frames as one time
step, and the comparison changes the temporal scale of the motion model along
with the number of observations.

This script adds the missing control. A catch-up arm predicts (gap - 1) extra
times before each update, where gap is the real interval in source frames, so
its filter advances by the elapsed time rather than by one step per observation.
Everything else is the arm it is compared with: same decode, same detections,
same tracker configuration, same buffer in processed frames, same scoring
instants. Global motion compensation already works on real displacement, since
it is estimated between the frames the arm actually sees.

A second family of arms varies the association gate (match_thresh) instead, so
the sensitivity of the effect to the IoU gate is measured rather than argued.

Arm syntax, comma-separated key=value, repeatable:

    --arm label=rel_dt,cfg=cfg/trackers/botsort_gmc.yaml,cadence=released,catchup=1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402

from track_grapemots_mot import (  # noqa: E402
    build_tracker,
    load_gt_tracks,
    merge_detections,
    tiled_raw,
)
from fullrate_tracking import (  # noqa: E402
    annotated_frames,
    read_weights_map,
    source_video,
    surface,
)
from decompose_count_error import decompose  # noqa: E402


def resolve_annotations(video: str, args) -> dict[int, Path]:
    """Annotated frame number -> label file, in source-video numbering."""
    annotations = annotated_frames(video, args.root)
    if not annotations:
        raise ValueError(f"No annotations found for {video} under {args.root}")
    offset = int(args.frame_offsets.get(video, 0))
    if offset:
        annotations = {number - offset: path for number, path in annotations.items()}
        if min(annotations) < 0:
            raise ValueError(f"{video}: offset {offset} pushes annotations before frame 0")
    mapping = args.frame_map.get(video)
    if mapping:
        rekeyed = {}
        for number, path in annotations.items():
            source = mapping.get(str(number), mapping.get(number))
            if source is None:
                raise ValueError(f"{video}: annotated frame {number} is not in the frame map")
            rekeyed[int(source)] = path
        if len(rekeyed) != len(annotations):
            raise ValueError(f"{video}: frame map is not injective")
        annotations = rekeyed
    return dict(sorted(annotations.items()))


def processed_frames(cadence: str, annotations: dict[int, Path]):
    """The frames one arm puts through the tracker, or None for every frame."""
    if cadence == "source":
        return None
    if cadence == "released":
        return set(annotations)
    raise SystemExit(f"unknown cadence {cadence}")


def catch_up(tracker, steps: int) -> int:
    """Advance every live track's motion model by `steps` extra time steps.

    The tracker predicts once inside update(), so a gap of g source frames needs
    g - 1 predictions here for the filter to have advanced by the elapsed time.
    Lost tracks are included because they are what a re-acquisition matches
    against, and they are the tracks a long gap is most likely to strand.
    """
    if steps <= 0:
        return 0
    pool = list(tracker.tracked_stracks) + list(tracker.lost_stracks)
    if not pool:
        return 0
    for _ in range(steps):
        tracker.multi_predict(pool)
    return steps * len(pool)


def run_video(video: str, video_path: Path, weights: Path, model, arms, args) -> list[dict]:
    annotations = resolve_annotations(video, args)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    last = max(annotations)

    state = {}
    for arm_spec in arms:
        frames = processed_frames(arm_spec["cadence"], annotations)
        rate = round(source_fps) if frames is None else args.released_frame_rate
        state[arm_spec["label"]] = {
            "tracker": build_tracker(arm_spec["cfg"], frame_rate=max(1, int(rate))),
            "cadence": arm_spec["cadence"],
            "tracker_cfg": arm_spec["cfg"],
            "det_conf": arm_spec["conf"],
            "catchup": arm_spec["catchup"],
            "frames": frames,
            "frame_rate": rate,
            "tracker_frames": 0,
            "kalman_steps": 0,
            "extra_predictions": 0,
            "previous_source_index": None,
            "frame_predicted_ids": [],
            "frame_predicted_boxes": [],
            "frame_gt_ids": [],
            "frame_gt_boxes": [],
            "frame_names": [],
        }

    everything = any(arm["frames"] is None for arm in state.values())
    wanted: set[int] = set()
    for arm in state.values():
        if arm["frames"] is not None:
            wanted |= arm["frames"]

    source_index = 0
    started = time.time()
    while source_index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        scored = source_index in annotations
        if not everything and source_index not in wanted:
            source_index += 1
            continue
        height, width = frame.shape[:2]
        boxes, scores = tiled_raw(model, frame, args.imgsz, args.conf, args.tile, args.stride)
        # merge after each arm's threshold, not before it, so a box below an
        # arm's floor cannot absorb or suppress one above it
        merged: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        ground_truth = None
        if scored:
            gt_ids, gt_boxes = load_gt_tracks(annotations[source_index], width, height)
            ground_truth = (list(gt_ids), np.asarray(gt_boxes, dtype=float).tolist())

        for label, arm in state.items():
            if arm["frames"] is not None and source_index not in arm["frames"]:
                continue
            floor = arm["det_conf"]
            if floor not in merged:
                keep = scores >= floor if len(boxes) else np.zeros(0, dtype=bool)
                merged[floor] = merge_detections(
                    boxes[keep], scores[keep], args.merge_threshold, args.merge_metric
                )
            arm_boxes, arm_scores = merged[floor]
            data = (
                np.concatenate(
                    [arm_boxes, arm_scores[:, None],
                     np.zeros((len(arm_boxes), 1), dtype=np.float32)], axis=1
                )
                if len(arm_boxes)
                else np.empty((0, 6), dtype=np.float32)
            )
            gap = 1
            if arm["previous_source_index"] is not None:
                gap = source_index - arm["previous_source_index"]
            if arm["catchup"]:
                arm["extra_predictions"] += catch_up(arm["tracker"], gap - 1)
                arm["kalman_steps"] += gap
            else:
                arm["kalman_steps"] += 1
            arm["previous_source_index"] = source_index
            tracks = arm["tracker"].update(
                Boxes(torch.as_tensor(data, dtype=torch.float32), (height, width)), frame
            )
            arm["tracker_frames"] += 1
            if scored:
                if len(tracks):
                    arm["frame_predicted_ids"].append([int(t) for t in tracks[:, 4]])
                    arm["frame_predicted_boxes"].append(
                        np.asarray(tracks[:, :4], dtype=float).tolist()
                    )
                else:
                    arm["frame_predicted_ids"].append([])
                    arm["frame_predicted_boxes"].append([])
                arm["frame_gt_ids"].append(ground_truth[0])
                arm["frame_gt_boxes"].append(ground_truth[1])
                arm["frame_names"].append(annotations[source_index].name)
        source_index += 1
    capture.release()

    records = []
    for label, arm in state.items():
        if len(arm["frame_predicted_ids"]) != len(annotations):
            raise RuntimeError(
                f"{video} {label}: scored {len(arm['frame_predicted_ids'])} of {len(annotations)}"
            )
        stub = {
            "video": video,
            "frame_predicted_ids": arm["frame_predicted_ids"],
            "frame_predicted_boxes": arm["frame_predicted_boxes"],
            "frame_gt_ids": arm["frame_gt_ids"],
            "frame_gt_boxes": arm["frame_gt_boxes"],
        }
        record = {
            "video": video,
            "arm": label,
            "cadence": arm["cadence"],
            "tracker_cfg": arm["tracker_cfg"],
            "det_conf": arm["det_conf"],
            "catchup": arm["catchup"],
            "tracker_frame_rate": arm["frame_rate"],
            "weights": str(weights),
            "source_fps": source_fps,
            "tracker_frames": arm["tracker_frames"],
            "kalman_steps": arm["kalman_steps"],
            "extra_predictions": arm["extra_predictions"],
            "scored_frames": len(annotations),
            "frame_predicted_ids": arm["frame_predicted_ids"],
            "frame_predicted_boxes": arm["frame_predicted_boxes"],
            "frame_gt_ids": arm["frame_gt_ids"],
            "frame_gt_boxes": arm["frame_gt_boxes"],
            "frame_names": arm["frame_names"],
            "count_error_surface": surface(
                arm["frame_predicted_ids"], arm["frame_gt_ids"], [len(annotations)], args.taus
            ),
            "decomposition": {
                str(tau): decompose(stub, args.match_iou, tau) for tau in args.taus
            },
        }
        records.append(record)
        one = record["decomposition"]["1"]
        print(
            f"{video} {label}: tracker={arm['tracker_frames']}, kalman={arm['kalman_steps']}, "
            f"P={one['P']} G={one['G']} U={one['U']} D={one['D']} M={one['M']} "
            f"e={one['signed_error']:+.4f} holds={one['identity_holds']}",
            flush=True,
        )
    print(f"  {video}: {time.time() - started:.0f}s", flush=True)
    return records


def pool(records: list[dict], taus) -> dict:
    pooled = {}
    for label in sorted({r["arm"] for r in records}):
        pooled[label] = {}
        for tau in taus:
            terms = Counter()
            for record in records:
                if record["arm"] != label:
                    continue
                one = record["decomposition"][str(tau)]
                for key in ("P", "G", "U", "D", "M"):
                    terms[key] += one[key]
            G = terms["G"]
            pooled[label][str(tau)] = {
                "videos": sum(1 for r in records if r["arm"] == label),
                "P": terms["P"], "G": G, "U": terms["U"], "D": terms["D"], "M": terms["M"],
                "assigned_fraction": 1 - terms["M"] / G if G else None,
                "signed_error": (terms["P"] - G) / G if G else None,
                "identity_holds": (terms["U"] + terms["D"] - terms["M"]) == (terms["P"] - G),
                "tracker_frames": sum(
                    r["tracker_frames"] for r in records if r["arm"] == label
                ),
                "kalman_steps": sum(
                    r["kalman_steps"] for r in records if r["arm"] == label
                ),
            }
    return pooled


def parse_arm(spec: str) -> dict:
    fields = {}
    for piece in spec.split(","):
        if "=" not in piece:
            raise SystemExit(f"cannot parse arm field {piece!r} in {spec!r}")
        key, value = piece.split("=", 1)
        fields[key.strip()] = value.strip()
    missing = {"label", "cfg", "cadence"} - set(fields)
    if missing:
        raise SystemExit(f"arm {spec!r} is missing {sorted(missing)}")
    return {
        "label": fields["label"],
        "cfg": fields["cfg"],
        "cadence": fields["cadence"],
        "conf": float(fields.get("conf", 0.25)),
        "catchup": bool(int(fields.get("catchup", 0))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-map", type=Path, default=None)
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--frame-offsets", type=Path, default=None)
    parser.add_argument("--arm", action="append", default=[],
                        help="key=value,... with label, cfg, cadence, conf, catchup")
    parser.add_argument("--released-frame-rate", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="detector extraction floor, shared by every arm")
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--merge-metric", choices=["iou", "ios"], default="iou")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--taus", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.frame_offsets = json.loads(args.frame_offsets.read_text()) if args.frame_offsets else {}
    if args.frame_map:
        report = json.loads(args.frame_map.read_text())
        args.frame_map = {
            video: entry["annotated_to_source"]
            for video, entry in report.get("sequences", {}).items()
            if entry.get("status") == "aligned" and "annotated_to_source" in entry
        }
    else:
        args.frame_map = {}
    if 1 not in args.taus:
        parser.error("--taus must include 1")
    if not args.arm:
        parser.error("give at least one --arm")
    return args


def main() -> None:
    args = parse_args()
    arms = [parse_arm(spec) for spec in args.arm]
    for arm in arms:
        if arm["conf"] < args.conf:
            raise SystemExit(
                f"arm {arm['label']} asks for {arm['conf']} below the floor {args.conf}"
            )

    weights_by_video = read_weights_map(args.weights_map)
    model_cache = {}
    records = []
    for video in args.videos:
        weights = weights_by_video.get(video, args.weights)
        if not weights.is_file():
            raise SystemExit(f"Missing checkpoint for {video}: {weights}")
        key = str(weights.resolve())
        if key not in model_cache:
            model_cache[key] = YOLO(key)
        records.extend(
            run_video(video, source_video(video, args.video_root), weights,
                      model_cache[key], arms, args)
        )

    pooled = pool(records, args.taus)
    for label, cells in pooled.items():
        one = cells["1"]
        print(f"POOLED {label}: P={one['P']} G={one['G']} U={one['U']} D={one['D']} "
              f"M={one['M']} assigned={one['assigned_fraction']:.4f} "
              f"e={one['signed_error']:+.4f} frames={one['tracker_frames']} "
              f"kalman={one['kalman_steps']} holds={one['identity_holds']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.pop("frame_map", None)
    args.out.write_text(
        json.dumps({"config": config, "runs": records, "pooled": pooled}, indent=1) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
