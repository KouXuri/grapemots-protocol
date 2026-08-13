#!/usr/bin/env python3
"""Cadence arms that differ in which frames they process and in nothing else.

tools/fullrate_decompose.py contrasts two cadences, both uniform: the released
one, which processes exactly the annotated frames, and the source rate, which
processes all of them. Two questions that contrast cannot answer are asked here,
and both are asked on the same decode and the same detection pass, so no arm ever
sees a pixel or a box another arm did not.

CONTENT-ADAPTIVE SAMPLING. An on-camera filter of the Reducto family spends its
frame budget where the picture changes, not on a fixed step, so uniform thinning
may simply be the wrong sampler to have tested. Arms whose cadence is a named
frame set take their processed frames from tools/adaptive_frame_selection.py,
which builds a uniform and an adaptive set of identical size for each budget.
Every set contains the annotated frames, because those are the scoring instants.

THE SECOND ASSOCIATION STAGE. The published arms filter detections at 0.25 before
the tracker, while the tracker configuration keeps a low threshold of 0.10 for its
second association stage, so that stage never receives a candidate. Detection here
runs once at the lowest floor any arm asks for and each arm applies its own
threshold to the shared boxes, which lets a full-candidate arm run beside the
published one on identical detections.
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


def processed_frames(video: str, cadence: str, annotations: dict[int, Path], args):
    """The frames one arm puts through the tracker, or None for every frame."""
    if cadence == "source":
        return None
    if cadence == "released":
        return set(annotations)
    if not cadence.startswith("set:"):
        raise SystemExit(f"unknown cadence {cadence}")
    name = cadence[4:]
    entry = args.frame_sets.get(video)
    if entry is None:
        raise SystemExit(f"{video}: no frame sets in --frame-sets")
    arm = entry["arms"].get(name)
    if arm is None:
        raise SystemExit(f"{video}: frame set {name} not in --frame-sets")
    chosen = set(int(index) for index in arm["processed"])
    missing = set(annotations) - chosen
    if missing:
        raise SystemExit(f"{video}/{name}: frame set omits {len(missing)} scoring instants")
    return chosen


def run_video(video: str, video_path: Path, weights: Path, model, arms, args) -> list[dict]:
    annotations = resolve_annotations(video, args)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    last = max(annotations)

    state = {}
    for label, tracker_cfg, cadence, det_conf in arms:
        frames = processed_frames(video, cadence, annotations, args)
        if frames is None:
            rate = round(source_fps)
        elif cadence == "released":
            rate = args.released_frame_rate
        else:
            # a set arm runs at its own mean rate, so its buffer keeps counting
            # processed frames while the seconds it buys follow the arm's density
            span = max(1, last)
            rate = max(1, round(source_fps * len(frames) / span))
        state[label] = {
            "tracker": build_tracker(tracker_cfg, frame_rate=max(1, int(rate))),
            "cadence": cadence,
            "tracker_cfg": tracker_cfg,
            "det_conf": det_conf,
            "frames": frames,
            "frame_rate": rate,
            "tracker_frames": 0,
            "detections_high": 0,
            "detections_low": 0,
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
        # merge after each arm's threshold, not before it: a box below an arm's
        # floor must not be able to absorb or suppress one above it, and an arm at
        # 0.25 has then seen exactly what a run extracting at 0.25 would have seen
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
            tracks = arm["tracker"].update(
                Boxes(torch.as_tensor(data, dtype=torch.float32), (height, width)), frame
            )
            arm["tracker_frames"] += 1
            # what the tracker's second association stage had to work with: the
            # published arms filter at 0.25 and leave it empty by construction
            arm["detections_high"] += int((arm_scores >= 0.25).sum()) if len(arm_scores) else 0
            arm["detections_low"] += int((arm_scores < 0.25).sum()) if len(arm_scores) else 0
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
            "tracker_frame_rate": arm["frame_rate"],
            "weights": str(weights),
            "source_fps": source_fps,
            "tracker_frames": arm["tracker_frames"],
            "detections_high": arm["detections_high"],
            "detections_low": arm["detections_low"],
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
            f"{video} {label}: tracker={arm['tracker_frames']}, scored={len(annotations)}, "
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
                "detections_high": sum(
                    r["detections_high"] for r in records if r["arm"] == label
                ),
                "detections_low": sum(
                    r["detections_low"] for r in records if r["arm"] == label
                ),
            }
    return pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-map", type=Path, default=None)
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--frame-offsets", type=Path, default=None)
    parser.add_argument("--frame-sets", type=Path, default=None)
    parser.add_argument(
        "--arm", action="append", default=[],
        help="label:tracker.yaml:{source,released,set:NAME}[:det_conf], repeatable",
    )
    parser.add_argument("--released-frame-rate", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10,
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
    if args.frame_sets:
        args.frame_sets = json.loads(args.frame_sets.read_text())["sequences"]
    else:
        args.frame_sets = {}
    if 1 not in args.taus:
        parser.error("--taus must include 1")
    if not args.arm:
        parser.error("give at least one --arm")
    return args


def main() -> None:
    args = parse_args()
    arms = []
    for spec in args.arm:
        parts = spec.split(":")
        if len(parts) == 3:
            label, tracker_cfg, cadence = parts
            det_conf = 0.25
        elif len(parts) == 4 and parts[2] != "set":
            label, tracker_cfg, cadence, det_conf = parts
            det_conf = float(det_conf)
        elif len(parts) == 4:
            label, tracker_cfg = parts[0], parts[1]
            cadence, det_conf = f"set:{parts[3]}", 0.25
        elif len(parts) == 5:
            label, tracker_cfg = parts[0], parts[1]
            cadence, det_conf = f"set:{parts[3]}", float(parts[4])
        else:
            raise SystemExit(f"cannot parse arm {spec}")
        if det_conf < args.conf:
            raise SystemExit(
                f"arm {label} asks for {det_conf} below the extraction floor {args.conf}"
            )
        arms.append((label, tracker_cfg, cadence, det_conf))

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
              f"holds={one['identity_holds']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.pop("frame_sets", None)
    config.pop("frame_map", None)
    args.out.write_text(
        json.dumps({"config": config, "runs": records, "pooled": pooled}, indent=1) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
