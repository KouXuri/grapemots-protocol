#!/usr/bin/env python3
"""U, D and M for both arms of the cadence intervention, from one detection pass.

The published contrast reports P and e for the released and source-rate arms; the
decomposition that explains them was only available for the thinning ladder and
the confidence sweep, on the other corpus. This closes that gap on the causal
experiment itself.

One decode of the video and one tiled detection per frame feed every arm, so the
arms share their pixels and their detections exactly and differ only in which
frames reach the tracker and in how long it keeps a lost track. That is stricter
than the published pair, where the released arm read the release's own frames
while the source-rate arm read the MP4.

Per-frame predicted and reference boxes are kept at the annotated instants, so
the ownership analysis of tools/decompose_count_error.py applies unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
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


def run_video(video: str, video_path: Path, weights: Path, model, arms, args) -> list[dict]:
    annotations = resolve_annotations(video, args)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    last = max(annotations)

    state = {}
    for label, tracker_cfg, cadence in arms:
        rate = round(source_fps) if cadence == "source" else args.released_frame_rate
        state[label] = {
            "tracker": build_tracker(tracker_cfg, frame_rate=max(1, int(rate))),
            "cadence": cadence,
            "tracker_cfg": tracker_cfg,
            "tracker_frames": 0,
            "frame_predicted_ids": [],
            "frame_predicted_boxes": [],
            "frame_gt_ids": [],
            "frame_gt_boxes": [],
            "frame_names": [],
        }

    source_index = 0
    started = time.time()
    while source_index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        scored = source_index in annotations
        # A released-only run needs the detector on the annotated frames alone.
        if not scored and not any(a["cadence"] == "source" for a in state.values()):
            source_index += 1
            continue
        height, width = frame.shape[:2]
        boxes, scores = tiled_raw(model, frame, args.imgsz, args.conf, args.tile, args.stride)
        boxes, scores = merge_detections(boxes, scores, args.merge_threshold, args.merge_metric)
        data = (
            np.concatenate(
                [boxes, scores[:, None], np.zeros((len(boxes), 1), dtype=np.float32)], axis=1
            )
            if len(boxes)
            else np.empty((0, 6), dtype=np.float32)
        )
        ground_truth = None
        if scored:
            gt_ids, gt_boxes = load_gt_tracks(annotations[source_index], width, height)
            ground_truth = (list(gt_ids), np.asarray(gt_boxes, dtype=float).tolist())

        for label, arm in state.items():
            if arm["cadence"] == "released" and not scored:
                continue
            # a fresh Boxes per arm: tracker.update must not see another arm's state
            tracks = arm["tracker"].update(
                Boxes(torch.as_tensor(data.copy(), dtype=torch.float32), (height, width)), frame
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
            "weights": str(weights),
            "source_fps": source_fps,
            "tracker_frames": arm["tracker_frames"],
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
    parser.add_argument(
        "--arm", action="append", default=[],
        help="label:tracker.yaml:{source,released}, repeatable",
    )
    parser.add_argument("--released-frame-rate", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
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
        parser.error("give at least one --arm label:tracker.yaml:cadence")
    return args


def main() -> None:
    args = parse_args()
    arms = []
    for spec in args.arm:
        label, tracker_cfg, cadence = spec.split(":")
        if cadence not in ("source", "released"):
            raise SystemExit(f"cadence must be source or released, not {cadence}")
        arms.append((label, tracker_cfg, cadence))

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
              f"e={one['signed_error']:+.4f} holds={one['identity_holds']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.out.write_text(
        json.dumps({"config": config, "runs": records, "pooled": pooled}, indent=1) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
