#!/usr/bin/env python3
"""How much work the low-score association stage actually does.

The published arms filter detections at 0.25 before the tracker while the tracker
configuration keeps a low threshold of 0.10, so a reviewer can reasonably ask
whether the second association stage of ByteTrack and BoT-SORT ever receives a
candidate, and whether the arms are therefore the standard trackers at all.
Handing the tracker every box above 0.10 answers the first half; this answers the
second, by counting the assignments each stage returns.

The three calls to linear_assignment inside one update are, in order: high-score
detections against tracked and lost tracks, low-score detections against the
tracks that first call left unmatched, and the remaining detections against
unconfirmed tracks. Wrapping the function and reading the calls in threes
separates them without touching the tracker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.utils import matching  # noqa: E402

from track_grapemots_mot import build_tracker, merge_detections, tiled_raw  # noqa: E402
from fullrate_tracking import annotated_frames, source_video  # noqa: E402

CALLS: list[dict] = []
ORIGINAL = matching.linear_assignment


def counting_assignment(cost_matrix, thresh, use_lap=True):
    matches, u_a, u_b = ORIGINAL(cost_matrix, thresh, use_lap)
    CALLS.append({
        "tracks": int(cost_matrix.shape[0]) if cost_matrix.size else 0,
        "detections": int(cost_matrix.shape[1]) if cost_matrix.size else 0,
        "matches": int(len(matches)),
        "thresh": float(thresh),
    })
    return matches, u_a, u_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tracker", default="cfg/trackers/botsort_gmc.yaml")
    parser.add_argument("--extraction-floor", type=float, default=0.10)
    parser.add_argument("--arm-floor", type=float, nargs="+", default=[0.10, 0.25])
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    matching.linear_assignment = counting_assignment
    model = YOLO(str(args.weights))
    path = source_video(args.video, args.video_root)
    annotations = annotated_frames(args.video, args.root)
    if not annotations:
        raise SystemExit(f"no annotations for {args.video}")

    report = {}
    for floor in args.arm_floor:
        CALLS.clear()
        capture = cv2.VideoCapture(str(path))
        tracker = build_tracker(args.tracker, frame_rate=30)
        high = low = 0
        index = 0
        while index < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            boxes, scores = tiled_raw(
                model, frame, args.imgsz, args.extraction_floor, args.tile, args.stride
            )
            keep = scores >= floor if len(boxes) else np.zeros(0, dtype=bool)
            boxes, scores = merge_detections(boxes[keep], scores[keep], 0.5, "iou")
            high += int((scores >= 0.25).sum()) if len(scores) else 0
            low += int((scores < 0.25).sum()) if len(scores) else 0
            data = (
                np.concatenate(
                    [boxes, scores[:, None], np.zeros((len(boxes), 1), dtype=np.float32)],
                    axis=1,
                )
                if len(boxes)
                else np.empty((0, 6), dtype=np.float32)
            )
            tracker.update(
                Boxes(torch.as_tensor(data, dtype=torch.float32), (height, width)), frame
            )
            index += 1
        capture.release()

        stages = {"first": [], "second": [], "unconfirmed": []}
        for position, call in enumerate(CALLS):
            stages[("first", "second", "unconfirmed")[position % 3]].append(call)
        report[f"floor {floor}"] = {
            "frames": index,
            "detections_at_or_above_0.25": high,
            "detections_below_0.25": low,
            "stages": {
                name: {
                    "calls": len(calls),
                    "calls_with_a_candidate": sum(1 for c in calls if c["detections"]),
                    "candidates_offered": sum(c["detections"] for c in calls),
                    "assignments_returned": sum(c["matches"] for c in calls),
                }
                for name, calls in stages.items()
            },
        }
        block = report[f"floor {floor}"]
        print(f"floor {floor}: {index} frames, {high} detections at 0.25+, {low} below",
              flush=True)
        for name, stage in block["stages"].items():
            print(f"   {name:12} candidates={stage['candidates_offered']:6} "
                  f"assignments={stage['assignments_returned']:6}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "video": args.video,
        "tracker": args.tracker,
        "extraction_floor": args.extraction_floor,
        "arms": report,
    }, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
