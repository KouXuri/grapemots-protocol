#!/usr/bin/env python3
"""Which frames a content-adaptive sampler would process, at a fixed frame budget.

Every cadence in this paper so far is uniform: the released cadence, the source
rate, the thinning ladder and the 15/30 Hz control all keep frames on a fixed
step. On-camera filters do not work that way. Reducto and its relatives compute a
cheap frame-difference feature and skip whatever the feature calls redundant, so
a natural objection to the cadence result is that adaptive sampling escapes it.

This produces the frame sets that objection needs, and produces them so the arms
are comparable. Every arm processes the annotated frames, because those are the
scoring instants and an arm that skipped one could not be read there. The budget
is what is spent between them: for a multiplier m, the uniform arm inserts m-1
equally spaced frames into each gap, and the adaptive arm is given exactly as
many extra frames as the uniform arm actually placed, then chooses them itself by
thresholding the mean absolute difference against the last frame it processed.
The threshold is bisected per sequence until the budget is met, so the two arms
differ in which frames they spend, never in how many.

The feature is the pixel-difference feature of Reducto, computed on a greyscale
thumbnail: cheap enough that a camera could run it, and the one that filter
family reports as the most useful for counting-like queries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))

from fullrate_tracking import annotated_frames, source_video  # noqa: E402

THUMB = (160, 90)


def annotated_source_indices(video: str, args) -> list[int]:
    """Annotated frames of one video, in source-video numbering."""
    annotations = annotated_frames(video, args.root)
    if not annotations:
        raise SystemExit(f"no annotations for {video} under {args.root}")
    offset = int(args.frame_offsets.get(video, 0))
    numbers = [number - offset for number in annotations]
    mapping = args.frame_map.get(video)
    if mapping:
        numbers = [int(mapping.get(str(n), mapping.get(n))) for n in numbers]
    return sorted(set(numbers))


def thumbnails(path: Path, last: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")
    frames: list[np.ndarray] = []
    index = 0
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(grey, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32))
        index += 1
    capture.release()
    return frames


def uniform_extra(annotated: list[int], multiplier: int, last: int) -> list[int]:
    """m-1 equally spaced frames inside each gap between annotated frames.

    A gap shorter than the insertion count contributes whatever fits, so the
    budget is what the sequence can hold rather than a nominal figure.
    """
    extra: list[int] = []
    for left, right in zip(annotated, annotated[1:]):
        span = right - left
        if span <= 1:
            continue
        for j in range(1, multiplier):
            candidate = left + round(span * j / multiplier)
            if left < candidate < right:
                extra.append(candidate)
    tail = [index for index in extra if index <= last]
    return sorted(set(tail))


def adaptive_select(frames: list[np.ndarray], forced: set[int], threshold: float) -> list[int]:
    """Process a frame when it has drifted far enough from the last one processed."""
    chosen: list[int] = []
    reference = None
    for index, thumb in enumerate(frames):
        if index in forced:
            reference = thumb
            continue
        if reference is None:
            reference = thumb
            continue
        if float(np.abs(thumb - reference).mean()) >= threshold:
            chosen.append(index)
            reference = thumb
    return chosen


def bisect_threshold(frames, forced: set[int], budget: int, iterations: int = 40):
    """Smallest threshold whose selection does not exceed the budget."""
    if budget <= 0:
        return float("inf"), []
    low, high = 0.0, 255.0
    best = (high, [])
    for _ in range(iterations):
        middle = (low + high) / 2
        chosen = adaptive_select(frames, forced, middle)
        if len(chosen) > budget:
            low = middle
        else:
            best = (middle, chosen)
            high = middle
        if high - low < 1e-4:
            break
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--frame-offsets", type=Path, default=None)
    parser.add_argument("--multipliers", type=int, nargs="+", default=[2, 4, 8])
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
    return args


def main() -> None:
    args = parse_args()
    payload: dict[str, dict] = {}
    for video in args.videos:
        started = time.time()
        annotated = annotated_source_indices(video, args)
        path = source_video(video, args.video_root)
        frames = thumbnails(path, max(annotated))
        if len(frames) <= max(annotated):
            raise SystemExit(
                f"{video}: decoded {len(frames)} frames, annotation needs {max(annotated) + 1}"
            )
        forced = set(annotated)
        entry = {
            "video": video,
            "source_frames_read": len(frames),
            "annotated": annotated,
            "arms": {},
        }
        for multiplier in args.multipliers:
            extra = uniform_extra(annotated, multiplier, len(frames) - 1)
            extra = [index for index in extra if index not in forced]
            budget = len(extra)
            threshold, chosen = bisect_threshold(frames, forced, budget)
            entry["arms"][f"uniform_{multiplier}"] = {
                "extra": extra,
                "processed": sorted(forced | set(extra)),
                "budget": budget,
            }
            entry["arms"][f"adaptive_{multiplier}"] = {
                "extra": sorted(chosen),
                "processed": sorted(forced | set(chosen)),
                "budget": budget,
                "threshold": threshold,
                "overlap_with_uniform": len(set(chosen) & set(extra)),
            }
            print(
                f"{video} m={multiplier}: annotated={len(annotated)} budget={budget} "
                f"adaptive={len(chosen)} threshold={threshold:.4f} "
                f"overlap={len(set(chosen) & set(extra))}",
                flush=True,
            )
        payload[video] = entry
        print(f"  {video}: {time.time() - started:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "root": str(args.root),
            "video_root": str(args.video_root),
            "videos": args.videos,
            "multipliers": args.multipliers,
            "thumbnail": list(THUMB),
            "feature": "mean absolute difference against the last processed frame",
        },
        "sequences": payload,
    }, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
