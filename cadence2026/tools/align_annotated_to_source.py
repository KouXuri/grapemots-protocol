#!/usr/bin/env python3
"""Recover which source video frame each annotated frame came from.

The Bodegas release publishes about 22 annotated frames per sequence while the
sequences themselves run 210 to 1782 source frames, and it does not state which
frames were labelled. That missing entry is why consecutive annotated boxes of
one trajectory barely overlap: the gap between two labelled frames is on the
order of a second of flight, not one frame. Recovering the mapping turns that
from an inference into a measurement, and it is what makes a source-rate
tracking comparison possible at all -- the same footage, the same detector, only
the cadence changing.

Matching is by appearance, not by guessing an offset. Each frame is reduced to a
small grey thumbnail and the annotated frame is assigned the source frame with
the smallest mean absolute difference. Two things are reported beside the match
so the result can be judged rather than trusted:

  residual   mean absolute grey difference at the chosen frame. On a true match
             this is a couple of grey levels; a mismatch runs an order higher.
  margin     how much worse the best non-adjacent candidate is. A small margin
             means the sequence is locally static and the index is uncertain
             even though the picture is right.

The assignment is forced to be non-decreasing, because annotated frames are
released in temporal order and a matcher that reorders them has failed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

FRAME_RE = re.compile(r"__frame_(\d+)\.[A-Za-z]+$")


def thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA).ravel()


def decode_video(path: Path, width: int, height: int, limit: int | None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(thumbnail(frame, width, height))
        if limit and len(frames) >= limit:
            break
    capture.release()
    if not frames:
        raise SystemExit(f"decoded no frames from {path}")
    return np.asarray(frames, dtype=np.int16)


def match(annotated: np.ndarray, source: np.ndarray, chunk: int = 256):
    """For each annotated thumbnail: best source index, residual, margin."""
    best_index = np.zeros(len(annotated), dtype=int)
    best_cost = np.full(len(annotated), np.inf)
    margin = np.zeros(len(annotated))
    for start in range(0, len(annotated), 8):
        block = annotated[start:start + 8].astype(np.int16)
        costs = np.empty((len(block), len(source)), dtype=np.float32)
        for offset in range(0, len(source), chunk):
            piece = source[offset:offset + chunk]
            costs[:, offset:offset + chunk] = np.abs(
                block[:, None, :] - piece[None, :, :]).mean(axis=2)
        for row in range(len(block)):
            order = np.argsort(costs[row])
            index = int(order[0])
            best_index[start + row] = index
            best_cost[start + row] = float(costs[row][index])
            # ignore neighbours of the winner: adjacent frames are nearly identical
            far = [j for j in order[1:] if abs(int(j) - index) > 2]
            margin[start + row] = float(costs[row][far[0]] - costs[row][index]) if far else 0.0
    return best_index, best_cost, margin


def enforce_monotone(indices: np.ndarray) -> tuple[np.ndarray, int]:
    fixed = indices.copy()
    violations = 0
    for position in range(1, len(fixed)):
        if fixed[position] <= fixed[position - 1]:
            violations += 1
            fixed[position] = fixed[position - 1] + 1
    return fixed, violations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track-root", type=Path, required=True,
                    help="root with images/train/<seq>__frame_<NNNNNN>.<ext>")
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--video-name", action="append", default=[], metavar="SEQ=FILE",
                    help="override the video file for a sequence")
    ap.add_argument("--thumb", type=int, nargs=2, default=(64, 36))
    ap.add_argument("--max-residual", type=float, default=12.0,
                    help="a sequence whose median residual exceeds this is reported "
                         "as unaligned rather than silently used")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    overrides = dict(pair.split("=", 1) for pair in args.video_name)
    width, height = args.thumb
    images_dir = args.track_root / "images" / "train"
    videos = {path.stem.casefold(): path for path in args.video_dir.glob("*.mp4")}

    report = {"thumb": [width, height], "sequences": {}}
    for sequence in args.sequences:
        frames = sorted(images_dir.glob(f"{sequence}__frame_*.*"),
                        key=lambda p: int(FRAME_RE.search(p.name).group(1)))
        if not frames:
            print(f"{sequence}: no annotated images, skipped")
            continue
        name = overrides.get(sequence, sequence.replace("row_", "Row").replace("_", "_", 1))
        path = videos.get(Path(name).stem.casefold()) or videos.get(name.casefold())
        if path is None:
            candidates = [p for key, p in videos.items()
                          if key.replace(".", "").replace("_", "")
                          == sequence.replace("row_", "row").replace(".", "").replace("_", "")]
            path = candidates[0] if candidates else None
        if path is None:
            print(f"{sequence}: no source video found (looked for {name}), skipped")
            report["sequences"][sequence] = {"status": "no_video"}
            continue

        source = decode_video(path, width, height, None)
        annotated = np.asarray([thumbnail(cv2.imread(str(p)), width, height)
                                for p in frames], dtype=np.int16)
        raw, residual, margin = match(annotated, source)
        indices, violations = enforce_monotone(raw)
        gaps = np.diff(indices)
        median_residual = float(np.median(residual))
        status = "aligned" if median_residual <= args.max_residual else "unaligned"

        report["sequences"][sequence] = {
            "status": status,
            "video": path.name,
            "source_frames": int(len(source)),
            "annotated_frames": int(len(frames)),
            "sampling_ratio": float(len(source) / len(frames)),
            "median_residual": median_residual,
            "max_residual": float(residual.max()),
            "median_margin": float(np.median(margin)),
            "monotonicity_fixes": int(violations),
            "source_gap_median": float(np.median(gaps)) if len(gaps) else None,
            "source_gap_min": int(gaps.min()) if len(gaps) else None,
            "source_gap_max": int(gaps.max()) if len(gaps) else None,
            "annotated_to_source": {int(FRAME_RE.search(p.name).group(1)): int(i)
                                    for p, i in zip(frames, indices)},
        }
        entry = report["sequences"][sequence]
        print(f"{sequence:12s} {status:9s} {len(frames):3d} annotated in "
              f"{len(source):5d} source frames (1 per {entry['sampling_ratio']:5.1f}), "
              f"gap median {entry['source_gap_median']:6.1f} "
              f"[{entry['source_gap_min']}, {entry['source_gap_max']}], "
              f"residual {median_residual:5.2f}, margin {entry['median_margin']:5.2f}, "
              f"fixes {violations}", flush=True)

    aligned = [s for s, e in report["sequences"].items() if e.get("status") == "aligned"]
    report["aligned_sequences"] = aligned
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{len(aligned)}/{len(report['sequences'])} sequences aligned; wrote {args.out}")


if __name__ == "__main__":
    main()
