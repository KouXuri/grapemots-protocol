#!/usr/bin/env python3
"""Structural statistics of a reference annotation, computed without any tracker.

The manuscript reports that tracking oracle boxes over-counts on every GrapeMOTS
sequence. Replicating the same sweep on other corpora does not reproduce that
sign: short vineyard clips under-count heavily and pedestrian sequences barely
move. A referee is then entitled to ask what distinguishes them, and "a different
dataset" is not an answer.

Everything here is measurable from the reference annotation alone, before a
tracker is run, so any of it can be used to predict where a distinct-track count
will be unreliable:

  lifetime          how many frames a trajectory is annotated for. A trajectory
                    shorter than the tracker's confirmation delay cannot become
                    a counted identity at all, which pushes the count DOWN.
  gaps              how often an annotated trajectory disappears and returns.
                    Each return is an opportunity to be issued a new identity,
                    which pushes the count UP.
  displacement      centre motion between consecutive annotated frames, divided
                    by the square root of the box area. This is the quantity a
                    constant-velocity motion model has to bridge; at 1.0 the
                    object moves its own width between updates.
  overlap           IoU between consecutive annotated boxes of one trajectory.
                    The complementary view of the same thing, and it is what an
                    IoU-gated associator actually thresholds on.
  crowding          simultaneous annotated objects per frame, and their scale
                    relative to the frame.

Displacement and overlap are reported per annotated step, so they already
incorporate the annotation cadence: a sequence labelled every second source
frame has twice the per-step motion of the same footage labelled every frame.
That is deliberate -- the tracker sees annotated frames, not source frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_grapemots_mot import load_gt_tracks  # noqa: E402

FRAME_RE = re.compile(r"__frame_(\d+)\.txt$")


def read_resolutions(root: Path) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    with (root / "manifest.csv").open() as handle:
        for row in csv.DictReader(handle):
            first = row["resolution"].split(";")[0].split(":")[0]
            width, height = (int(value) for value in first.split("x"))
            sizes[row["video"]] = (width, height)
    return sizes


def video_frames(root: Path, video: str) -> list[Path]:
    found: list[Path] = []
    for split in ("train", "val", "test"):
        found.extend((root / "tracks" / split).glob(f"{video}__frame_*.txt"))
    if not found:
        raise SystemExit(f"no track sidecars for {video} under {root}")
    return sorted(found, key=lambda p: int(FRAME_RE.search(p.name).group(1)))


def iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def describe(root: Path, video: str) -> dict:
    width, height = read_resolutions(root)[video]
    frames = video_frames(root, video)

    # index -> {track: box}; index is the dense annotated-frame position
    per_frame: list[dict[int, np.ndarray]] = []
    for path in frames:
        ids, boxes = load_gt_tracks(path, width, height)
        per_frame.append({int(t): box for t, box in zip(ids, boxes)})

    appearances: dict[int, list[int]] = defaultdict(list)
    for index, entry in enumerate(per_frame):
        for track in entry:
            appearances[track].append(index)

    lifetimes, spans, gap_counts, gap_lengths = [], [], [], []
    for track, indices in appearances.items():
        lifetimes.append(len(indices))
        spans.append(indices[-1] - indices[0] + 1)
        gaps = [b - a - 1 for a, b in zip(indices, indices[1:]) if b - a > 1]
        gap_counts.append(len(gaps))
        gap_lengths.extend(gaps)

    steps, overlaps, areas = [], [], []
    for track, indices in appearances.items():
        for a, b in zip(indices, indices[1:]):
            if b - a != 1:
                continue                       # only genuinely consecutive updates
            box_a, box_b = per_frame[a][track], per_frame[b][track]
            size = np.sqrt(max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
            centre_a = np.array([(box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2])
            centre_b = np.array([(box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2])
            steps.append(float(np.linalg.norm(centre_b - centre_a) / size))
            overlaps.append(iou(box_a, box_b))
    for entry in per_frame:
        for box in entry.values():
            areas.append(float((box[2] - box[0]) * (box[3] - box[1])))

    counts = [len(entry) for entry in per_frame]
    new_per_frame = [len(indices) for indices in appearances.values()]

    def med(values, default=None):
        return float(np.median(values)) if len(values) else default

    return {
        "video": video,
        "frames": len(frames),
        "trajectories": len(appearances),
        "instances": int(sum(counts)),
        "visible_mean": float(np.mean(counts)) if counts else 0.0,
        "lifetime_median": med(lifetimes, 0.0),
        "lifetime_mean": float(np.mean(lifetimes)) if lifetimes else 0.0,
        "lifetime_median_frac": med(lifetimes, 0.0) / len(frames) if frames else None,
        "short_life_frac": float(np.mean([life <= 2 for life in lifetimes])) if lifetimes else None,
        "span_median": med(spans, 0.0),
        "gap_frac": float(np.mean([count > 0 for count in gap_counts])) if gap_counts else None,
        "gaps_per_trajectory": float(np.mean(gap_counts)) if gap_counts else None,
        "gap_length_median": med(gap_lengths),
        "step_over_size_median": med(steps),
        "step_over_size_p90": float(np.percentile(steps, 90)) if steps else None,
        "consecutive_iou_median": med(overlaps),
        "consecutive_iou_below_0p3": float(np.mean([value < 0.3 for value in overlaps])) if overlaps else None,
        "box_area_frac_median": med(areas, 0.0) / (width * height),
        "turnover_per_frame": len(appearances) / len(frames) if frames else None,
        "resolution": f"{width}x{height}",
        "new_ids_total": int(sum(new_per_frame) and len(appearances)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", required=True,
                    help="track root; repeat to describe several corpora at once")
    ap.add_argument("--label", action="append",
                    help="corpus name per --root, in the same order")
    ap.add_argument("--videos", nargs="+", help="restrict to these sequences")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labels = args.label or [root.name for root in args.root]
    if len(labels) != len(args.root):
        raise SystemExit("--label must be given once per --root")

    records = []
    for root, label in zip(args.root, labels):
        wanted = args.videos or sorted(read_resolutions(root))
        for video in wanted:
            if video not in read_resolutions(root):
                continue
            record = describe(root, video)
            record["corpus"] = label
            records.append(record)
            print(f"{label:12s} {video:14s} frames={record['frames']:5d} "
                  f"G={record['trajectories']:5d} life~{record['lifetime_median']:6.1f} "
                  f"step/size~{record['step_over_size_median']:.3f} "
                  f"IoU~{record['consecutive_iou_median']:.3f} "
                  f"gapfrac={record['gap_frac']:.2f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"sequences": records}, indent=1) + "\n")
    print(f"\nwrote {args.out}  ({len(records)} sequences)")


if __name__ == "__main__":
    main()
