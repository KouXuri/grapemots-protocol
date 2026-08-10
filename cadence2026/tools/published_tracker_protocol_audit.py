#!/usr/bin/env python3
"""Re-aggregate a published tracker's own released output under the counting protocol.

Everything else in this study runs a tracker we configured, which leaves the
objection that the protocol sensitivity is a property of our pipeline. This
script removes that objection without running a tracker at all: the per-frame
identities are taken from a published system's released result files, and only
the counting rule is varied.

That is also the manuscript's own recommendation applied to somebody else's work
-- releasing per-track outputs rather than a final count is exactly what makes
this possible, and here it costs no GPU and no images.

Input is MOTChallenge result format, `frame,id,x,y,w,h,conf,...`, one file per
sequence, beside a ground-truth root already converted by
tools/build_external_track_root.py. Frame numbering is checked, not assumed: the
converter renumbers annotated frames densely, so a tracker file whose frame span
does not match the reference length is refused rather than silently shifted.
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
from oracle_master import prefix_surface, read_resolutions, video_frames  # noqa: E402
from track_grapemots_mot import load_gt_tracks  # noqa: E402

# MOT17 result files are named per public detector; the identities are the same.
MOT_SUFFIX = re.compile(r"-(DPM|FRCNN|SDP)$")


def read_tracker_file(path: Path) -> dict[int, list[int]]:
    """source frame (1-based) -> predicted identities."""
    by_frame: dict[int, list[int]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            frame = int(float(row[0]))
            track = int(float(row[1]))
            by_frame[frame].append(track)
    return by_frame


def reference_ids(root: Path, video: str, size) -> list[list[int]]:
    width, height = size
    out = []
    for path in video_frames(root, video):
        ids, _ = load_gt_tracks(path, width, height)
        out.append([int(t) for t in ids])
    return out


def retained(surface, gt_full: int, coverage: float):
    for cell in surface:
        if cell["min_track_len"] * 2 > cell["window_frames"]:
            continue
        if not gt_full or cell["gt_tracks"] / gt_full < coverage:
            continue
        if cell["signed_relative_error"] is None:
            continue
        yield cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker-dir", type=Path, required=True,
                    help="directory of MOTChallenge-format result .txt files")
    ap.add_argument("--gt-root", type=Path, required=True,
                    help="track root built by build_external_track_root.py")
    ap.add_argument("--tracker-name", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--window-lengths", type=int, nargs="+", required=True)
    ap.add_argument("--min-track-lens", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sizes = read_resolutions(args.gt_root)
    records, skipped = [], []

    for path in sorted(args.tracker_dir.glob("*.txt")):
        video = MOT_SUFFIX.sub("", path.stem)
        if video not in sizes:
            skipped.append(f"{path.name}: no reference sequence named {video}")
            continue
        if any(record["video"] == video for record in records):
            continue                       # MOT17 ships the same identities three times

        gt_ids = reference_ids(args.gt_root, video, sizes[video])
        by_frame = read_tracker_file(path)
        if not by_frame:
            skipped.append(f"{path.name}: no rows")
            continue

        # The converter renumbers densely from the first annotated frame. Refuse
        # rather than guess if the tracker's frame span does not line up.
        first, last = min(by_frame), max(by_frame)
        if last - first + 1 > len(gt_ids):
            skipped.append(f"{path.name}: spans {last - first + 1} frames but the "
                           f"reference has {len(gt_ids)}")
            continue
        pred_ids = [by_frame.get(first + index, []) for index in range(len(gt_ids))]

        surface = prefix_surface(pred_ids, gt_ids, args.window_lengths, args.min_track_lens)
        gt_full = max(cell["gt_tracks"] for cell in surface)
        cells = list(retained(surface, gt_full, args.coverage))
        tau1 = [cell for cell in surface if cell["min_track_len"] == 1]
        whole = max(tau1, key=lambda cell: cell["window_frames"])
        errors = [cell["signed_relative_error"] for cell in cells]

        record = {
            "corpus": args.corpus, "tracker": args.tracker_name, "video": video,
            "frames": len(gt_ids), "gt_tracks": gt_full,
            "whole_sequence_predicted": whole["predicted_tracks"],
            "whole_sequence_error": whole["signed_relative_error"],
            "retained_cells": len(cells),
            "count_error_surface": surface,
        }
        if errors:
            lo, hi = min(errors), max(errors)
            record.update({
                "retained_median_error": float(np.median(errors)),
                "retained_min_error": lo, "retained_max_error": hi,
                "retained_span_ratio": (1 + hi) / (1 + lo) if lo > -1 else float("inf"),
            })
        records.append(record)
        span = record.get("retained_span_ratio")
        print(f"{args.corpus:8s} {video:12s} frames={len(gt_ids):5d} G={gt_full:5d} "
              f"P={whole['predicted_tracks']:5d} e={whole['signed_relative_error']:+.3f} "
              f"cells={len(cells):3d} span={span:.2f}x" if span else
              f"{args.corpus:8s} {video:12s} frames={len(gt_ids):5d} G={gt_full:5d} "
              f"P={whole['predicted_tracks']:5d} e={whole['signed_relative_error']:+.3f} "
              f"cells=0", flush=True)

    if not records:
        raise SystemExit(f"no usable tracker files under {args.tracker_dir}")

    errors = [record["whole_sequence_error"] for record in records]
    spans = [record["retained_span_ratio"] for record in records
             if np.isfinite(record.get("retained_span_ratio", float("inf")))]
    summary = {
        "corpus": args.corpus, "tracker": args.tracker_name,
        "sequences": len(records),
        "whole_sequence_error_median": float(np.median(errors)),
        "over_counting": int(sum(1 for e in errors if e > 0)),
        "under_counting": int(sum(1 for e in errors if e < 0)),
        "span_ratio_median": float(np.median(spans)) if spans else None,
        "span_ratio_max": float(max(spans)) if spans else None,
        "skipped": skipped,
    }
    print("\n" + "=" * 80)
    print(f"{args.tracker_name} on {args.corpus}: {summary['sequences']} sequences, "
          f"median e {summary['whole_sequence_error_median']:+.3f}, "
          f"{summary['over_counting']} over / {summary['under_counting']} under")
    if summary["span_ratio_median"]:
        print(f"  protocol span: median {summary['span_ratio_median']:.2f}x, "
              f"max {summary['span_ratio_max']:.2f}x")
    for text in skipped:
        print(f"  skipped {text}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "sequences": records},
                                   indent=1, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
