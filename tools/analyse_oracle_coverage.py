#!/usr/bin/env python3
"""Re-analyse the oracle counting surface with a window-coverage constraint.

Why this exists
---------------
The first pass reported "11/11 videos admit an exactly-zero-error (L, tau)".
That claim does not survive scrutiny: 9 of those 11 zeros sit at L=5, where the
window contains only 14-58% of the bunches the full sequence contains. At that
length each track lives at most 5 frames, so the tracker has no opportunity to
fragment an identity -- the task has degenerated into per-frame detection, and
"zero counting error" is trivial. Reporting it would invite the obvious referee
objection and would amount to picking a favourable corner of the grid.

So every cell is scored by coverage = gt_tracks(L) / gt_tracks(full sequence),
and only windows that actually pose the counting task are kept. The surviving
claim is stronger and does not depend on a favourable region:

    with a PERFECT detector and windows covering >=80% of the bunches, unique
    track counting still over-counts on every video, by an amount the two
    undeclared knobs (L, tau) move by up to 4.8x on a single sequence.

Usage: python analyse_oracle_coverage.py <oracle_count_surface_v2.json> [coverage]
"""
import json
import sys
from collections import defaultdict


def load(path):
    with open(path) as fh:
        return json.load(fh)["runs"]


def full_gt(runs):
    """Ground-truth track count over the whole sequence, per video."""
    out = {}
    for r in runs:
        best = max(c["gt_tracks"] for c in r["count_error_surface"])
        out[r["video"]] = max(out.get(r["video"], 0), best)
    return out


def cells(run, gt_full, min_cov):
    """Non-degenerate cells: tau <= L/2, and the window sees most of the bunches."""
    for c in run["count_error_surface"]:
        if c["min_track_len"] * 2 > c["window_frames"]:
            continue                                   # degenerate: tau > L/2
        if not gt_full or c["gt_tracks"] / gt_full < min_cov:
            continue                                   # window too short to pose the task
        yield c


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/oracle_count_surface_v2.json"
    min_cov = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    runs = load(path)
    gt = full_gt(runs)

    print(f"coverage threshold: gt_tracks(L) >= {min_cov:.0%} of full-sequence GT")
    print("degenerate cells (tau > L/2) excluded\n")

    for miss in sorted({r["miss_rate"] for r in runs}):
        arm = [r for r in runs if r["miss_rate"] == miss and r["tracker"] == "bytetrack.yaml"]
        if not arm:
            continue
        print(f"=== per-frame miss rate p = {miss} ===")
        print(f"{'video':20s} {'cells':>5s} {'min':>8s} {'max':>8s} {'ratio':>7s}  sign")
        spans, crossers = [], 0
        for r in sorted(arm, key=lambda x: x["video"]):
            errs = [c["signed_relative_error"] for c in cells(r, gt[r["video"]], min_cov)]
            if not errs:
                print(f"{r['video']:20s} {0:5d}   no window reaches this coverage")
                continue
            lo, hi = min(errs), max(errs)
            crosses = lo < 0.0 < hi
            crossers += crosses
            # how far the two knobs alone move the reported number
            ratio = (1 + hi) / (1 + lo) if lo > -1 else float("inf")
            spans.append(ratio)
            sign = "CROSSES 0" if crosses else ("all over" if lo >= 0 else "all under")
            print(f"{r['video']:20s} {len(errs):5d} {lo:+8.3f} {hi:+8.3f} {ratio:7.2f}x  {sign}")
        if spans:
            spans.sort()
            med = spans[len(spans) // 2]
            print(f"  -> reported count moves by up to {max(spans):.2f}x on one video "
                  f"(median {med:.2f}x); {crossers}/{len(spans)} videos cross zero\n")


if __name__ == "__main__":
    main()
