#!/usr/bin/env python3
"""Join reference-annotation structure to oracle count error, across corpora.

The manuscript's oracle result is that tracking ground-truth boxes still
over-counts. Running the identical sweep on other corpora does not reproduce
that sign, and the honest reading is not "the finding failed to replicate" but
"the sign is a regime, and the regime is visible in the annotation before any
tracker runs".

This script produces the evidence for that claim: one row per sequence with its
structural statistics beside its oracle count error, the per-corpus summary, and
rank correlations pooled over every sequence. Rank correlation is used because
the structural statistics span orders of magnitude across corpora and nothing
here justifies a linear form.

Retained cells follow the manuscript's rule exactly, reusing the same three
conditions as tools/analyse_oracle_coverage.py: G(L) > 0, tau <= L/2, and
G(L)/G(full) >= the coverage threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STRUCTURE_KEYS = [
    "consecutive_iou_median", "consecutive_iou_below_0p3", "step_over_size_median",
    "lifetime_median", "lifetime_median_frac", "short_life_frac", "gap_frac",
    "gaps_per_trajectory", "visible_mean", "box_area_frac_median",
    "turnover_per_frame", "frames", "trajectories",
]


def spearman(x: list[float], y: list[float]) -> tuple[float, int]:
    """Rank correlation, written out so the result does not depend on scipy."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
             and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    a = _rank([p[0] for p in pairs])
    b = _rank([p[1] for p in pairs])
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return (float((a * b).sum() / denominator) if denominator else float("nan")), len(pairs)


def _rank(values: list[float]) -> list[float]:
    """Average ranks, so ties do not bias the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def full_gt(runs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for run in runs:
        best = max(cell["gt_tracks"] for cell in run["count_error_surface"])
        out[run["video"]] = max(out.get(run["video"], 0), best)
    return out


def retained(run: dict, gt_full: int, coverage: float):
    for cell in run["count_error_surface"]:
        if cell["min_track_len"] * 2 > cell["window_frames"]:
            continue
        if not gt_full or cell["gt_tracks"] / gt_full < coverage:
            continue
        if cell["signed_relative_error"] is None:
            continue
        yield cell


def summarise_run(run: dict, gt_full: int, coverage: float) -> dict | None:
    cells = list(retained(run, gt_full, coverage))
    tau1 = [cell for cell in run["count_error_surface"] if cell["min_track_len"] == 1]
    whole = max(tau1, key=lambda cell: cell["window_frames"]) if tau1 else None
    if not cells or whole is None:
        return None
    errors = [cell["signed_relative_error"] for cell in cells]
    lo, hi = min(errors), max(errors)
    ratio = (1 + hi) / (1 + lo) if lo > -1 else float("inf")
    return {
        "whole_sequence_error": whole["signed_relative_error"],
        "whole_sequence_P": whole["predicted_tracks"],
        "whole_sequence_G": whole["gt_tracks"],
        "retained_cells": len(cells),
        "retained_median_error": float(np.median(errors)),
        "retained_min_error": lo,
        "retained_max_error": hi,
        "retained_span_ratio": ratio,
        "crosses_zero": bool(lo < 0.0 < hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", type=Path, required=True)
    ap.add_argument("--oracle", action="append", required=True,
                    metavar="CORPUS=PATH", help="repeat once per corpus")
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    structure = {(row["corpus"], row["video"]): row
                 for row in json.loads(args.structure.read_text())["sequences"]}

    rows = []
    for spec in args.oracle:
        corpus, _, path = spec.partition("=")
        runs = [run for run in json.loads(Path(path).read_text())["runs"]
                if run["miss_rate"] == 0.0]
        gt = full_gt(runs)
        for run in runs:
            key = (corpus, run["video"])
            if key not in structure:
                print(f"  ! no structure record for {key}, skipped")
                continue
            summary = summarise_run(run, gt[run["video"]], args.coverage)
            if summary is None:
                print(f"  ! no retained cell for {key}, skipped")
                continue
            rows.append({"corpus": corpus, "video": run["video"],
                         **{k: structure[key][k] for k in STRUCTURE_KEYS},
                         **summary})

    fields = ["corpus", "video"] + STRUCTURE_KEYS + [
        "whole_sequence_error", "whole_sequence_P", "whole_sequence_G",
        "retained_cells", "retained_median_error", "retained_min_error",
        "retained_max_error", "retained_span_ratio", "crosses_zero"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 96)
    print(f"PER-CORPUS SUMMARY  (oracle boxes, ByteTrack, coverage >= {args.coverage:.0%})")
    print("=" * 96)
    header = (f"{'corpus':<13}{'seq':>4}{'consecIoU':>11}{'step/size':>11}"
              f"{'life':>7}{'e(whole)':>11}{'over':>6}{'under':>7}{'span':>9}")
    print(header)
    per_corpus = {}
    for corpus in dict.fromkeys(row["corpus"] for row in rows):
        group = [row for row in rows if row["corpus"] == corpus]
        errors = [row["whole_sequence_error"] for row in group]
        spans = [row["retained_span_ratio"] for row in group if np.isfinite(row["retained_span_ratio"])]
        entry = {
            "sequences": len(group),
            "consecutive_iou_median": float(np.median([row["consecutive_iou_median"] for row in group])),
            "step_over_size_median": float(np.median([row["step_over_size_median"] for row in group])),
            "lifetime_median": float(np.median([row["lifetime_median"] for row in group])),
            "whole_sequence_error_median": float(np.median(errors)),
            "over_counting": int(sum(1 for e in errors if e > 0)),
            "under_counting": int(sum(1 for e in errors if e < 0)),
            "exactly_zero": int(sum(1 for e in errors if e == 0)),
            "span_ratio_median": float(np.median(spans)) if spans else None,
            "span_ratio_max": float(max(spans)) if spans else None,
            "span_ratio_infinite": int(sum(1 for row in group
                                           if not np.isfinite(row["retained_span_ratio"]))),
        }
        per_corpus[corpus] = entry
        span = f"{entry['span_ratio_median']:.2f}x" if entry["span_ratio_median"] else "-"
        print(f"{corpus:<13}{entry['sequences']:>4}{entry['consecutive_iou_median']:>11.3f}"
              f"{entry['step_over_size_median']:>11.3f}{entry['lifetime_median']:>7.0f}"
              f"{entry['whole_sequence_error_median']:>+11.3f}"
              f"{entry['over_counting']:>6}{entry['under_counting']:>7}{span:>9}")

    def correlate(subset: list[dict]) -> dict:
        targets = {
            "whole_sequence_error": [row["whole_sequence_error"] for row in subset],
            "retained_span_ratio": [row["retained_span_ratio"]
                                    if np.isfinite(row["retained_span_ratio"]) else None
                                    for row in subset],
        }
        out = {}
        for key in STRUCTURE_KEYS:
            values = [row[key] for row in subset]
            out[key] = {}
            for name, target in targets.items():
                rho, n = spearman(values, target)
                out[key][name] = {"rho": rho, "n": n}
        return out

    print()
    print("=" * 96)
    print(f"RANK CORRELATION, POOLED over {len(rows)} sequences")
    print("  Descriptive only. The sequences are not independent: they arrive in four corpora")
    print("  that are internally homogeneous and far apart, so a pooled coefficient is driven")
    print("  mainly by differences BETWEEN four clusters, not by 51 independent observations.")
    print("=" * 96)
    correlations = correlate(rows)
    for key in STRUCTURE_KEYS:
        entry = correlations[key]
        print(f"  {key:<26}  error: rho={entry['whole_sequence_error']['rho']:+.3f}"
              f" (n={entry['whole_sequence_error']['n']:2d})"
              f"   span: rho={entry['retained_span_ratio']['rho']:+.3f}"
              f" (n={entry['retained_span_ratio']['n']:2d})")

    print()
    print("=" * 96)
    print("RANK CORRELATION WITHIN EACH CORPUS  (this is the inferentially honest view)")
    print("=" * 96)
    within = {}
    for corpus in per_corpus:
        subset = [row for row in rows if row["corpus"] == corpus]
        within[corpus] = correlate(subset)
        if len(subset) < 5:
            print(f"\n  {corpus} (n={len(subset)}): too few sequences to report a rank correlation")
            continue
        print(f"\n  {corpus} (n={len(subset)})")
        for key in ("lifetime_median", "step_over_size_median",
                    "consecutive_iou_below_0p3", "gap_frac", "visible_mean"):
            entry = within[corpus][key]
            print(f"    {key:<26}  error: rho={entry['whole_sequence_error']['rho']:+.3f}"
                  f" (n={entry['whole_sequence_error']['n']:2d})"
                  f"   span: rho={entry['retained_span_ratio']['rho']:+.3f}"
                  f" (n={entry['retained_span_ratio']['n']:2d})")

    args.out_json.write_text(json.dumps({
        "coverage": args.coverage,
        "per_corpus": per_corpus,
        "spearman_pooled": correlations,
        "spearman_within_corpus": within,
        "pooled_caveat": "sequences cluster by corpus; pooled coefficients are descriptive",
        "sequences": rows,
    }, indent=1, default=float) + "\n")
    print(f"\nwrote {args.out_csv}\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
