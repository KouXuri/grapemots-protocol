#!/usr/bin/env python3
"""Uncertainty for the headline numbers, resampling the unit that actually varies.

Every count summary in the manuscript is a median of sequence-level medians over
eleven videos, reported as a point value. Eleven is small, so a reviewer is right
to ask how much of the number is sampling noise. The resampling unit here is the
SEQUENCE, because sequences are what could have been collected differently; cells
of the (L, tau) grid are nested inside a sequence and resampling those would
manufacture precision that does not exist.

Two things are reported and they answer different questions:

  interval    a percentile bootstrap over sequences, for the median error and
              for the protocol span. This says how well the corpus median is
              pinned down.

  sign        the count of sequences on the same side of zero, with an exact
              binomial tail against a fair-coin null. This is the claim the
              manuscript actually leans on ("over-counts on all eleven"), and it
              needs no distributional assumption.

The exact tail is computed in integer arithmetic so the result does not depend on
scipy being installed, which it has not always been on this host.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def retained_cells(run: dict, gt_full: int, coverage: float):
    for cell in run["count_error_surface"]:
        if cell["min_track_len"] * 2 > cell["window_frames"]:
            continue
        if not gt_full or cell["gt_tracks"] / gt_full < coverage:
            continue
        if cell["signed_relative_error"] is None:
            continue
        yield cell


def sequence_values(path: Path, coverage: float) -> list[dict]:
    runs = [run for run in json.loads(path.read_text())["runs"] if run["miss_rate"] == 0.0]
    full: dict[str, int] = {}
    for run in runs:
        best = max(cell["gt_tracks"] for cell in run["count_error_surface"])
        full[run["video"]] = max(full.get(run["video"], 0), best)
    out = []
    for run in runs:
        cells = list(retained_cells(run, full[run["video"]], coverage))
        tau1 = [c for c in run["count_error_surface"] if c["min_track_len"] == 1]
        if not cells or not tau1:
            continue
        whole = max(tau1, key=lambda c: c["window_frames"])
        errors = [c["signed_relative_error"] for c in cells]
        lo, hi = min(errors), max(errors)
        out.append({
            "video": run["video"],
            "whole_sequence_error": whole["signed_relative_error"],
            "retained_median_error": float(np.median(errors)),
            "retained_span_ratio": (1 + hi) / (1 + lo) if lo > -1 else float("inf"),
        })
    return out


def bootstrap_median(values: list[float], resamples: int, rng) -> dict:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return {"n": len(finite), "median": float(finite[0]) if finite else None,
                "ci95": [None, None]}
    array = np.asarray(finite, float)
    draws = rng.integers(0, len(array), size=(resamples, len(array)))
    medians = np.median(array[draws], axis=1)
    return {
        "n": len(finite),
        "median": float(np.median(array)),
        "ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
        "bootstrap_sd": float(np.std(medians, ddof=1)),
    }


def exact_binomial_tail(successes: int, trials: int) -> float:
    """Two-sided p for a fair-coin null, by summing exact terms."""
    if trials == 0:
        return float("nan")
    target = math.comb(trials, successes)
    total = sum(math.comb(trials, k) for k in range(trials + 1)
                if math.comb(trials, k) <= target)
    return total / (2 ** trials)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", action="append", required=True, metavar="CORPUS=PATH")
    ap.add_argument("--oof-results", type=Path,
                    help="directory of real-pipeline tracking JSONs, for the U/D/M pooling")
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    report = {"coverage": args.coverage, "resamples": args.resamples,
              "resampling_unit": "sequence", "corpora": {}}

    print("=" * 92)
    print(f"BOOTSTRAP OVER SEQUENCES  ({args.resamples} resamples, seed {args.seed})")
    print("=" * 92)
    print(f"{'corpus':<14}{'n':>4}{'median e':>26}{'over':>7}{'under':>7}{'exact p':>11}")

    for spec in args.oracle:
        corpus, _, path = spec.partition("=")
        values = sequence_values(Path(path), args.coverage)
        if not values:
            print(f"{corpus:<14}  no retained sequence")
            continue
        whole = [entry["whole_sequence_error"] for entry in values]
        retained = [entry["retained_median_error"] for entry in values]
        spans = [entry["retained_span_ratio"] for entry in values]

        over = sum(1 for value in whole if value > 0)
        under = sum(1 for value in whole if value < 0)
        majority = max(over, under)
        tail = exact_binomial_tail(majority, over + under)

        entry = {
            "sequences": len(values),
            "whole_sequence_error": bootstrap_median(whole, args.resamples, rng),
            "retained_median_error": bootstrap_median(retained, args.resamples, rng),
            "retained_span_ratio": bootstrap_median(spans, args.resamples, rng),
            "over_counting": over,
            "under_counting": under,
            "exactly_zero": sum(1 for value in whole if value == 0),
            "sign_test_exact_p": tail,
            "infinite_spans": sum(1 for value in spans if not np.isfinite(value)),
            "per_sequence": values,
        }
        report["corpora"][corpus] = entry
        interval = entry["whole_sequence_error"]["ci95"]
        print(f"{corpus:<14}{len(values):>4}"
              f"{entry['whole_sequence_error']['median']:>+12.3f}"
              f"  [{interval[0]:+.3f}, {interval[1]:+.3f}]"
              f"{over:>7}{under:>7}{tail:>11.2e}")

    if args.oof_results and args.oof_results.is_dir():
        print()
        print("=" * 92)
        print("REAL-PIPELINE OOF FILES FOUND (pooled U/D/M is bootstrapped by the")
        print("analysis that owns the decomposition; listed here for provenance)")
        print("=" * 92)
        found = sorted(path.name for path in args.oof_results.glob("*.json"))
        report["oof_files"] = found
        print(f"  {len(found)} JSON files under {args.oof_results}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
