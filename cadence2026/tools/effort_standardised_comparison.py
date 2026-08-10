#!/usr/bin/env python3
"""Separate the effort-dependent part of a count disagreement from the identity part.

Ecology settled the first half of this problem decades ago. Observed species
richness rises with sampling effort, so richness measured at unequal effort
cannot be compared; the standard remedy is to interpolate or extrapolate every
sample to a common SAMPLE COVERAGE and compare there (Chao et al. 2014,
Ecological Monographs 84:45-67). Our retained-cell rule, G(L)/G(full) >= 0.8,
is a crude version of exactly that.

The borrowing has to stop at a specific place, and saying where is the point of
this script. In a biodiversity survey every detected individual is assigned to
the right species, so the only error is that rare species go undetected, and the
bias is one-directional: observed richness underestimates true richness. A
tracker has a second error source that the ecological framework assumes away --
it can split one object across several identities, which drives the count UP,
and it can fail to assign an object at all, which drives it DOWN. Applying an
asymptotic richness estimator to a tracker's output would treat fragmentation as
undetected rarity, which it is not.

So: standardise by coverage to remove the sampling-effort component, then read
what remains through U, D and M. Effort explains the part a richness estimator
would explain; identity explains the rest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cells(run: dict):
    for cell in run["count_error_surface"]:
        if cell["signed_relative_error"] is not None and cell["gt_tracks"]:
            yield cell


def coverage_curve(run: dict, tau: int = 1):
    """(coverage, signed error) along the window axis at one eligibility level."""
    rows = [cell for cell in cells(run) if cell["min_track_len"] == tau]
    if not rows:
        return []
    full = max(cell["gt_tracks"] for cell in rows)
    return sorted(((cell["gt_tracks"] / full, cell["signed_relative_error"],
                    cell["window_frames"]) for cell in rows), key=lambda item: item[0])


def error_at_coverage(curve, target: float):
    """Linear interpolation along the coverage axis; None outside the observed range."""
    if not curve or target < curve[0][0] or target > curve[-1][0]:
        return None
    for (c0, e0, _), (c1, e1, _) in zip(curve, curve[1:]):
        if c0 <= target <= c1:
            if c1 == c0:
                return e0
            weight = (target - c0) / (c1 - c0)
            return e0 + weight * (e1 - e0)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--coverages", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = {"tau": args.tau, "coverages": args.coverages, "groups": {}}
    print(f"Signed count error at matched sample coverage, tau={args.tau}")
    print(f"{'group':<14}" + "".join(f"{c:>12.0%}" for c in args.coverages) + f"{'whole seq':>12}")

    for spec in args.oracle:
        label, _, path = spec.partition("=")
        runs = [run for run in json.loads(Path(path).read_text())["runs"]
                if run.get("miss_rate", 0.0) == 0.0]
        at = {c: [] for c in args.coverages}
        whole = []
        for run in runs:
            curve = coverage_curve(run, args.tau)
            if not curve:
                continue
            whole.append(curve[-1][1])
            for target in args.coverages:
                value = error_at_coverage(curve, target)
                if value is not None:
                    at[target].append(value)
        entry = {"sequences": len(whole),
                 "whole_sequence_median": float(np.median(whole)) if whole else None,
                 "at_coverage": {str(c): {"n": len(v),
                                          "median": float(np.median(v)) if v else None}
                                 for c, v in at.items()}}
        report["groups"][label] = entry
        line = f"{label:<14}"
        for c in args.coverages:
            value = entry["at_coverage"][str(c)]["median"]
            line += f"{value:>+12.3f}" if value is not None else f"{'-':>12}"
        line += f"{entry['whole_sequence_median']:>+12.3f}" if whole else f"{'-':>12}"
        print(line)

    # How much of the between-group spread survives standardisation? If coverage
    # were the whole story the spread would collapse; whatever remains is what
    # U/D/M has to account for.
    print("\nSpread between groups (max - min of the group medians):")
    for c in args.coverages:
        vals = [e["at_coverage"][str(c)]["median"] for e in report["groups"].values()
                if e["at_coverage"][str(c)]["median"] is not None]
        if len(vals) > 1:
            print(f"  at coverage {c:.0%}: {max(vals) - min(vals):.3f}")
    wholes = [e["whole_sequence_median"] for e in report["groups"].values()
              if e["whole_sequence_median"] is not None]
    if len(wholes) > 1:
        print(f"  whole sequence : {max(wholes) - min(wholes):.3f}")
    print("\nStandardising by coverage removes the sampling-effort component only.")
    print("A residual spread is identity error, which no richness estimator addresses.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
