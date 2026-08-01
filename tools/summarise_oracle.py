#!/usr/bin/env python3
"""Headline numbers from the oracle counting sweep.

The drift model is refitted here rather than reused from the sweep's own
drift_fit, because the dense window grid contains a degenerate corner: when the
window is shorter than the minimum track length (L < tau) no track can possibly
qualify, the count is identically zero and the error is exactly -100% for
reasons that have nothing to do with identity drift.  Those rows are excluded
from the regression (L >= 2*tau) and reported separately -- they are a real part
of the story, just not part of the linear regime.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

data = json.loads(Path(sys.argv[1]).read_text())
runs = data["runs"]
BASE = "bytetrack.yaml"


def rows(run, tau=None, min_ratio=None):
    out = []
    for row in run["count_error_surface"]:
        if tau is not None and row["min_track_len"] != tau:
            continue
        if min_ratio is not None and row["window_frames"] < min_ratio * row["min_track_len"]:
            continue
        out.append(row)
    return out


def cell(run, length, tau):
    for row in run["count_error_surface"]:
        if row["window_frames"] == length and row["min_track_len"] == tau:
            return row
    return None


def refit(run, tau):
    points = [(r["window_frames"], r["predicted_tracks"] - r["gt_tracks"])
              for r in rows(run, tau, min_ratio=2) if r["gt_tracks"]]
    if len(points) < 4:
        return None
    lengths = np.array([p[0] for p in points], float)
    excess = np.array([p[1] for p in points], float)
    phi, neg_delta = np.polyfit(lengths, excess, 1)
    resid = excess - (phi * lengths + neg_delta)
    ss_tot = float(((excess - excess.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None

    errs = [(r["window_frames"], r["signed_relative_error"])
            for r in rows(run, tau, min_ratio=2) if r["signed_relative_error"] is not None]
    observed = None
    for (l0, e0), (l1, e1) in zip(errs, errs[1:]):
        if e0 == 0 and e1 > 0:
            observed = l0
            break
        if e0 * e1 < 0:
            observed = l0 + (l1 - l0) * abs(e0) / (abs(e0) + abs(e1))
            break
    return {"phi": float(phi), "delta": float(-neg_delta), "r2": r2,
            "L_star_pred": float(-neg_delta / phi) if phi > 0 else None,
            "L_star_obs": observed}


print("=" * 80)
print("1. PERFECT DETECTOR (p=0, ground-truth boxes): the error is a range, not a number")
print("=" * 80)
print(f"{'video':<20}{'frames':>7}{'GT':>5}{'err@20f':>10}{'err@50f':>10}{'err@full':>10}"
      f"{'min over grid':>15}{'max over grid':>15}")
spans = []
for run in runs:
    if run["miss_rate"] != 0.0 or run["tracker"] != BASE:
        continue
    valid = [r for r in run["count_error_surface"] if r["signed_relative_error"] is not None]
    lo = min(valid, key=lambda r: r["signed_relative_error"])
    hi = max(valid, key=lambda r: r["signed_relative_error"])
    full = rows(run, tau=1)[-1]

    def fmt(length):
        row = cell(run, length, 1)
        return f"{row['signed_relative_error']:+.3f}" if row else "-"
    spans.append((hi["signed_relative_error"] - lo["signed_relative_error"]))
    print(f"{run['video']:<20}{run['frames']:>7}{full['gt_tracks']:>5}"
          f"{fmt(20):>10}{fmt(50):>10}{full['signed_relative_error']:>+10.3f}"
          f"{lo['signed_relative_error']:>+11.3f} @L{lo['window_frames']:<3}"
          f"{hi['signed_relative_error']:>+11.3f} @L{hi['window_frames']:<3}")
print(f"\n  the (L, tau) grid alone spans a median of {np.median(spans):.2f} in relative error "
      f"on one video with one system and a PERFECT detector")

print()
print("=" * 80)
print("2. EVERY SEQUENCE HAS AN (L, tau) THAT REPORTS ZERO ERROR  (p=0)")
print("=" * 80)
exact = 0
for run in runs:
    if run["miss_rate"] != 0.0 or run["tracker"] != BASE:
        continue
    valid = [r for r in run["count_error_surface"]
             if r["signed_relative_error"] is not None
             and r["window_frames"] >= 2 * r["min_track_len"]]
    best = min(valid, key=lambda r: abs(r["signed_relative_error"]))
    exact += abs(best["signed_relative_error"]) < 1e-9
    print(f"  {run['video']:<20} |err| {best['signed_relative_error']:+.3f} at "
          f"L={best['window_frames']:>4}, tau={best['min_track_len']}   "
          f"(predicted {best['predicted_tracks']} vs true {best['gt_tracks']})")
print(f"\n  {exact}/11 videos admit an EXACTLY zero-error (L, tau) in the non-degenerate regime")

print()
print("=" * 80)
print("3. DEGENERATE CORNER: tau > L forces the count to zero")
print("=" * 80)
degenerate = [r for run in runs for r in run["count_error_surface"]
              if r["window_frames"] < r["min_track_len"]
              and r["signed_relative_error"] is not None]
if degenerate:
    at_minus_one = sum(abs(r["signed_relative_error"] + 1.0) < 1e-9 for r in degenerate)
    print(f"  {at_minus_one}/{len(degenerate)} cells with tau > L report exactly -100% error.")
    print("  Reported as the degenerate regime and excluded from the linear fit (L >= 2*tau).")

print()
print("=" * 80)
print("4. DRIFT MODEL  P(L) - G(L) = phi*L - delta,  refitted on L >= 2*tau")
print("=" * 80)
r2s, pred, obs = [], [], []
for run in runs:
    for tau in sorted({r["min_track_len"] for r in run["count_error_surface"]}):
        fit = refit(run, tau)
        if not fit:
            continue
        if fit["r2"] is not None:
            r2s.append(fit["r2"])
        if fit["L_star_pred"] and fit["L_star_obs"] and 0 < fit["L_star_pred"] < 1000:
            pred.append(fit["L_star_pred"])
            obs.append(fit["L_star_obs"])
print(f"  linear fit R^2 over {len(r2s)} (run, tau) pairs: median {np.median(r2s):.3f}, "
      f"10th pct {np.percentile(r2s, 10):.3f}")
if pred:
    pred, obs = np.array(pred), np.array(obs)
    print(f"  zero-error length L* = delta/phi, {len(pred)} pairs: "
          f"corr {np.corrcoef(pred, obs)[0, 1]:.3f}, "
          f"median |pred-obs| {np.median(np.abs(pred - obs)):.1f} frames, "
          f"median observed L* {np.median(obs):.0f}")

print()
print("=" * 80)
print("5. phi RISES WITH MISS RATE, FALLS WITH tau, AND MOVES WITH track_buffer")
print("=" * 80)
table = defaultdict(list)
for run in runs:
    for tau in sorted({r["min_track_len"] for r in run["count_error_surface"]}):
        fit = refit(run, tau)
        if fit:
            table[(run["tracker"], run["miss_rate"], tau)].append(fit["phi"])
taus = sorted({key[2] for key in table})
for tracker in sorted({key[0] for key in table}):
    print(f"\n  {tracker}")
    print("    p      " + "".join(f"tau={t:<8}" for t in taus))
    for miss in sorted({key[1] for key in table if key[0] == tracker}):
        cells = "".join(f"{np.mean(table[(tracker, miss, t)]):<12.3f}" for t in taus)
        print(f"    {miss:<7.1f}{cells}")

print()
print("=" * 80)
print("6. DOES THE LAW SURVIVE SWAPPING THE ASSOCIATION COMPONENT?")
print("=" * 80)
for tracker in sorted({run["tracker"] for run in runs}):
    v20, v200 = [], []
    for run in runs:
        if run["tracker"] != tracker or run["miss_rate"] != 0.0:
            continue
        for length, bucket in ((20, v20), (200, v200)):
            row = cell(run, length, 1)
            if row and row["signed_relative_error"] is not None:
                bucket.append(row["signed_relative_error"])
    print(f"  {tracker:<38} mean err @20f {np.mean(v20):+.3f}   "
          f"@200f {np.mean(v200):+.3f}   (n={len(v200)})")
