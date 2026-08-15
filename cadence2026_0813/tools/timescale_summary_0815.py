#!/usr/bin/env python3
"""Summarise the timescale and gate controls against the published contrast.

Every pair below is read from one run, so the arms share the decode, the
detections, the checkpoint, the buffer and the scoring instants:

  rel     vs src       the published contrast, re-run here as a check
  rel     vs rel_dt    what the motion model's time step alone is worth
  rel_dt  vs src       the contrast once both arms advance by elapsed time
  rel_g03 vs src_g03   the contrast at an association gate of IoU 0.3
  rel_g05 vs src_g05   and at 0.5, against the default 0.2

Reported per pair: the pooled decomposition, the per-sequence paired median, a
sequence-level percentile interval and a flight-clustered one, and the direction
count. The sequence-level interval is the one Table III prints, so the published
pair doubles as a reproduction check on this script.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

FLIGHT = re.compile(r"^row_(\d+)")

PAIRS = [
    ("rel", "src", "published contrast"),
    ("rel", "rel_dt", "elapsed-time prediction, sparse arm only"),
    ("rel_dt", "src", "contrast with both arms on one clock"),
    ("rel_g03", "src_g03", "association gate IoU 0.3"),
    ("rel_g05", "src_g05", "association gate IoU 0.5"),
]


def flight_of(video: str) -> str:
    match = FLIGHT.match(video)
    if not match:
        raise SystemExit(f"cannot read a flight from {video}")
    return f"row {match.group(1)}"


def bootstrap_median(deltas, draws, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(deltas, dtype=float)
    medians = np.median(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def bootstrap_clustered(by_flight, draws, seed):
    rng = np.random.default_rng(seed)
    groups = list(by_flight.values())
    medians = np.empty(draws)
    for index in range(draws):
        picked = rng.choice(len(groups), size=len(groups), replace=True)
        pooled = [value for choice in picked for value in groups[choice]]
        medians[index] = float(np.median(pooled))
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 3, 5, 8])
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = []
    for path in args.results:
        records.extend(json.loads(path.read_text())["runs"])
    arms = sorted({r["arm"] for r in records})
    videos = sorted({r["video"] for r in records})
    print(f"{len(videos)} sequences, arms: {', '.join(arms)}")

    report = {"videos": videos, "arms": arms, "pooled": {}, "pairs": {},
              "kalman_steps": {}}

    for arm in arms:
        report["kalman_steps"][arm] = {
            "tracker_frames": sum(r["tracker_frames"] for r in records if r["arm"] == arm),
            "kalman_steps": sum(r["kalman_steps"] for r in records if r["arm"] == arm),
        }
        for tau in args.taus:
            terms = Counter()
            for record in records:
                if record["arm"] != arm:
                    continue
                one = record["decomposition"][str(tau)]
                if not one["identity_holds"]:
                    raise SystemExit(f"{arm}/{record['video']}: identity fails at tau={tau}")
                for key in ("P", "G", "U", "D", "M"):
                    terms[key] += one[key]
            G = terms["G"]
            report["pooled"][f"{arm}/tau{tau}"] = {
                "arm": arm, "tau": tau,
                "P": terms["P"], "G": G, "U": terms["U"], "D": terms["D"], "M": terms["M"],
                "signed_error": (terms["P"] - G) / G,
                "assigned_fraction": 1 - terms["M"] / G,
            }

    print(f"\n{'pair':>18s} {'tau':>3s} {'first':>8s} {'second':>8s} {'delta':>8s} "
          f"{'seq CI':>18s} {'flight CI':>18s} {'u/d/t':>9s}")
    for first, second, note in PAIRS:
        if first not in arms or second not in arms:
            print(f"skip {first} vs {second}: arm missing")
            continue
        for tau in args.taus:
            per_video = {}
            for record in records:
                if record["arm"] not in (first, second):
                    continue
                cell = per_video.setdefault(record["video"], {})
                cell[record["arm"]] = record["decomposition"][str(tau)]["signed_error"]
            deltas, by_flight = [], {}
            firsts, seconds = [], []
            for video in videos:
                cell = per_video[video]
                delta = cell[second] - cell[first]
                deltas.append(delta)
                firsts.append(cell[first])
                seconds.append(cell[second])
                by_flight.setdefault(flight_of(video), []).append(delta)
            low, high = bootstrap_median(deltas, args.draws, args.seed)
            clow, chigh = bootstrap_clustered(by_flight, args.draws, args.seed)
            entry = {
                "first": first, "second": second, "note": note, "tau": tau,
                "sequences": len(deltas),
                "first_median": float(np.median(firsts)),
                "second_median": float(np.median(seconds)),
                "delta_median": float(np.median(deltas)),
                "ci95_sequence": [low, high],
                "ci95_flight": [clow, chigh],
                "up": sum(1 for d in deltas if d > 0),
                "down": sum(1 for d in deltas if d < 0),
                "tie": sum(1 for d in deltas if d == 0),
                "flights": len(by_flight),
                "flights_all_positive": sum(
                    1 for values in by_flight.values() if all(v > 0 for v in values)),
            }
            report["pairs"][f"{first}->{second}/tau{tau}"] = entry
            print(f"{first + '->' + second:>18s} {tau:3d} {entry['first_median']:+8.3f} "
                  f"{entry['second_median']:+8.3f} {entry['delta_median']:+8.3f} "
                  f"[{low:+.2f},{high:+.2f}]".rjust(0)
                  + f" [{clow:+.2f},{chigh:+.2f}]"
                  + f" {entry['up']}/{entry['down']}/{entry['tie']}")

    print("\npooled at tau=1")
    for arm in arms:
        cell = report["pooled"][f"{arm}/tau1"]
        print(f"{arm:>10s} P={cell['P']:5d} G={cell['G']:5d} U={cell['U']:5d} "
              f"D={cell['D']:5d} M={cell['M']:5d} e={cell['signed_error']:+.4f} "
              f"1-M/G={cell['assigned_fraction']:.4f} "
              f"kalman={report['kalman_steps'][arm]['kalman_steps']} "
              f"frames={report['kalman_steps'][arm]['tracker_frames']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
