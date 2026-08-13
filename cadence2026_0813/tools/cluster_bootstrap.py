#!/usr/bin/env python3
"""The cadence contrast resampled by flight rather than by sequence.

The published interval resamples the 17 model-unseen sequences independently, but
they are not independent: each flight recorded one side of one row and
contributed several sequences, so a sequence-level bootstrap treats repeated
measurements within a flight as replicates of it. Four flights is too few for a
narrow interval, which is the point. Reported here are the per-flight paired
differences, whose direction is what the design can actually support, beside a
cluster bootstrap that says how little the corpus constrains the population.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

FLIGHT = re.compile(r"^row_(\d+)")


def flight_of(video: str) -> str:
    match = FLIGHT.match(video)
    if not match:
        raise SystemExit(f"cannot read a flight from {video}")
    return f"row {match.group(1)}"


def load(paths: list[Path], released: str, source: str, tau: int):
    per_video: dict[str, dict[str, float]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for record in payload["runs"]:
            if record["arm"] not in (released, source):
                continue
            cell = per_video.setdefault(record["video"], {})
            cell[record["arm"]] = record["decomposition"][str(tau)]["signed_error"]
    rows = []
    for video, cell in sorted(per_video.items()):
        if released not in cell or source not in cell:
            raise SystemExit(f"{video}: missing an arm")
        rows.append({
            "video": video,
            "flight": flight_of(video),
            "released": cell[released],
            "source": cell[source],
            "delta": cell[source] - cell[released],
        })
    return rows


def bootstrap(values: list[list[float]], draws: int, seed: int):
    """Percentile interval for the median paired difference, resampling clusters."""
    rng = np.random.default_rng(seed)
    clusters = np.arange(len(values))
    medians = np.empty(draws)
    for index in range(draws):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        pooled = [value for choice in picked for value in values[choice]]
        medians[index] = float(np.median(pooled))
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--released-arm", default="rel_buf30")
    parser.add_argument("--source-arm", default="src_buf30")
    parser.add_argument("--taus", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = {"released_arm": args.released_arm, "source_arm": args.source_arm,
               "draws": args.draws, "seed": args.seed, "by_tau": {}}
    for tau in args.taus:
        rows = load(args.results, args.released_arm, args.source_arm, tau)
        by_flight: dict[str, list[float]] = {}
        for row in rows:
            by_flight.setdefault(row["flight"], []).append(row["delta"])
        order = sorted(by_flight)
        values = [by_flight[flight] for flight in order]
        low, high = bootstrap(values, args.draws, args.seed)
        sequence_median = float(np.median([row["delta"] for row in rows]))
        flights = {
            flight: {
                "sequences": len(by_flight[flight]),
                "median_delta": float(np.median(by_flight[flight])),
                "min_delta": float(min(by_flight[flight])),
                "max_delta": float(max(by_flight[flight])),
                "all_positive": bool(all(value > 0 for value in by_flight[flight])),
            }
            for flight in order
        }
        payload["by_tau"][str(tau)] = {
            "sequences": len(rows),
            "flights": flights,
            "sequence_level_median_delta": sequence_median,
            "cluster_bootstrap_ci95": [low, high],
            "flights_all_positive": sum(1 for f in flights.values() if f["all_positive"]),
            "flight_count": len(flights),
        }
        print(f"tau={tau}: median delta {sequence_median:+.4f}, "
              f"cluster CI [{low:+.3f},{high:+.3f}], "
              f"{payload['by_tau'][str(tau)]['flights_all_positive']}/{len(flights)} "
              f"flights all-positive", flush=True)
        for flight in order:
            block = flights[flight]
            print(f"   {flight}: n={block['sequences']} median={block['median_delta']:+.4f} "
                  f"range [{block['min_delta']:+.4f},{block['max_delta']:+.4f}]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
