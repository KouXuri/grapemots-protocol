#!/usr/bin/env python3
"""Reviewer question 1: what observation-loss rate does each miss mode actually apply?

The paper puts all four modes on one axis labelled p. Only Bernoulli drops each
observation with probability p; block uses a hazard approximation, and the two
entity-level modes drop round(p * n_identities) whole identities, which removes a
share of observations set by how long those identities happen to live.

This replays the thinning of tools/oracle_master.py without running the tracker,
so it is seconds rather than hours, and reports the realised rates.
"""
from __future__ import annotations

import os
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools"))
from oracle_master import (  # noqa: E402
    choose_dropped_identities,
    identity_areas,
    read_resolutions,
    video_frames,
)
from track_grapemots_mot import load_gt_tracks  # noqa: E402

DATA = ROOT / "datasets" / "grapemots_det_721"
VIDEOS = [
    "PathPlanning_1", "PathPlanning_2", "PathPlanning_3", "PathPlanning_4",
    "PathPlanning_5", "PathPlanning_6", "PathPlanning_7", "PathPlanning_8",
    "NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3",
]
RATES = [0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 1, 2]
BLOCK_LEN = 15


def realised_loss(frames, size, mode, miss, seed, areas):
    """Return (dropped observations, total observations, dropped identities)."""
    width, height = size
    rng = np.random.default_rng(seed)
    dropped_ids = choose_dropped_identities(mode, areas, miss, rng) \
        if mode in ("identity", "size") else set()
    blocked_until: dict[int, int] = {}
    total = dropped = 0
    for index, path in enumerate(frames):
        gt_ids, _ = load_gt_tracks(path, width, height)
        if not len(gt_ids):
            continue
        total += len(gt_ids)
        if mode == "block":
            hazard = miss / max(BLOCK_LEN * (1.0 - miss), 1e-9)
            for track in gt_ids:
                if blocked_until.get(track, -1) > index:
                    dropped += 1
                elif rng.random() < min(hazard, 1.0):
                    blocked_until[track] = index + BLOCK_LEN
                    dropped += 1
        elif mode in ("identity", "size"):
            dropped += sum(1 for t in gt_ids if int(t) in dropped_ids)
        else:
            dropped += int((rng.random(len(gt_ids)) < miss).sum())
    return dropped, total, len(dropped_ids)


def main() -> None:
    sizes = read_resolutions(DATA)
    out: dict = {}
    for video in VIDEOS:
        frames = video_frames(DATA, video)
        size = sizes[video]
        areas = identity_areas(frames, *size)
        n_ids = len(areas)
        for mode in ("bernoulli", "block", "identity", "size"):
            for miss in RATES:
                seeds = [0] if mode == "size" else SEEDS
                rates, id_counts = [], []
                for seed in seeds:
                    dropped, total, n_dropped = realised_loss(
                        frames, size, mode, miss, seed, areas)
                    rates.append(dropped / total if total else 0.0)
                    id_counts.append(n_dropped)
                out.setdefault(mode, {}).setdefault(f"{miss}", {})[video] = {
                    "realised_obs_loss": float(np.mean(rates)),
                    "identities_dropped": float(np.mean(id_counts)),
                    "identities_total": n_ids,
                }
        print(f"  {video} done", flush=True)

    dest = ROOT / "results/realised_loss_rates.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}\n")
    print("nominal p -> realised observation-loss rate (median over 11 videos)")
    print(f"{'mode':12s}" + "".join(f"{r:>10.1f}" for r in RATES))
    for mode in ("bernoulli", "block", "identity", "size"):
        row = [float(np.median([v["realised_obs_loss"]
                                for v in out[mode][f"{r}"].values()])) for r in RATES]
        print(f"{mode:12s}" + "".join(f"{x:>10.3f}" for x in row))
    print()
    print("identities removed (median over 11 videos, of ~n identities)")
    for mode in ("identity", "size"):
        row = [float(np.median([v["identities_dropped"]
                                for v in out[mode][f"{r}"].values()])) for r in RATES]
        print(f"{mode:12s}" + "".join(f"{x:>10.1f}" for x in row))


if __name__ == "__main__":
    main()
