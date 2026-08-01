#!/usr/bin/env python3
"""U, D and M across the whole oracle grid, not just eight nested pipeline arms.

A reviewer's central objection to the ranking reversal is that it rests on eight
non-independent configurations on two videos. The oracle needs no model inference, so
the same decomposition can be computed over eleven sequences x four miss modes x five
rates x three seeds. Because the tracker is fed ground-truth boxes, each returned
track carries the index of the box it was matched to, so the predicted-track to
trajectory assignment is exact and needs no IoU threshold.
"""
from __future__ import annotations
import os
import json, sys, statistics
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools"))
import torch
from ultralytics.engine.results import Boxes
from oracle_count_surface import video_frames, read_resolutions
from track_grapemots_mot import build_tracker, load_gt_tracks

DATA = ROOT / "datasets/grapemots_det_721"
TRACKER = "bytetrack.yaml"
MODES = ["bernoulli", "block", "identity", "size"]
RATES = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 1, 2]
BLOCK = 15


def thin(gt_ids, gt_boxes, mode, p, rng, state, order):
    n = len(gt_ids)
    if n == 0 or p == 0:
        return np.ones(n, dtype=bool)
    if mode == "bernoulli":
        return rng.random(n) >= p
    if mode == "block":
        hazard = p / max(BLOCK * (1.0 - p), 1e-9)
        keep = []
        for t in gt_ids:
            if state["until"].get(t, -1) > state["i"]:
                keep.append(False)
            elif rng.random() < min(hazard, 1.0):
                state["until"][t] = state["i"] + BLOCK
                keep.append(False)
            else:
                keep.append(True)
        return np.array(keep, dtype=bool)
    return np.array([t not in state["dropped"] for t in gt_ids], dtype=bool)


def run(frames, size, mode, p, seed):
    width, height = size
    rng = np.random.default_rng(seed)
    tracker = build_tracker(TRACKER, frame_rate=15)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    # entity-level modes need the identity set up front
    areas, allids = defaultdict(list), set()
    for f in frames:
        ids, boxes = load_gt_tracks(f, width, height)
        for t, b in zip(ids, boxes):
            areas[t].append((b[2] - b[0]) * (b[3] - b[1])); allids.add(t)
    state = {"until": {}, "dropped": set(), "i": 0}
    if mode == "identity" and p > 0:
        ordered = sorted(allids)
        rng.shuffle(ordered)
        state["dropped"] = set(ordered[:int(round(p * len(ordered)))])
    elif mode == "size" and p > 0:
        ordered = sorted(allids, key=lambda t: statistics.median(areas[t]))
        state["dropped"] = set(ordered[:int(round(p * len(ordered)))])

    cover = defaultdict(Counter)       # gt id -> Counter(pred id -> frames)
    gt_seen, pred_seen = set(), set()
    for i, f in enumerate(frames):
        state["i"] = i
        ids, boxes = load_gt_tracks(f, width, height)
        gt_seen.update(ids)
        keep = thin(ids, boxes, mode, p, rng, state, None)
        kept_ids = [t for t, k in zip(ids, keep) if k]
        kb = boxes[keep] if len(boxes) else boxes
        data = (np.concatenate([kb.astype(np.float32),
                                np.full((len(kb), 1), 0.9, np.float32),
                                np.zeros((len(kb), 1), np.float32)], axis=1)
                if len(kb) else np.empty((0, 6), np.float32))
        tracks = tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32),
                                      (height, width)), blank)
        for row in tracks:
            pid, det = int(row[4]), int(row[7])
            pred_seen.add(pid)
            if 0 <= det < len(kept_ids):
                cover[kept_ids[det]][pid] += 1
    # assign each predicted track to the trajectory it covers in the most frames
    best = {}
    for g, c in cover.items():
        for pid, n in c.items():
            if pid not in best or n > best[pid][1]:
                best[pid] = (g, n)
    owner = defaultdict(int)
    for pid, (g, _) in best.items():
        owner[g] += 1
    U = len(pred_seen) - len(best)
    D = sum(v - 1 for v in owner.values() if v > 0)
    M = len(gt_seen) - len(owner)
    return {"P": len(pred_seen), "G": len(gt_seen), "U": U, "D": D, "M": M,
            "e": (len(pred_seen) - len(gt_seen)) / len(gt_seen)}


def main():
    sizes = read_resolutions(DATA)
    videos = sorted(sizes)
    out = []
    for mode in MODES:
        for p in RATES:
            seeds = SEEDS if p > 0 else SEEDS[:1]
            for seed in seeds:
                for v in videos:
                    r = run(video_frames(DATA, v), sizes[v], mode, p, seed)
                    r.update(video=v, mode=mode, p=p, seed=seed)
                    out.append(r)
            print(f"  {mode} p={p}: {len(out)} rows", flush=True)
    dest = ROOT / "runs/cbdcom2026_queue/results/oracle_decomposition.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    bad = [r for r in out if r["P"] - r["G"] != r["U"] + r["D"] - r["M"]]
    print(f"identity holds on {len(out)-len(bad)}/{len(out)} runs")
    print("wrote", dest)


if __name__ == "__main__":
    main()
