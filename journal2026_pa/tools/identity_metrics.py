#!/usr/bin/env python3
"""HOTA, DetA, AssA and IDF1 for every arm of the whole-frame surface (TrackEval 1.3.0).

Two evaluation timelines, as in the September draft (common_timeline_hota_0920.py):
  * processed: the frames the arm processed, against the reference on those frames;
  * common:    every annotated frame. An identity seen at two CONSECUTIVE processed
               frames is carried across the skipped frames between them by linear
               interpolation (ByteTrack's tracklet interpolation, Zhang et al., ECCV
               2022); no extrapolation and no filling of gaps inside processed
               frames. The reference on all annotated frames is read from the
               interval-1 arm of the same seed and scale, which processed every one.
At interval 1 both timelines are the same frames, so they must agree to the digit.
Per sequence, HOTA is the mean over the 19 alpha thresholds; one arm is one sequence.

    .venv/bin/python cadence_journal_1119/tools/identity_metrics.py RUN_DIR OUT.json [FAMILY ...]

FAMILY defaults to surface_seed0 surface_seed1 surface_seed2.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from trackeval.metrics import HOTA, Identity

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FAMILIES = sys.argv[3:] or ["surface_seed0", "surface_seed1", "surface_seed2"]
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def record(gt_ids, gt_boxes, pred):
    """pred: list over frames of {track id: box}. Dense ids, TrackEval layout."""
    gmap = {g: j for j, g in enumerate(sorted({g for f in gt_ids for g in f}))}
    pmap = {p: j for j, p in enumerate(sorted({p for f in pred for p in f}))}
    g = [np.array([gmap[x] for x in f], dtype=int) for f in gt_ids]
    t = [np.array([pmap[x] for x in f], dtype=int) for f in pred]
    sims = [iou_matrix(gt_boxes[k], list(pred[k].values())) for k in range(len(gt_ids))]
    return {"num_timesteps": len(g), "num_gt_ids": len(gmap), "num_tracker_ids": len(pmap),
            "num_gt_dets": int(sum(len(x) for x in g)), "num_tracker_dets": int(sum(len(x) for x in t)),
            "gt_ids": g, "tracker_ids": t, "similarity_scores": sims}


def processed(v):
    pred = [dict(zip(ids, boxes)) for ids, boxes in zip(v["frame_predicted_ids"], v["frame_predicted_boxes"])]
    return record(v["frame_gt_ids"], v["frame_gt_boxes"], pred)


def common(v, full):
    names = full["frame_names"]
    index = {n: i for i, n in enumerate(names)}
    frame_of = [index[n] for n in v["frame_names"]]
    pred = [dict() for _ in names]
    obs = defaultdict(list)
    for k, (ids, boxes) in enumerate(zip(v["frame_predicted_ids"], v["frame_predicted_boxes"])):
        for i, b in zip(ids, boxes):
            pred[frame_of[k]][i] = b
            obs[i].append((frame_of[k], np.asarray(b, float)))
    nxt = dict(zip(frame_of, frame_of[1:]))
    for i, seq in obs.items():
        for (t0, b0), (t1, b1) in zip(seq, seq[1:]):
            if nxt.get(t0) != t1:
                continue
            for t in range(t0 + 1, t1):
                a = (t - t0) / (t1 - t0)
                pred[t][i] = (b0 + a * (b1 - b0)).tolist()
    return record(full["frame_gt_ids"], full["frame_gt_boxes"], pred)


def scores(data):
    h = HOTA().eval_sequence(data)
    i = Identity().eval_sequence(data)
    return {"HOTA": float(np.mean(h["HOTA"])), "DetA": float(np.mean(h["DetA"])),
            "AssA": float(np.mean(h["AssA"])), "IDF1": float(i["IDF1"])}


out = json.loads(OUT.read_text()) if OUT.exists() else {}
for fam in FAMILIES:
    for v in VIDEOS:
        for s in SIGMAS:
            full = json.loads((RUN / fam / f"{v}_s{s}_d1.json").read_text())["videos"][0]
            for d in DELTAS:
                key = f"{fam}|{v}|{s}|{d}"
                if key in out:
                    continue
                arm = full if d == 1 else json.loads((RUN / fam / f"{v}_s{s}_d{d}.json").read_text())["videos"][0]
                row = {"processed": scores(processed(arm)), "common": scores(common(arm, full))}
                if d == 1:
                    assert all(abs(row["processed"][k] - row["common"][k]) < 1e-9 for k in row["processed"]), key
                out[key] = row
                print(f"{key:40s} HOTA {row['processed']['HOTA']:.4f}/{row['common']['HOTA']:.4f} "
                      f"IDF1 {row['processed']['IDF1']:.4f}/{row['common']['IDF1']:.4f}", flush=True)
            OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
