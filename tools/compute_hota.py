#!/usr/bin/env python3
"""HOTA, DetA and AssA for the eight tracking arms, from the stored per-frame boxes.

Reviewers asked for the metric family designed for the association question this
paper is about. No re-inference is needed: every arm JSON already carries
frame_gt_boxes / frame_gt_ids and frame_predicted_boxes / frame_predicted_ids, so
the similarity matrices can be rebuilt offline and handed to TrackEval.
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np

RES = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve() / "results"
from trackeval.metrics import HOTA  # noqa: E402

ARMS = [
    ("Tiles, conf 0.55",   "arm_conf0.55.json"),
    ("Resize + BoT-SORT",  "arm_resize.json"),
    ("Tiles, conf 0.40",   "arm_conf0.40.json"),
    ("Tiles + IoS merge",  "arm_ios_tiled.json"),
    ("Tiles + BoT-SORT",   "arm_botsort_tiled.json"),
    ("Tiles + YOLO11s",    "arm_yolo11s_tiled.json"),
    ("Tiles + ByteTrack",  "arm_bytetrack_tiled.json"),
    ("Tiles + ReID",       "arm_reid.json"),
]


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def build(entries):
    """Pool consecutive videos into one sequence-like record, ids offset per video."""
    gt_ids, tk_ids, sims = [], [], []
    gt_off = tk_off = 0
    for e in entries:
        g_ids, g_box = e["frame_gt_ids"], e["frame_gt_boxes"]
        p_ids, p_box = e["frame_predicted_ids"], e["frame_predicted_boxes"]
        gmax = max([max(f) for f in g_ids if f] + [-1]) + 1
        pmax = max([max(f) for f in p_ids if f] + [-1]) + 1
        for t in range(len(g_ids)):
            gt_ids.append(np.array([i + gt_off for i in g_ids[t]], dtype=int))
            tk_ids.append(np.array([i + tk_off for i in p_ids[t]], dtype=int))
            sims.append(iou_matrix(g_box[t], p_box[t]))
        gt_off += gmax; tk_off += pmax
    return {
        "num_timesteps": len(gt_ids),
        "num_gt_ids": gt_off, "num_tracker_ids": tk_off,
        "num_gt_dets": int(sum(len(x) for x in gt_ids)),
        "num_tracker_dets": int(sum(len(x) for x in tk_ids)),
        "gt_ids": gt_ids, "tracker_ids": tk_ids, "similarity_scores": sims,
    }


def main():
    metric = HOTA()
    out = {}
    print(f"{'arm':22s} {'HOTA':>7s} {'DetA':>7s} {'AssA':>7s} {'LocA':>7s}")
    for label, fname in ARMS:
        d = json.loads((RES / fname).read_text())
        res = metric.eval_sequence(build(d["videos"]))
        row = {k: float(np.mean(res[k])) for k in ("HOTA", "DetA", "AssA", "LocA")}
        out[label] = row
        print(f"{label:22s} {row['HOTA']:7.4f} {row['DetA']:7.4f} "
              f"{row['AssA']:7.4f} {row['LocA']:7.4f}")
    (RES / "hota_arms.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", RES / "hota_arms.json")


if __name__ == "__main__":
    main()
