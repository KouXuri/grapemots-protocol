#!/usr/bin/env python3
"""HOTA, DetA and AssA for every row of the configuration table, one cohort.

The count table ranks configurations by a signed error. A reviewer is entitled to
ask what the association metric family says about the same rows, and the answer
is already in the stored per-frame boxes: no re-inference is needed. All eleven
rows are the same six out-of-fold 2024 videos under leave-one-video-out, so the
column is comparable with the U, D, M columns beside it.

Re-derives P, G, U, D, M as well, so the cohort is proved identical to the one in
the table rather than assumed to be.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", Path.cwd()))
STRIDE = ROOT / "runs/stride_and_cache_0809/results"
REBUTTAL = ROOT / "runs/rebuttal_0811/results"
OUT = ROOT / "runs/decomp_0812/results/hota_assoc_rows.json"
VIDEOS = ["PathPlanning_2", "PathPlanning_4", "PathPlanning_5",
          "PathPlanning_6", "PathPlanning_7", "PathPlanning_8"]

# label, arm token, directory holding cached_{video}_{arm}.json
ARMS = [
    ("ByteTrack, buffer 60", "assoc_buf60",  STRIDE),
    ("BoT-SORT, GMC off",    "assoc_nogmc",  STRIDE),
    ("BoT-SORT + ReID",      "reid",         STRIDE),
    ("ByteTrack, buffer 30", "bytetrack",    STRIDE),
    ("ByteTrack, buffer 10", "assoc_buf10",  STRIDE),
]

from trackeval.metrics import HOTA  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from decompose_count_error import decompose  # noqa: E402


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def build(entries):
    """Six videos as one sequence-like record, identities offset per video.

    Tracker ids come from a global counter and are sparse, so they are densely
    renumbered per video first. That is a relabelling and leaves HOTA unchanged,
    but it keeps the id-by-id matrices at the number of tracks that exist rather
    than at the largest id ever issued.
    """
    gt_ids, tk_ids, sims = [], [], []
    gt_off = tk_off = 0
    for e in entries:
        g_ids, g_box = e["frame_gt_ids"], e["frame_gt_boxes"]
        p_ids, p_box = e["frame_predicted_ids"], e["frame_predicted_boxes"]
        gmap = {i: k for k, i in enumerate(sorted({i for f in g_ids for i in f}))}
        pmap = {i: k for k, i in enumerate(sorted({i for f in p_ids for i in f}))}
        for t in range(len(g_ids)):
            gt_ids.append(np.array([gmap[i] + gt_off for i in g_ids[t]], dtype=int))
            tk_ids.append(np.array([pmap[i] + tk_off for i in p_ids[t]], dtype=int))
            sims.append(iou_matrix(g_box[t], p_box[t]))
        gt_off += len(gmap)
        tk_off += len(pmap)
    return {
        "num_timesteps": len(gt_ids),
        "num_gt_ids": gt_off,
        "num_tracker_ids": tk_off,
        "num_gt_dets": int(sum(len(x) for x in gt_ids)),
        "num_tracker_dets": int(sum(len(x) for x in tk_ids)),
        "gt_ids": gt_ids,
        "tracker_ids": tk_ids,
        "similarity_scores": sims,
    }


def main() -> None:
    metric = HOTA()
    out = {}
    print(f"{'row':22s} {'e':>8s} {'assigned':>9s} {'HOTA':>7s} {'DetA':>7s} {'AssA':>7s}")
    for label, token, directory in ARMS:
        entries = []
        for video in VIDEOS:
            path = directory / f"cached_{video}_{token}.json"
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            payload = json.loads(path.read_text())
            entries.extend(payload["videos"])
        if len(entries) != len(VIDEOS):
            raise SystemExit(f"{label}: {len(entries)} video records, expected {len(VIDEOS)}")

        terms = Counter()
        for entry in entries:
            one = decompose(entry, 0.5, 1)
            if not one["identity_holds"]:
                raise SystemExit(f"{label}/{entry['video']}: P-G != U+D-M")
            for key in ("P", "G", "U", "D", "M"):
                terms[key] += one[key]
        e = (terms["P"] - terms["G"]) / terms["G"]
        assigned = 1 - terms["M"] / terms["G"]

        res = metric.eval_sequence(build(entries))
        row = {k: float(np.mean(res[k])) for k in ("HOTA", "DetA", "AssA", "LocA")}
        row.update({k: int(terms[k]) for k in ("P", "G", "U", "D", "M")})
        row["signed_error"] = e
        row["assigned_fraction"] = assigned
        out[label] = row
        print(f"{label:22s} {e:+8.4f} {assigned:9.4f} {row['HOTA']:7.4f} "
              f"{row['DetA']:7.4f} {row['AssA']:7.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"videos": VIDEOS, "rows": out}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
