"""GrapeMOTS re-scored under the design of the 2021 cadence intervention.

Input: the archived per-frame outputs of the ten out-of-fold GrapeMOTS sequences
(grapemots-protocol cadence2026_0919/raw/lovo_surface.tar.gz): each sequence read by
a leave-one-video-out YOLO26s checkpoint, whole frame at scale sigma, BoT-SORT with
cfg/trackers/botsort_gmc.yaml, at processing interval d in annotated frames.
Each archived arm is scored at the frames it processed. Here the dense arm (d=1) is
scored only at frames 0, k, 2k, ... -- the frames the d=k arm processed -- so both
arms share pixels, checkpoint, tracker, reference and scoring instants and differ
only in what the tracker saw between them. No GPU, no re-tracking.

    python3 gm_matched.py DECOMPOSE_DIR RAW_DIR OUT.json
"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from decompose_count_error import decompose

RAW = Path(sys.argv[2])
PER_FRAME = ("frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes")
SCALES = (1536, 2048, 2560, 3072, 3840)
KS = (2, 4, 8)
VIDEOS = sorted({p.name.split("_s")[0] for p in RAW.glob("*_s*_d*.json")})

def video(name, s, d):
    vids = json.loads((RAW / f"{name}_s{s}_d{d}.json").read_text())["videos"]
    assert len(vids) == 1 and vids[0]["video"] == name
    return vids[0]

def subset(v, k):
    out = dict(v)
    for key in PER_FRAME:
        out[key] = v[key][0::k]
    return out

cells = []
for name in VIDEOS:
    for s in SCALES:
        dense = video(name, s, 1)
        full = decompose(dense, 0.5, 1)
        for k in KS:
            sparse = video(name, s, k)
            dsub = subset(dense, k)
            assert len(dsub["frame_gt_ids"]) == len(sparse["frame_gt_ids"]), (name, s, k)
            assert dsub["frame_gt_ids"] == sparse["frame_gt_ids"], (name, s, k)     # same instants
            a, b = decompose(dsub, 0.5, 1), decompose(sparse, 0.5, 1)
            assert a["G"] == b["G"]
            cells.append({"video": name, "group": "multi-view" if name.startswith("PathPlanning") else "frontal",
                          "sigma": s, "k": k, "G": a["G"],
                          "dense": {t: a[t] for t in ("P", "U", "D", "M")},
                          "sparse": {t: b[t] for t in ("P", "U", "D", "M")},
                          "e_dense": (a["P"] - a["G"]) / a["G"], "e_sparse": (b["P"] - b["G"]) / b["G"],
                          "full_rate": {"P": full["P"], "G": full["G"], "U": full["U"], "D": full["D"], "M": full["M"]}})

def summary(rows):
    up = sum(r["dense"]["P"] > r["sparse"]["P"] for r in rows)
    down = sum(r["dense"]["P"] < r["sparse"]["P"] for r in rows)
    opp = [r for r in rows if r["e_dense"] > 0 > r["e_sparse"]]
    G = sum(r["G"] for r in rows)
    pd = sum(r["dense"]["P"] for r in rows); ps = sum(r["sparse"]["P"] for r in rows)
    md = sum(r["dense"]["M"] for r in rows); ms = sum(r["sparse"]["M"] for r in rows)
    return {"n": len(rows), "rises": up, "falls": down, "ties": len(rows) - up - down,
            "opposite_sign": len(opp), "opposite_sign_sequences": sorted({r["video"] for r in opp}),
            "pooled_e_dense": (pd - G) / G, "pooled_e_sparse": (ps - G) / G,
            "coverage_dense": 1 - md / G, "coverage_sparse": 1 - ms / G,
            "median_delta": statistics.median(r["e_dense"] - r["e_sparse"] for r in rows)}

out = {"videos": VIDEOS, "cells": cells, "all": summary(cells),
       "by_group": {g: summary([r for r in cells if r["group"] == g]) for g in ("multi-view", "frontal")},
       "by_sigma_k": {f"s{s}_k{k}": summary([r for r in cells if r["sigma"] == s and r["k"] == k])
                      for s in SCALES for k in KS}}
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
a = out["all"]
print(f"{len(VIDEOS)} sequences, {a['n']} paired comparisons: rises {a['rises']}, falls {a['falls']}, ties {a['ties']}; "
      f"opposite sign (dense>0>sparse) in {a['opposite_sign']} cells, sequences {a['opposite_sign_sequences']}")
for g, v in out["by_group"].items():
    print(f"  {g:10}: n={v['n']} rises={v['rises']} falls={v['falls']} opposite={v['opposite_sign']} {v['opposite_sign_sequences']}")
print(f"{'cell':12} {'e dense':>8} {'e sparse':>8} {'cov d':>6} {'cov s':>6} rises opp")
for k, v in out["by_sigma_k"].items():
    print(f"{k:12} {v['pooled_e_dense']:+8.3f} {v['pooled_e_sparse']:+8.3f} {v['coverage_dense']:6.3f} {v['coverage_sparse']:6.3f} {v['rises']:2d}/{v['n']} {v['opposite_sign']}")
