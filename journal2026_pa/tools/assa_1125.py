#!/usr/bin/env python3
"""HOTA, DetA and AssA for all 840 arms and for the oracle (ledger 10.28.17, review addendum of 25 November 2026).

identity_metrics.py (19 November) scored the 420 whole-frame arms with TrackEval 1.3.0 (results/identity_metrics.json).
This script scores the 420 crop arms of the archive (grapemots-protocol/cadence2026_0919/raw/retrain.tar.gz) and the
BoT-SORT oracle arms (reference boxes at score 0.9, runs/oracle_cadence_1119/botsort_gmc, offset 0) the same way:
processed frames only, TrackEval's HOTA metric averaged over its 19 alpha thresholds, one arm = one sequence
(record and scores copied from identity_metrics.py). It re-scores the whole-frame arms as a check and requires them to
reproduce identity_metrics.json to 1e-9.

Definitions (Luiten et al., 2021, IJCV 129:548-578): at each alpha, DetA = |TP| / (|TP| + |FN| + |FP|);
AssA = (1/|TP|) sum over TP pairs c of |TPA(c)| / (|TPA(c)| + |FNA(c)| + |FPA(c)|); HOTA = sqrt(DetA * AssA).
AssA falls when a track mixes bunches (FPA) and when a bunch is split over tracks (FNA), which the majority
assignment of the count decomposition does not score.

Summaries: AssA of the arms that counted within 10% of the reference (|e| <= 0.10 with e = (P - G_full)/G_full, the
rule of journal_numbers.py), its median over all arms, and the oracle at each interval.
    .venv/bin/python cadence_journal_1119/tools/assa_1125.py
"""
from __future__ import annotations

import io
import json
import statistics as st
import sys
import tarfile
from pathlib import Path

import numpy as np
from trackeval.metrics import HOTA

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

assert HAVE_SCIPY, "SciPy is required"
RUN = ROOT / "runs/fullframe_pponly_0922"
TAR = ROOT / "grapemots-protocol/cadence2026_0919/raw/retrain.tar.gz"
ORACLE = ROOT / "runs/oracle_cadence_1119/botsort_gmc"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)


def iou_matrix(a, b):  # identity_metrics.py
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def record(gt_ids, gt_boxes, pred):  # identity_metrics.py
    gmap = {g: j for j, g in enumerate(sorted({g for f in gt_ids for g in f}))}
    pmap = {p: j for j, p in enumerate(sorted({p for f in pred for p in f}))}
    g = [np.array([gmap[x] for x in f], dtype=int) for f in gt_ids]
    t = [np.array([pmap[x] for x in f], dtype=int) for f in pred]
    sims = [iou_matrix(gt_boxes[k], list(pred[k].values())) for k in range(len(gt_ids))]
    return {"num_timesteps": len(g), "num_gt_ids": len(gmap), "num_tracker_ids": len(pmap),
            "num_gt_dets": int(sum(len(x) for x in g)), "num_tracker_dets": int(sum(len(x) for x in t)),
            "gt_ids": g, "tracker_ids": t, "similarity_scores": sims}


def processed(v):  # identity_metrics.py
    pred = [dict(zip(ids, boxes)) for ids, boxes in zip(v["frame_predicted_ids"], v["frame_predicted_boxes"])]
    return record(v["frame_gt_ids"], v["frame_gt_boxes"], pred)


def scores(v):
    h = HOTA().eval_sequence(processed(v))
    return {"HOTA": float(np.mean(h["HOTA"])), "DetA": float(np.mean(h["DetA"])), "AssA": float(np.mean(h["AssA"]))}


# ------------------------------------------------------------------ the 840 arms, as journal_numbers.py loads them
REC = {}
with tarfile.open(TAR) as tar:
    for m in tar.getmembers():
        if not m.isfile():
            continue
        fold, arm = Path(m.name).name[:-5].split("__")
        v, seed = fold.rsplit("_s", 1)
        if v not in VIDEOS:
            continue
        s, d = arm.rsplit("_s", 1)[1].split("_d")
        REC[("crop", int(seed), v, int(s), int(d))] = json.load(io.TextIOWrapper(tar.extractfile(m)))["videos"][0]
for seed in SEEDS:
    for v in VIDEOS:
        for s in SIGMAS:
            for d in DELTAS:
                REC[("whole", seed, v, s, d)] = json.loads(
                    (RUN / f"surface_seed{seed}" / f"{v}_s{s}_d{d}.json").read_text())["videos"][0]
assert len(REC) == 840
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_batch8_1127  # noqa: E402  batch-8 retrain of crop PathPlanning_4_s2 (ledger 10.28.17, R rule)
crop_batch8_1127.apply(REC)
T = {k: decompose(r, 0.5, 1) for k, r in REC.items()}
GF = {v: T[("whole", 0, v, 3072, 1)]["G"] for v in VIDEOS}
err = lambda k: (T[k]["P"] - GF[k[2]]) / GF[k[2]]
cov = lambda k: 1 - (T[k]["M"] + GF[k[2]] - T[k]["G"]) / GF[k[2]]

S = {}
for i, k in enumerate(sorted(REC)):
    S[k] = scores(REC[k])
    if i % 60 == 0:
        print(i, k, {m: round(x, 3) for m, x in S[k].items()}, flush=True)

# check against the 19 November scores
I = json.loads((HERE / "results/identity_metrics.json").read_text())
dev = max(abs(S[("whole", sd, v, s, d)][m] - I[f"surface_seed{sd}|{v}|{s}|{d}"]["processed"][m])
          for sd in SEEDS for v in VIDEOS for s in SIGMAS for d in DELTAS for m in ("HOTA", "DetA", "AssA"))
assert dev < 1e-9, dev

near = [k for k in S if abs(err(k)) <= 0.10]
assert len(near) == 103, len(near)  # 108 with the published batch-4 checkpoint (ledger 10.28.17, R rule)
summ = lambda xs: dict(n=len(xs), min=round(min(xs), 3), median=round(st.median(xs), 3), max=round(max(xs), 3))
out = {"check_whole_vs_identity_metrics_max_abs_dev": dev,
       "near_zero": {m: summ([S[k][m] for k in near]) for m in ("AssA", "DetA", "HOTA")},
       "near_zero_coverage": summ([cov(k) for k in near]),
       "all_arms": {m: summ([S[k][m] for k in S]) for m in ("AssA", "DetA", "HOTA")},
       "by_family": {f: {m: summ([S[k][m] for k in S if k[0] == f]) for m in ("AssA", "DetA")} for f in ("crop", "whole")},
       "by_interval": {str(d): {m: round(st.median(S[k][m] for k in S if k[4] == d), 3) for m in ("AssA", "DetA", "HOTA")}
                       for d in DELTAS}}

# AssA against count error over all arms (Spearman, ties by average rank)
def ranks(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2; i = j + 1
    return r
keys = sorted(S)
out["spearman_AssA_vs_abs_e"] = round(float(np.corrcoef(ranks([S[k]["AssA"] for k in keys]),
                                                         ranks([abs(err(k)) for k in keys]))[0, 1]), 3)

# oracle: association alone (BoT-SORT, reference boxes at 0.9, offset 0)
orc = {}
for d in DELTAS:
    rows = {v: scores(json.loads((ORACLE / f"{v}_d{d}_o0.json").read_text())["videos"][0]) for v in VIDEOS}
    orc[str(d)] = {m: dict(min=round(min(r[m] for r in rows.values()), 3), median=round(st.median(r[m] for r in rows.values()), 3),
                           max=round(max(r[m] for r in rows.values()), 3)) for m in ("AssA", "DetA", "HOTA")}
out["oracle_botsort"] = orc
out["per_arm"] = {"|".join(map(str, k)): {m: round(x, 6) for m, x in S[k].items()} | {"e": round(err(k), 6), "coverage": round(cov(k), 6)}
                  for k, x in sorted(S.items())}

dest = HERE / "results/assa_1125.json"
dest.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "per_arm"}, indent=1))
