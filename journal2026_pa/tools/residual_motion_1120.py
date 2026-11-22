#!/usr/bin/env python3
"""Why identities break while a bunch is in view: residual motion after compensation.

For every reference trajectory present in two consecutive processed frames, its previous
box is carried into the current frame by the cached GMC warp exactly as Ultralytics'
STrack.multi_gmc moves a track state (R applied to the centre and to (w, h), then t added
to the centre; runs/error_sources_1120/gmc_cache, the library's own warps), and its IoU
with the current reference box is taken. BoT-SORT's first stage needs IoU x 0.9 >= 0.2 for
an oracle box, i.e. IoU >= 0.222. The Kalman velocity is ignored, so this is the overlap the
motion model starts from, not the one it ends with. Also recorded: IoU without the warp and
the area ratio of consecutive boxes (shape jumps from occlusion change).
    .venv/bin/python cadence_journal_1119/tools/residual_motion_1120.py
"""
from __future__ import annotations
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import iou_matrix  # noqa: E402
ORC = ROOT / "runs/oracle_cadence_1119/botsort_gmc"
CACHE = ROOT / "runs/error_sources_1120/gmc_cache"
ALL11 = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]
GATE = 0.2 / 0.9


def warp_box(b, H):
    R, t = H[:2, :2], H[:2, 2]
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
    c = R @ np.array([cx, cy]) + t
    wh = R @ np.array([w, h])
    return [c[0] - wh[0] / 2, c[1] - wh[1] / 2, c[0] + wh[0] / 2, c[1] + wh[1] / 2]


def iou(a, b):
    return float(iou_matrix(np.array([a], float), np.array([b], float))[0, 0])


reent = json.loads((ROOT / "cadence_journal_1119/results/reentry_analysis_1120.json").read_text())
out = {}
for d in (1, 2, 4, 8):
    for v in ALL11:
        rec = json.loads((ORC / f"{v}_d{d}_o0.json").read_text())["videos"][0]
        W = np.load(CACHE / f"{v}_d{d}_o0.npy")
        assert len(W) == len(rec["frame_gt_ids"])
        steps = beyond = beyond_raw = jumps = 0
        ious = []
        for i in range(1, len(W)):
            prev = dict(zip(rec["frame_gt_ids"][i - 1], rec["frame_gt_boxes"][i - 1]))
            for g, b in zip(rec["frame_gt_ids"][i], rec["frame_gt_boxes"][i]):
                if g not in prev:
                    continue
                pb = prev[g]
                x = iou(warp_box(pb, W[i]), b)
                steps += 1
                ious.append(x)
                beyond += x < GATE
                beyond_raw += iou(pb, b) < GATE
                a0 = (pb[2] - pb[0]) * (pb[3] - pb[1]); a1 = (b[2] - b[0]) * (b[3] - b[1])
                jumps += a0 > 0 and a1 > 0 and abs(math.log(a1 / a0)) > math.log(1.5)
        out[f"{v}|d{d}"] = dict(steps=steps, beyond_gate_after_gmc=round(beyond / steps, 4),
                                beyond_gate_raw=round(beyond_raw / steps, 4),
                                area_jump_gt_1p5=round(jumps / steps, 4),
                                median_iou_after_gmc=round(st.median(ious), 3))
# per-video switches in view at the same 15 Hz (PP2-8 at d1, the 30 Hz videos at d2), phase 0 only
# from the oracle files directly: duplicates per reference trajectory at that interval
sys.path.insert(0, str(ROOT / "cadence_journal_1119/tools"))
rows = []
for v in ALL11:
    d = 1 if v not in ("NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3", "PathPlanning_1") else 2
    from decompose_count_error import decompose  # noqa: E402
    rec = json.loads((ORC / f"{v}_d{d}_o0.json").read_text())["videos"][0]
    t = decompose(rec, 0.5, 1)
    rows.append((v, d, t["D"] / t["G"], out[f"{v}|d{d}"]["beyond_gate_after_gmc"], out[f"{v}|d{d}"]["area_jump_gt_1p5"]))
out["at_15hz"] = [dict(video=v, delta=d, D_per_G=round(x, 3), beyond_gate_after_gmc=b, area_jump=j) for v, d, x, b, j in rows]
out["spearman_D_vs_beyond_gate_15hz"] = round(spearmanr([r[2] for r in rows], [r[3] for r in rows])[0], 3)
out["spearman_D_vs_area_jump_15hz"] = round(spearmanr([r[2] for r in rows], [r[4] for r in rows])[0], 3)
(ROOT / "cadence_journal_1119/results/residual_motion_1120.json").write_text(json.dumps(out, indent=1))
for r in out["at_15hz"]:
    print(r)
print("spearman D/G vs beyond-gate:", out["spearman_D_vs_beyond_gate_15hz"], " vs area jump:", out["spearman_D_vs_area_jump_15hz"])
for d in (1, 2, 4, 8):
    pp = [out[f"PathPlanning_{i}|d{d}"]["beyond_gate_after_gmc"] for i in range(2, 9)]
    npv = [out[f"NoPathPlanning_{i}|d{d}"]["beyond_gate_after_gmc"] for i in range(1, 4)]
    print(f"d{d}: PP2-8 beyond gate {min(pp):.3f}-{max(pp):.3f}; NP {min(npv):.3f}-{max(npv):.3f}")
