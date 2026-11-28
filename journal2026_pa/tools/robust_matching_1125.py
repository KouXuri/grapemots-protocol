"""Robustness of the cancellation findings to the identity-scoring rule (reviewer comment 5, 2026-11-25).

Reuses the paper's loader (calibration_transfer_1120.py via runpy: the 840 frozen per-frame records) and the
paper's decomposition (decompose_count_error.decompose). Only frame_matches is swapped for the variants:
  default  : Hungarian on IoU over all pairs, then pairs below the threshold dropped (the paper)
  gated    : pairs below the threshold made ineligible first (cost 1e6), then Hungarian -> maximum number of
             eligible matches, then maximum IoU (the CLEAR-MOT / py-motmetrics convention)
Thresholds 0.3 / 0.5 / 0.7 and minimum track length tau 1 / 3 (tau on predictions only, as in the paper).
Quantities as in journal_numbers.py / story_checks_1120.py: e=(P-G_full)/G_full, M_full=M+G_full-G,
Gamma=(U+D+M_full)/G_full, coverage=1-M_full/G_full.
"""
import contextlib, io, json, runpy, statistics as st, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment

with contextlib.redirect_stdout(io.StringIO()):
    NS = runpy.run_path("cadence_journal_1119/tools/calibration_transfer_1120.py", run_name="calib")
REC, GF = NS["REC"], NS["GF"]
dc = sys.modules.get("decompose_count_error")
assert dc is not None and dc.HAVE_SCIPY
default_fm = dc.frame_matches


def gated_fm(pred_boxes, pred_ids, gt_boxes, gt_ids, threshold):
    if not pred_ids or not gt_ids:
        return []
    s = dc.iou_matrix(np.asarray(pred_boxes, dtype=float), np.asarray(gt_boxes, dtype=float))
    cost = np.where(s >= threshold, -s, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(pred_ids[r], gt_ids[c]) for r, c in zip(rows, cols) if s[r, c] >= threshold]


def terms(t, v):
    G = GF[v]; Mf = t["M"] + G - t["G"]
    return dict(e=(t["P"] - G) / G, gross=(t["U"] + t["D"] + Mf) / G, cov=1 - Mf / G, U=t["U"], D=t["D"], M=Mf)


def run(fm, thr, tau):
    dc.frame_matches = fm
    try:
        return {k: terms(dc.decompose(r, thr, tau), k[2]) for k, r in REC.items()}
    finally:
        dc.frame_matches = default_fm


def near_summary(A):
    near = [x for x in A.values() if abs(x["e"]) <= 0.10]
    return dict(arms=len(A), near_zero=len(near), gross_min=round(min(x["gross"] for x in near), 4),
                gross_median=round(st.median(x["gross"] for x in near), 3),
                coverage_max=round(max(x["cov"] for x in near), 4))


out = {}
t0 = time.time()
base = run(default_fm, 0.5, 1)
out["default_0.5_tau1"] = near_summary(base)
print("default", out["default_0.5_tau1"], f"{time.time()-t0:.0f}s", flush=True)
g = run(gated_fm, 0.5, 1)
out["gated_0.5_tau1"] = near_summary(g)
diff = [max(abs(g[k][q] - base[k][q]) for q in ("U", "D", "M")) for k in base]
out["gated_vs_default"] = dict(arms_changed=sum(d > 0 for d in diff), max_abs_change_in_U_D_or_M=max(diff))
print("gated", out["gated_0.5_tau1"], out["gated_vs_default"], flush=True)
for thr in (0.3, 0.7):
    out[f"default_{thr}_tau1"] = near_summary(run(default_fm, thr, 1)); print(thr, out[f"default_{thr}_tau1"], flush=True)
out["default_0.5_tau3"] = near_summary(run(default_fm, 0.5, 3)); print("tau3", out["default_0.5_tau3"], flush=True)
json.dump(out, open("cadence_journal_1119/results/robust_matching_1125.json", "w"), indent=1)
print(json.dumps(out, indent=1))
