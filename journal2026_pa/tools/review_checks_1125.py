#!/usr/bin/env python3
"""Numbers added to the manuscript after the pre-submission review of 2026-11-25 (ledger 10.28.17, review addendum).

Every value is recomputed from frozen outputs with the paper's own code; nothing is re-run on a GPU.
  miss_alone_per_vine    the error substitution at sigma 3072 (runs/error_sources_1120/substitution, six detector sets,
                         seven vines, four intervals): |e| with only the missed boxes supplied against |e| as detected
  reentry_absences       reference trajectories of PP2-PP8 at 15 Hz (frame_gt_ids of the BoT-SORT oracle arms,
                         runs/oracle_cadence_1119): absences of 30 or more annotated frames and of 150 or more
  never_shown            trajectories absent from every processed frame, per interval (seed-0 whole-frame arms at 3072,
                         the frames are the same for every detector)
  calibration_configs    per configuration (120), leave-one-vine-out ratio and straight-line calibration against the mean
                         reference count of the other six vines, both summarised by median and interpolated 90th
                         percentile (calibration_transfer_1120.py: P, GF, CONFIGS, q), and the configurations that beat
                         the baseline on both
  independent_vines      independent detectors (vine_calibration_1122.json): vines whose two-vine ratio error is below the
                         error of the two-vine mean, at 15 and 1.9 Hz
  recall_vs_error        (second pass) Spearman rho between the frame recall of the six detector sets and their net error
  vine_bunches           (second pass) the vine-level errors of Results 4 in bunches per vine
    .venv/bin/python cadence_journal_1119/tools/review_checks_1125.py
"""
from __future__ import annotations

import contextlib
import io
import json
import runpy
import statistics as st
from pathlib import Path
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_batch8_1127  # noqa: E402  batch-8 retrain of crop PathPlanning_4_s2 (ledger 10.28.17, R rule)

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
out = {}

# ------------------------------------------------------------------ miss alone, per vine
pairs = []
for p in sorted((ROOT / "runs/error_sources_1120/substitution").glob("*/*__*_s3072.json")):
    r = crop_batch8_1127.read_substitution(p)
    if "arms" not in r:
        continue
    for d in ("1", "2", "4", "8"):
        v = r["arms"][d]["values"]
        pairs.append(dict(set=p.stem.split("__")[0], video=r["video"], interval=int(d),
                          e_detected=v["none"]["e"], e_miss_supplied=v["miss"]["e"]))
worse = [x for x in pairs if abs(x["e_miss_supplied"]) > abs(x["e_detected"])]
better = [x for x in pairs if abs(x["e_miss_supplied"]) < abs(x["e_detected"])]
out["miss_alone_per_vine"] = dict(runs=len(pairs), worse=len(worse), better=len(better),
                                  exceptions=better,
                                  exceptions_e_detected_range=[min(x["e_detected"] for x in better),
                                                               max(x["e_detected"] for x in better)])

# ------------------------------------------------------------------ re-entry absences at 15 Hz
traj = n30 = n150 = 0
for v in VIDEOS:
    rec = json.loads((ROOT / f"runs/oracle_cadence_1119/botsort_gmc/{v}_d1_o0.json").read_text())["videos"][0]
    seen = {}
    for i, ids in enumerate(rec["frame_gt_ids"]):
        for g in ids:
            seen.setdefault(g, []).append(i)
    for frames in seen.values():
        gaps = [b - a - 1 for a, b in zip(frames, frames[1:]) if b - a - 1 >= 30]
        traj += bool(gaps)
        n30 += len(gaps)
        n150 += sum(g >= 150 for g in gaps)
out["reentry_absences"] = dict(trajectories_with_absence_ge30=traj, absences_ge30=n30, absences_ge150=n150,
                               absences_shorter_than_150=n30 - n150)

# ------------------------------------------------------------------ the paper's per-vine records and calibration code
with contextlib.redirect_stdout(io.StringIO()):
    NS = runpy.run_path(str(HERE / "tools/calibration_transfer_1120.py"), run_name="calib")
REC, GF, P, CONFIGS, q = NS["REC"], NS["GF"], NS["P"], NS["CONFIGS"], NS["q"]
G_total = sum(GF.values())
out["never_shown"] = {str(d): G_total - sum(len({g for fr in REC[("whole", 0, v, 3072, d)]["frame_gt_ids"] for g in fr})
                                            for v in VIDEOS) for d in (1, 2, 4, 8)}
out["never_shown_of"] = G_total

pv = lambda c, v: P[(c[0], c[1], v, c[2], c[3])]
base = [abs(st.fmean(GF[u] for u in VIDEOS if u != v) / GF[v] - 1) for v in VIDEOS]
bmed, bp90 = st.median(base), q(base, 0.9)
beat = {"ratio": [], "line": []}
for c in CONFIGS:
    rat, lin = [], []
    for v in VIDEOS:
        others = [u for u in VIDEOS if u != v]
        k = sum(GF[u] for u in others) / sum(pv(c, u) for u in others)
        rat.append(abs(k * pv(c, v) / GF[v] - 1))
        xs, ys = [pv(c, u) for u in others], [GF[u] for u in others]
        mx, my = st.fmean(xs), st.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        lin.append(abs((my - b * mx + b * pv(c, v)) / GF[v] - 1))
    for name, xs in (("ratio", rat), ("line", lin)):
        if st.median(xs) < bmed and q(xs, 0.9) < bp90:
            beat[name].append(dict(config="|".join(map(str, c)), median=st.median(xs), p90=q(xs, 0.9)))
out["calibration_configs"] = dict(configurations=len(CONFIGS), baseline_median=bmed, baseline_p90=bp90,
                                  beat_ratio=beat["ratio"], beat_line=beat["line"])

# ------------------------------------------------------------------ independent detectors, vine by vine
VC = json.loads((HERE / "results/vine_calibration_1122.json").read_text())["metrics"]
mean_two = [abs(x) for x in VC["mean_two"]["signed"]]
out["independent_vines"] = {k: sum(abs(x) < m for x, m in zip(VC[k]["signed"], mean_two)) for k in ("ratio_d1", "ratio_d8")}
out["independent_vines_of"] = len(mean_two)

# ------------------------------------------------------------------ added 25 November, second pass
# recall_vs_error: the six detector sets at sigma 3072, pooled frame recall (supplementary_checks_1121.json) against
#   the seven-vine net error with the real detections (error_sources_summary_s3072.json, "none"), per interval; the
#   sets share videos and folds, so the coefficient is descriptive only
# vine_bunches: the vine-level errors of Results 4 in bunches per vine instead of fractions of G
from scipy.stats import spearmanr  # noqa: E402

SETS = [f"{f}_s{sd}" for f in ("crop", "whole") for sd in (0, 1, 2)]
REC6 = json.loads((HERE / "results/supplementary_checks_1121.json").read_text())["recall_s3072"]
ES6 = json.loads((HERE / "results/error_sources_summary_s3072.json").read_text())
rec6 = [REC6[s]["pooled"] for s in SETS]
out["recall_vs_error"] = dict(recall={s: REC6[s]["pooled"] for s in SETS},
                              rho_e={d: round(spearmanr(rec6, [ES6[s]["by_interval"][d]["none"]["e"] for s in SETS]).statistic, 3)
                                     for d in ("1", "2", "4", "8")},
                              rho_D_15Hz=round(spearmanr(rec6, [ES6[s]["by_interval"]["1"]["none"]["D"] for s in SETS]).statistic, 3),
                              rho_M_15Hz=round(spearmanr(rec6, [ES6[s]["by_interval"]["1"]["none"]["M"] for s in SETS]).statistic, 3),
                              e_15Hz={s: ES6[s]["by_interval"]["1"]["none"]["e"] for s in SETS})

near_tot = [c for c in CONFIGS if abs(sum(pv(c, v) for v in VIDEOS) / G_total - 1) <= 0.10]
b_near = [abs(pv(c, v) - GF[v]) for c in near_tot for v in VIDEOS]
b_ratio = []
for c in CONFIGS:
    for v in VIDEOS:
        others = [u for u in VIDEOS if u != v]
        k = sum(GF[u] for u in others) / sum(pv(c, u) for u in others)
        b_ratio.append(abs(k * pv(c, v) - GF[v]))
b_base = [abs(st.fmean(GF[u] for u in VIDEOS if u != v) - GF[v]) for v in VIDEOS]
bs = lambda xs: dict(n=len(xs), median=round(st.median(xs), 1), p90=round(q(xs, 0.9), 1))
out["vine_bunches"] = dict(G=GF, near_total_configs=len(near_tot), near_total=bs(b_near), ratio_other_six=bs(b_ratio),
                           mean_other_six=bs(b_base))

dest = HERE / "results/review_checks_1125.json"
dest.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "miss_alone_per_vine"}, indent=1))
print("miss alone:", {k: out["miss_alone_per_vine"][k] for k in ("runs", "worse", "better", "exceptions_e_detected_range")})
