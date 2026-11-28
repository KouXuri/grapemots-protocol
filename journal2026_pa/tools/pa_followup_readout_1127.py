#!/usr/bin/env python3
"""Readout of the pre-submission checks queued on 24 November (ledger 10.28.17; rules written before any run).

Reads the outputs of tools/queue_pa_followup_1124.py, mirrored from raptor into runs/pa_followup_1124/
(lettuce/substitution, r/replay, r/substitution, m/...). L and R finished on 27 November, M (YOLO26m, 21 checkpoints)
on 28 November; the H6 verdicts are written only when all 21 checkpoints are in.

L (LettuceMOTS, real YOLO26s detector on the authors' split; test 0003/0007/0011, the three return passes)
  H7a  pooled D/G at the native 1088 px and 15 Hz with the real detections:
       >= 0.30 -> "the return passes over-count with real detections too"; <= 0.10 -> "the surplus disappears";
       otherwise report as is.
  H7b  supplying the missed boxes alone (fix "miss"), pooled over the three sequences, in all 16 (scale x interval)
       combinations: D rises in every one, and |e| rises in every one whose e was >= 0 -> "holds"; else report.
R (crop checkpoint PathPlanning_4_s2 retrained at batch 8; the published one ran at batch 4 after an OOM)
  |delta e| between the retrained and the published checkpoint over the 20 (scale x interval) cells of PP4:
  median < 0.30 (the median seed-pair difference of one recipe) -> one sentence in the manuscript, numbers
  unchanged; otherwise use the batch-8 checkpoint and recompute every result that uses crop seed 2.
  Pipeline check: at sigma 3072 the published checkpoint's replay (runs/error_sources_1120) against its live
  arms in the archive.
    .venv/bin/python cadence_journal_1119/tools/pa_followup_readout_1127.py
"""
from __future__ import annotations

import io
import json
import statistics as st
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
RUN = ROOT / "runs/pa_followup_1124"
TAR = ROOT / "grapemots-protocol/cadence2026_0919/raw/retrain.tar.gz"
ES = ROOT / "runs/error_sources_1120/substitution/other"
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = ("1", "2", "4", "8")
out = {}

# ------------------------------------------------------------------ L
LV, LS = ("0003", "0007", "0011"), (272, 544, 816, 1088)
L = {(v, s): json.loads((RUN / f"lettuce/substitution/{v}_s{s}.json").read_text()) for v in LV for s in LS}
G = {v: L[(v, 1088)]["G_full"] for v in LV}
assert G == {"0003": 40, "0007": 37, "0011": 37}, G
GT = sum(G.values())


def pooled(s, d, fix):
    vals = [L[(v, s)]["arms"][d]["values"][fix] for v in LV]
    P = sum(x["P"] for x in vals)
    return dict(P=P, U=sum(x["U"] for x in vals), D=sum(x["D"] for x in vals), M=sum(x["M"] for x in vals),
                e=(P - GT) / GT)


rows = {}
for s in LS:
    for d in DELTAS:
        a, b = pooled(s, d, "none"), pooled(s, d, "miss")
        rows[f"{s}|{d}"] = dict(none=a, miss=b, D_up=b["D"] > a["D"],
                                abs_e_up=(abs(b["e"]) > abs(a["e"])) if a["e"] >= 0 else None)
h7a = pooled(1088, "1", "none")["D"] / GT
up = [r for r in rows.values() if r["abs_e_up"] is not None]
out["L"] = dict(
    G=G, recall_iou50={f"{v}|{s}": L[(v, s)]["detection_stats"]["recall_iou50"] for v in LV for s in LS},
    H7a=dict(D_per_G=round(h7a, 3), per_video={v: L[(v, 1088)]["arms"]["1"]["values"]["none"]["D"] / G[v] for v in LV},
             e_real=round(pooled(1088, "1", "none")["e"], 3), e_reference_boxes=round(pooled(1088, "1", "bkg+jit+loc+miss+score")["e"], 3),
             verdict="over-counts with real detections too" if h7a >= 0.30 else
             ("surplus disappears" if h7a <= 0.10 else "in between: report as is")),
    H7b=dict(D_up=sum(r["D_up"] for r in rows.values()), combos=len(rows),
             abs_e_up=sum(r["abs_e_up"] for r in up), combos_e_nonneg=len(up),
             verdict="holds" if all(r["D_up"] for r in rows.values()) and all(r["abs_e_up"] for r in up) else "report"),
    rows=rows)

# ------------------------------------------------------------------ R
pub = {}
with tarfile.open(TAR) as tar:
    for s in SIGMAS:
        for d in DELTAS:
            m = tar.getmember(f"retrain/PathPlanning_4_s2__PathPlanning_4_s{s}_d{d}.json")
            pub[(s, d)] = json.load(io.TextIOWrapper(tar.extractfile(m)))["videos"][0]
GP = len({g for fr in pub[(3072, "1")]["frame_gt_ids"] for g in fr})
assert GP == 93, GP
e_pub = {k: (len({t for fr in r["frame_predicted_ids"] for t in fr}) - GP) / GP for k, r in pub.items()}
rep = {s: json.loads((RUN / f"r/replay/r_s2__PathPlanning_4_s{s}.json").read_text()) for s in SIGMAS}
assert all(r["G_full"] == GP for r in rep.values())
e_r = {(s, d): rep[s]["arms"][d]["none"]["e"] for s in SIGMAS for d in DELTAS}
diff = {f"{s}|{d}": round(e_r[(s, d)] - e_pub[(s, d)], 4) for s in SIGMAS for d in DELTAS}
med = st.median(abs(x) for x in diff.values())
es = json.loads((ES / "crop_s2__PathPlanning_4_s3072.json").read_text())
check = {d: dict(replay=round(es["arms"][d]["values"]["none"]["e"], 4), live=round(e_pub[(3072, d)], 4)) for d in DELTAS}
trained = json.loads((ROOT / "runs/detect/rev_lovo_0920_b8check_1124/PathPlanning_4_s2/trained.json").read_text()) \
    if (ROOT / "runs/detect/rev_lovo_0920_b8check_1124/PathPlanning_4_s2/trained.json").exists() else None
out["R"] = dict(e_published={f"{s}|{d}": round(e_pub[(s, d)], 4) for s in SIGMAS for d in DELTAS},
                e_batch8={f"{s}|{d}": round(e_r[(s, d)], 4) for s in SIGMAS for d in DELTAS}, delta=diff,
                median_abs_delta=round(med, 4), max_abs_delta=round(max(abs(x) for x in diff.values()), 4),
                sign_changes=sum((e_r[k] >= 0) != (e_pub[k] >= 0) for k in e_r),
                threshold=0.30, verdict="within retraining noise: one sentence, numbers unchanged" if med < 0.30
                else "replace the checkpoint and recompute crop seed 2",
                pipeline_check_3072=check, trained=trained)

# ------------------------------------------------------------------ M (YOLO26m crop detectors, H6)
# Rules (ledger 10.28.17, written before any run):
#   H6a  frame recall at sigma 3072, YOLO26m against the crop YOLO26s of the same seed
#   H6b  Spearman rho over the nine detector sets between recall and the seven-vine net error at 15 Hz:
#        >= 0.700 (two-sided 5% for n = 9; the sets share videos, so descriptive) -> "holds across detectors"
#   H6c  supplying the missed boxes alone at sigma 3072: |e| larger in all 12 (seed x interval) combinations ->
#        "holds on the larger model"; otherwise report
#   H6d  leave-one-vine-out ratio calibration over the 60 configurations (3 seeds x 5 scales x 4 intervals): if the
#        median AND the 90th percentile both beat the no-video baseline (0.230 / 0.458), the abstract and
#        conclusions must restrict "calibration does not beat the baseline" to YOLO26s; otherwise "unchanged"
#   also: at 15 Hz and sigma 3072, the median paired difference (same seed, same fold) of e and of recall per vine
# Verdicts are written only when all 21 checkpoints are in; before that the complete seeds are summarised.
import sys  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_batch8_1127  # noqa: E402

VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
MS = (0, 1, 2)


def q(xs, p):                                         # calibration_transfer_1120.q
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def m_ready(sd, v):
    return all((RUN / f"m/replay/m_s{sd}__{v}_s{s}.json").exists() for s in SIGMAS) and \
        (RUN / f"m/substitution/m_s{sd}__{v}_s3072.json").exists()


ready = {(sd, v): m_ready(sd, v) for sd in MS for v in VIDEOS}
seeds = [sd for sd in MS if all(ready[(sd, v)] for v in VIDEOS)]
complete = len(seeds) == len(MS)
batch = {}
for sd in MS:
    for v in VIDEOS:
        tj = ROOT / f"runs/detect/m_lovo_1124/{v}_s{sd}/trained.json"
        if tj.exists():
            t = json.loads(tj.read_text())
            batch[f"{v}_s{sd}"] = t["batch_used"]
MREP = {(sd, v, s): json.loads((RUN / f"m/replay/m_s{sd}__{v}_s{s}.json").read_text())
        for sd in seeds for v in VIDEOS for s in SIGMAS}
MSUB = {(sd, v): json.loads((RUN / f"m/substitution/m_s{sd}__{v}_s3072.json").read_text()) for sd in seeds for v in VIDEOS}
GF = {v: 0 for v in VIDEOS}
for (sd, v, s), r in MREP.items():
    GF[v] = r["G_full"]
if seeds:
    assert GF == {"PathPlanning_2": 45, "PathPlanning_3": 57, "PathPlanning_4": 93, "PathPlanning_5": 84,
                  "PathPlanning_6": 51, "PathPlanning_7": 65, "PathPlanning_8": 77}, GF
    for (sd, v), r in MSUB.items():                   # the substitution's "none" is the replay's "none"
        assert all(r["arms"][d]["values"]["none"]["P"] == MREP[(sd, v, 3072)]["arms"][d]["none"]["P"] for d in DELTAS)
GT7 = sum(GF.values())
CSUB = {(sd, v): crop_batch8_1127.read_substitution(ROOT / f"runs/error_sources_1120/substitution/other/crop_s{sd}__{v}_s3072.json")
        for sd in MS for v in VIDEOS}
SX = json.loads((HERE / "results/supplementary_checks_1121.json").read_text())["recall_s3072"]
ES6 = json.loads((HERE / "results/error_sources_summary_s3072.json").read_text())
RC = json.loads((HERE / "results/review_checks_1125.json").read_text())["calibration_configs"]
BMED, BP90 = RC["baseline_median"], RC["baseline_p90"]


def pooled_e(sub, sd, d, fix):
    P = sum(sub[(sd, v)]["arms"][d]["values"][fix]["P"] for v in VIDEOS)
    return (P - GT7) / GT7


def recall(sub, sd):
    st_ = [sub[(sd, v)]["detection_stats"] for v in VIDEOS]
    return sum(x["tp"] for x in st_) / sum(x["reference_boxes"] for x in st_)


M = dict(complete=complete, seeds_complete=seeds, checkpoints_ready=sum(ready.values()), batch_used=batch)
if seeds:
    M["H6a"] = {f"s{sd}": dict(recall_m=round(recall(MSUB, sd), 4), recall_crop_s=SX[f"crop_s{sd}"]["pooled"],
                               diff=round(recall(MSUB, sd) - SX[f"crop_s{sd}"]["pooled"], 4)) for sd in seeds}
    sets = {f"{f}_s{sd}": (SX[f"{f}_s{sd}"]["pooled"], ES6[f"{f}_s{sd}"]["by_interval"]["1"]["none"]["e"])
            for f in ("crop", "whole") for sd in MS}
    sets.update({f"m_s{sd}": (recall(MSUB, sd), pooled_e(MSUB, sd, "1", "none")) for sd in seeds})
    rho = spearmanr([x[0] for x in sets.values()], [x[1] for x in sets.values()]).statistic
    M["H6b"] = dict(sets={k: dict(recall=round(a, 4), e_15Hz=round(b, 4)) for k, (a, b) in sets.items()}, n=len(sets),
                    rho=round(float(rho), 3))
    c = {f"s{sd}|{d}": dict(e_none=round(pooled_e(MSUB, sd, d, "none"), 4), e_miss=round(pooled_e(MSUB, sd, d, "miss"), 4))
         for sd in seeds for d in DELTAS}
    for x in c.values():
        x["worse"] = abs(x["e_miss"]) > abs(x["e_none"])
    M["H6c"] = dict(combos=c, worse=sum(x["worse"] for x in c.values()), n=len(c))
    errs = []
    for sd in seeds:
        for s in SIGMAS:
            for d in DELTAS:
                P = {v: MREP[(sd, v, s)]["arms"][d]["none"]["P"] for v in VIDEOS}
                for v in VIDEOS:
                    o = [u for u in VIDEOS if u != v]
                    k = sum(GF[u] for u in o) / sum(P[u] for u in o)
                    errs.append(abs(k * P[v] / GF[v] - 1))
    M["H6d"] = dict(n=len(errs), median=round(st.median(errs), 4), p90=round(q(errs, 0.9), 4),
                    baseline_median=BMED, baseline_p90=BP90)
    pe = [MSUB[(sd, v)]["arms"]["1"]["values"]["none"]["e"] - CSUB[(sd, v)]["arms"]["1"]["values"]["none"]["e"]
          for sd in seeds for v in VIDEOS]
    pr = [MSUB[(sd, v)]["detection_stats"]["recall_iou50"] - CSUB[(sd, v)]["detection_stats"]["recall_iou50"]
          for sd in seeds for v in VIDEOS]
    M["paired_15Hz_3072"] = dict(n=len(pe), median_diff_e=round(st.median(pe), 4), median_diff_recall=round(st.median(pr), 4))
    if complete:
        M["H6b"]["verdict"] = "holds across detectors" if rho >= 0.700 else "report and rewrite"
        M["H6c"]["verdict"] = "holds on the larger model" if M["H6c"]["worse"] == 12 else "report"
        M["H6d"]["verdict"] = ("restrict the claim to YOLO26s" if M["H6d"]["median"] < BMED and M["H6d"]["p90"] < BP90
                               else "unchanged with a larger detector")

    # -------------------------------------------------------------- descriptive additions (2026-11-28, ledger
    # 10.28.17 "读数三"). Not part of the rules above; they place the YOLO26m numbers next to those the article
    # already prints for the six YOLO26s sets (review_checks_1125.py: recall_vs_error; calibration_transfer_1120.py: A).
    RV = json.loads((HERE / "results/review_checks_1125.json").read_text())["recall_vs_error"]
    rec9 = dict(RV["recall"]) | {f"m_s{sd}": recall(MSUB, sd) for sd in seeds}

    def m_terms(sd, d):
        t = {x: sum(MSUB[(sd, v)]["arms"][d]["values"]["none"][x] for v in VIDEOS) / GT7 for x in ("U", "D", "M")}
        t["e"] = pooled_e(MSUB, sd, d, "none")
        return t

    T9 = {k: {d: ES6[k]["by_interval"][d]["none"] for d in DELTAS} for k in RV["recall"]}
    T9.update({f"m_s{sd}": {d: m_terms(sd, d) for d in DELTAS} for sd in seeds})
    ks = list(rec9)
    sp = lambda ys: round(float(spearmanr([rec9[k] for k in ks], ys).statistic), 3)  # noqa: E731
    vine_recall = [MSUB[(sd, v)]["detection_stats"]["recall_iou50"] for sd in seeds for v in VIDEOS]
    seed_pairs = [abs(CSUB[(a, v)]["arms"]["1"]["values"]["none"]["e"] - CSUB[(b, v)]["arms"]["1"]["values"]["none"]["e"])
                  for a, b in ((0, 1), (0, 2), (1, 2)) for v in VIDEOS]
    import contextlib  # noqa: E402
    import runpy  # noqa: E402
    with contextlib.redirect_stdout(io.StringIO()):
        CTN = runpy.run_path(str(HERE / "tools/calibration_transfer_1120.py"), run_name="calib")
    lovo_s = {}
    for fam in ("crop", "whole"):
        xs = []
        for c in CTN["CONFIGS"]:
            if c[0] != fam:
                continue
            pv = {v: CTN["P"][(c[0], c[1], v, c[2], c[3])] for v in VIDEOS}
            for v in VIDEOS:
                o = [u for u in VIDEOS if u != v]
                k = sum(CTN["GF"][u] for u in o) / sum(pv[u] for u in o)
                xs.append(abs(k * pv[v] / CTN["GF"][v] - 1))
        lovo_s[fam] = dict(n=len(xs), median=round(st.median(xs), 4), p90=round(q(xs, 0.9), 4))
    M["descriptive"] = dict(
        note="not in the pre-registered rules; descriptive only, the nine sets share videos",
        rho_e_nine={d: sp([T9[k][d]["e"] for k in ks]) for d in DELTAS},
        rho_D_15Hz_nine=sp([T9[k]["1"]["D"] for k in ks]), rho_M_15Hz_nine=sp([T9[k]["1"]["M"] for k in ks]),
        m_terms_15Hz={f"m_s{sd}": {x: round(y, 4) for x, y in T9[f"m_s{sd}"]["1"].items()} for sd in seeds},
        m_recall_per_vine=dict(min=round(min(vine_recall), 4), max=round(max(vine_recall), 4)),
        paired_15Hz_3072=dict(higher_e=sum(x > 0 for x in pe), lower_e=sum(x < 0 for x in pe), equal_e=sum(x == 0 for x in pe),
                              higher_recall=sum(x > 0 for x in pr), median_abs_diff_e=round(st.median(abs(x) for x in pe), 4),
                              recall_diff_min=round(min(pr), 4), recall_diff_max=round(max(pr), 4)),
        crop_s_seed_pairs_15Hz_3072=dict(n=len(seed_pairs), median_abs_diff_e=round(st.median(seed_pairs), 4)),
        lovo_ratio_yolo26s=lovo_s)
out["M"] = M

dest = HERE / "results/pa_followup_readout_1127.json"
dest.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"L": {k: out["L"][k] for k in ("H7a", "H7b")},
                  "R": {k: out["R"][k] for k in ("median_abs_delta", "max_abs_delta", "sign_changes", "verdict",
                                                  "pipeline_check_3072")},
                  "M": {k: v for k, v in M.items() if k not in ("batch_used",)} | {"batch_values": sorted(set(batch.values()))}},
                 indent=1))
