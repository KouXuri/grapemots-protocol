#!/usr/bin/env python3
"""CPU checks behind the 2026-11-20 storyline (gross vs net error, label-free spread, A/A seed test).

Reads the same frozen per-frame outputs and the same decomposition (decompose_count_error.py,
IoU 0.5, SciPy Hungarian) as journal_numbers.py; nothing is re-implemented.
    .venv/bin/python cadence_journal_1119/tools/story_checks_1120.py
"""
from __future__ import annotations
import io, json, math, statistics as st, sys, tarfile
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402
assert HAVE_SCIPY
RUN = ROOT / "runs/fullframe_pponly_0922"
TAR = ROOT / "grapemots-protocol/cadence2026_0919/raw/retrain.tar.gz"
ORACLE = ROOT / "runs/oracle_cadence_1119"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
ALL11 = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]
SIGMAS, DELTAS, SEEDS, FAMS = (1536, 2048, 2560, 3072, 3840), (1, 2, 4, 8), (0, 1, 2), ("crop", "whole")

REC = {}
with tarfile.open(TAR) as tar:
    for m in tar.getmembers():
        if not m.isfile():
            continue
        fold, arm = Path(m.name).name[:-5].split("__")
        v, seed = fold.rsplit("_s", 1)
        if v in VIDEOS:
            s, d = arm.rsplit("_s", 1)[1].split("_d")
            REC[("crop", int(seed), v, int(s), int(d))] = json.load(io.TextIOWrapper(tar.extractfile(m)))["videos"][0]
for sd in SEEDS:
    for v in VIDEOS:
        for s in SIGMAS:
            for d in DELTAS:
                REC[("whole", sd, v, s, d)] = json.loads((RUN / f"surface_seed{sd}" / f"{v}_s{s}_d{d}.json").read_text())["videos"][0]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_batch8_1127  # noqa: E402  batch-8 retrain of crop PathPlanning_4_s2 (ledger 10.28.17, R rule)
crop_batch8_1127.apply(REC)
T = {k: decompose(r, 0.5, 1) for k, r in REC.items()}
GF = {v: T[("whole", 0, v, 3072, 1)]["G"] for v in VIDEOS}


def terms(t, v):
    G = GF[v]
    Mf = t["M"] + G - t["G"]           # unreached incl. never shown
    return dict(e=(t["P"] - G) / G, U=t["U"] / G, D=t["D"] / G, M=Mf / G, P=t["P"],
                gross=(t["U"] + t["D"] + Mf) / G, cov=1 - Mf / G)


A = {k: terms(t, k[2]) for k, t in T.items()}
out = {}
# 1. gross vs net
for f in FAMS:
    for d in DELTAS:
        xs = [A[(f, sd, v, s, d)] for sd in SEEDS for v in VIDEOS for s in SIGMAS]
        out[f"gross|{f}|d{d}"] = dict(median_gross=round(st.median(x["gross"] for x in xs), 3),
                                      median_abs_e=round(st.median(abs(x["e"]) for x in xs), 3))
near = [x for x in A.values() if abs(x["e"]) <= 0.10]
out["near_zero_arms"] = dict(n=len(near), gross_min=round(min(x["gross"] for x in near), 3),
                             gross_min_exact=min(x["gross"] for x in near),
                             gross_median=round(st.median(x["gross"] for x in near), 3))
# 2. seed spread (label-free CV of P) vs gross and vs |e|
cells = [(f, v, s, d) for f in FAMS for v in VIDEOS for s in SIGMAS for d in DELTAS]
cvP, spread_e, mgross, mabs = [], [], [], []
for f, v, s, d in cells:
    Ps = [A[(f, sd, v, s, d)]["P"] for sd in SEEDS]
    es = [A[(f, sd, v, s, d)]["e"] for sd in SEEDS]
    cvP.append(st.pstdev(Ps) / st.fmean(Ps) if st.fmean(Ps) > 0 else float("nan"))
    spread_e.append(max(es) - min(es))
    mgross.append(st.fmean(A[(f, sd, v, s, d)]["gross"] for sd in SEEDS))
    mabs.append(abs(st.fmean(es)))
ok = [i for i, c in enumerate(cvP) if not math.isnan(c)]
out["seed_spread"] = dict(
    cells=len(cells),
    spearman_spread_e_vs_mean_gross=round(spearmanr([spread_e[i] for i in ok], [mgross[i] for i in ok])[0], 3),
    spearman_spread_e_vs_abs_mean_e=round(spearmanr([spread_e[i] for i in ok], [mabs[i] for i in ok])[0], 3),
    spearman_cvP_vs_abs_mean_e=round(spearmanr([cvP[i] for i in ok], [mabs[i] for i in ok])[0], 3),
    spearman_cvP_vs_cov=round(spearmanr([cvP[i] for i in ok], [st.fmean(A[(c[0], sd, *c[1:])]["cov"] for sd in SEEDS) for c in [cells[i] for i in ok]])[0], 3),
    median_cvP=round(st.median(cvP[i] for i in ok), 3))
# 3. A/A: two checkpoints of one recipe, same cell
aa = []
for f, v, s, d in cells:
    es = [A[(f, sd, v, s, d)]["e"] for sd in SEEDS]
    aa += [abs(es[i] - es[j]) for i in range(3) for j in range(i + 1, 3)]
aa.sort()
q = lambda p: round(aa[int(p * (len(aa) - 1))], 3)
out["aa_seed_pairs"] = dict(n=len(aa), median=q(0.5), p75=q(0.75), p90=q(0.9), max=round(aa[-1], 3),
                           share_ge_0p62=round(sum(x >= 0.62 for x in aa) / len(aa), 4),
                           share_ge_1p08=round(sum(x >= 1.08 for x in aa) / len(aa), 4))
# per-video mean over a 'typical reporting' configuration: sigma 3072, delta 1 and delta 2
for d in (1, 2, 4, 8):
    row = {}
    for f in FAMS:
        diffs = []
        for v in VIDEOS:
            es = [A[(f, sd, v, 3072, d)]["e"] for sd in SEEDS]
            diffs.append(round(max(es) - min(es), 3))
        row[f] = diffs
    out[f"aa_sigma3072_d{d}_per_video_range"] = row
# 4. label-free interval sensitivity vs surplus at delta 1
sens, surplus = [], []
for f in FAMS:
    for sd in SEEDS:
        for v in VIDEOS:
            for s in SIGMAS:
                p1, p2 = A[(f, sd, v, s, 1)]["P"], A[(f, sd, v, s, 2)]["P"]
                if p1 > 0 and p2 > 0:
                    sens.append(math.log(p1 / p2))
                    surplus.append(A[(f, sd, v, s, 1)]["U"] + A[(f, sd, v, s, 1)]["D"])
out["interval_sensitivity"] = dict(n=len(sens), spearman_log_P1_over_P2_vs_surplus_d1=round(spearmanr(sens, surplus)[0], 3))
# 5. oracle vs real: terms by interval (BoT-SORT GMC, phase-mean), multi-view PP2-8
orc = {}
for v in ALL11:
    for d in DELTAS:
        for k in range(d):
            r = json.loads((ORACLE / "botsort_gmc" / f"{v}_d{d}_o{k}.json").read_text())["videos"][0]
            orc[(v, d, k)] = decompose(r, 0.5, 1)
GO = {v: orc[(v, 1, 0)]["G"] for v in ALL11}
ot = {}
for grp, vids in (("PP2-8", VIDEOS), ("NP1-3", ALL11[:3]), ("PP1", ["PathPlanning_1"])):
    for d in DELTAS:
        G = sum(GO[v] for v in vids)
        D = sum(st.fmean(orc[(v, d, k)]["D"] for k in range(d)) for v in vids) / G
        M = sum(st.fmean(orc[(v, d, k)]["M"] + GO[v] - orc[(v, d, k)]["G"] for k in range(d)) for v in vids) / G
        U = sum(st.fmean(orc[(v, d, k)]["U"] for k in range(d)) for v in vids) / G
        ot[f"{grp}|d{d}"] = dict(U=round(U, 3), D=round(D, 3), M=round(M, 3), e=round(U + D - M, 3))
out["oracle_terms_botsort_gmc"] = ot
for d in DELTAS:
    xs = [A[("whole", sd, v, s, d)] for sd in SEEDS for v in VIDEOS for s in SIGMAS]
    G = len(SEEDS) * len(SIGMAS) * sum(GF.values())
    out[f"real_whole_pooled|d{d}"] = {k: round(sum(x[k] * GF[v] for x, (sd, v, s) in zip(xs, [(sd, v, s) for sd in SEEDS for v in VIDEOS for s in SIGMAS])) / G, 3) for k in ("U", "D", "M", "e")}
print(json.dumps(out, indent=1))
(ROOT / "cadence_journal_1119/results/story_checks_1120.json").write_text(json.dumps(out, indent=1))
