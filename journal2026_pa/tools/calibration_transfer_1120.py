#!/usr/bin/env python3
"""Can a calibration factor absorb the count error?  (new analysis, 2026-11-20, frozen outputs only)

Yield studies correct image counts with a factor or regression fitted on calibration vines
(Nuske et al. 2014; Laurent et al. 2021; Palacios et al. 2023). Each GrapeMOTS multi-view video
films one vine, and its reference count G is the 'true' count here, so a factor only has to
absorb the pipeline's identity errors, not occlusion. Every test below reads the per-video
records that journal_numbers.py reads (crop: retrain.tar.gz; whole: runs/fullframe_pponly_0922)
and the oracle runs (runs/oracle_cadence_1119); nothing is re-run.

  A  leave-one-vine-out ratio calibration inside one configuration (set, scale, interval)
  B  the same with a straight line G = a + b P fitted on the other six vines
  C  a factor fitted on one training seed, applied to another seed of the same recipe
     (same seven vines, so only the seed changes)
  D  a factor fitted at one interval / scale / training representation, applied to another
  E  the oracle (reference boxes as detections): per-vine ratios, leave-one-vine-out error
  F  vine ranking: Spearman rho between P and G over the seven vines
  G  which identity term carries the between-vine spread (U, D, M per G), and whether the
     vine's oracle error (association alone) predicts its real-pipeline error

    .venv/bin/python cadence_journal_1119/tools/calibration_transfer_1120.py
"""
from __future__ import annotations

import io
import itertools as it
import json
import statistics as st
import sys
import tarfile
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

assert HAVE_SCIPY, "SciPy is required"
RUN = ROOT / "runs/fullframe_pponly_0922"
TAR = ROOT / "grapemots-protocol/cadence2026_0919/raw/retrain.tar.gz"
ORACLE = ROOT / "runs/oracle_cadence_1119"
OUT = ROOT / "cadence_journal_1119/results/calibration_transfer_1120.json"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
FAMS = ("crop", "whole")

# ------------------------------------------------------------------ per-video records (as journal_numbers.py)
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
T = {k: decompose(r, 0.5, 1) for k, r in REC.items()}
assert len(T) == 840
GF = {v: T[("whole", 0, v, 3072, 1)]["G"] for v in VIDEOS}
assert all(T[k]["G"] == GF[k[2]] for k in T if k[4] == 1)
assert all(t["identity_holds"] for t in T.values())
G_TOT = sum(GF.values())
P = {k: t["P"] for k, t in T.items()}
CONFIGS = [(f, sd, s, d) for f in FAMS for sd in SEEDS for s in SIGMAS for d in DELTAS]


def q(xs, p):
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summ(abs_errs):
    return dict(n=len(abs_errs), median=round(st.median(abs_errs), 3), p90=round(q(abs_errs, 0.9), 3),
                max=round(max(abs_errs), 3), share_gt_010=round(sum(x > 0.10 for x in abs_errs) / len(abs_errs), 3),
                share_gt_020=round(sum(x > 0.20 for x in abs_errs) / len(abs_errs), 3),
                share_gt_030=round(sum(x > 0.30 for x in abs_errs) / len(abs_errs), 3))


def pv(c, v):
    f, sd, s, d = c
    return P[(f, sd, v, s, d)]


out = {"G_full": GF, "G_total": G_TOT, "configs": len(CONFIGS)}

# ------------------------------------------------------------------ A, B: leave-one-vine-out inside a configuration
raw, ratio, line = [], [], []
by_delta = {d: {"raw": [], "ratio": []} for d in DELTAS}
for c in CONFIGS:
    for v in VIDEOS:
        others = [u for u in VIDEOS if u != v]
        e_raw = pv(c, v) / GF[v] - 1
        sp = sum(pv(c, u) for u in others)
        k = sum(GF[u] for u in others) / sp
        e_ratio = k * pv(c, v) / GF[v] - 1
        xs, ys = [pv(c, u) for u in others], [GF[u] for u in others]
        mx, my = st.fmean(xs), st.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        a = my - b * mx
        e_line = (a + b * pv(c, v)) / GF[v] - 1
        raw.append(abs(e_raw)); ratio.append(abs(e_ratio)); line.append(abs(e_line))
        by_delta[c[3]]["raw"].append(abs(e_raw)); by_delta[c[3]]["ratio"].append(abs(e_ratio))
out["A_lovo_uncalibrated"] = summ(raw)
out["A_lovo_ratio"] = summ(ratio)
out["B_lovo_line"] = summ(line)
out["A_by_interval"] = {str(d): dict(uncalibrated=summ(x["raw"]), ratio=summ(x["ratio"])) for d, x in by_delta.items()}

# ------------------------------------------------------------------ C, D: a factor carried to another configuration
def k_of(c):
    return G_TOT / sum(pv(c, v) for v in VIDEOS)


def transfer(pairs):
    pooled, per_vine, signs = [], [], 0
    for c0, c1 in pairs:
        k = k_of(c0)
        e = k * sum(pv(c1, v) for v in VIDEOS) / G_TOT - 1
        pooled.append(abs(e))
        per_vine += [abs(k * pv(c1, v) / GF[v] - 1) for v in VIDEOS]
    return dict(pooled=summ(pooled), per_vine=summ(per_vine))


seed_pairs = [((f, a, s, d), (f, b, s, d)) for f in FAMS for s in SIGMAS for d in DELTAS
              for a, b in it.permutations(SEEDS, 2)]
out["C_across_seed"] = transfer(seed_pairs)
out["D_across_interval_1_to_8"] = transfer([((f, sd, s, 1), (f, sd, s, 8)) for f in FAMS for sd in SEEDS for s in SIGMAS])
out["D_across_interval_adjacent"] = transfer([((f, sd, s, a), (f, sd, s, b)) for f in FAMS for sd in SEEDS for s in SIGMAS
                                               for a, b in zip(DELTAS, DELTAS[1:])])
out["D_across_scale_from_3072"] = transfer([((f, sd, 3072, d), (f, sd, s, d)) for f in FAMS for sd in SEEDS for d in DELTAS
                                             for s in SIGMAS if s != 3072])
out["D_across_representation"] = transfer([((a, sd, s, d), (b, sd, s, d)) for sd in SEEDS for s in SIGMAS for d in DELTAS
                                            for a, b in it.permutations(FAMS, 2)])
# reference: the same seven vines, same configuration (pooled error is 0 by construction; per-vine error is the
# between-vine part alone)
out["C_reference_same_config_per_vine"] = summ([abs(k_of(c) * pv(c, v) / GF[v] - 1) for c in CONFIGS for v in VIDEOS])

# ------------------------------------------------------------------ E: oracle
orc = {}
for v in VIDEOS:
    for d in DELTAS:
        r = json.loads((ORACLE / "botsort_gmc" / f"{v}_d{d}_o0.json").read_text())["videos"][0]
        orc[(v, d)] = decompose(r, 0.5, 1)
GO = {v: orc[(v, 1)]["G"] for v in VIDEOS}
assert GO == GF, (GO, GF)
E = {}
for d in DELTAS:
    rat = {v: orc[(v, d)]["P"] / GO[v] for v in VIDEOS}
    lovo = []
    for v in VIDEOS:
        others = [u for u in VIDEOS if u != v]
        k = sum(GO[u] for u in others) / sum(orc[(u, d)]["P"] for u in others)
        lovo.append(abs(k * orc[(v, d)]["P"] / GO[v] - 1))
    E[str(d)] = dict(ratio_min=round(min(rat.values()), 3), ratio_max=round(max(rat.values()), 3),
                     per_vine_ratio={v: round(x, 3) for v, x in rat.items()}, lovo_ratio=summ(lovo))
out["E_oracle_botsort_phase0"] = E

# ------------------------------------------------------------------ F: ranking of vines
rhos = []
for c in CONFIGS:
    rho = spearmanr([pv(c, v) for v in VIDEOS], [GF[v] for v in VIDEOS]).statistic
    rhos.append(rho)
rho_orc = {str(d): round(spearmanr([orc[(v, d)]["P"] for v in VIDEOS], [GO[v] for v in VIDEOS]).statistic, 3)
           for d in DELTAS}
out["F_spearman_P_vs_G"] = dict(median=round(st.median(rhos), 3), q10=round(q(rhos, 0.1), 3), q90=round(q(rhos, 0.9), 3),
                                share_ge_0786=round(sum(r >= 0.786 for r in rhos) / len(rhos), 3),
                                share_le_0=round(sum(r <= 0 for r in rhos) / len(rhos), 3), oracle=rho_orc,
                                note="0.786 = two-sided 5% critical value of Spearman rho for n = 7")

# ------------------------------------------------------------------ G: between-vine spread of the identity terms
spread = {"U": [], "D": [], "M": [], "e": []}
link = []
for c in CONFIGS:
    f, sd, s, d = c
    rows = {v: T[(f, sd, v, s, d)] for v in VIDEOS}
    spread["U"].append(st.pstdev(rows[v]["U"] / GF[v] for v in VIDEOS))
    spread["D"].append(st.pstdev(rows[v]["D"] / GF[v] for v in VIDEOS))
    spread["M"].append(st.pstdev((rows[v]["M"] + GF[v] - rows[v]["G"]) / GF[v] for v in VIDEOS))
    spread["e"].append(st.pstdev(pv(c, v) / GF[v] - 1 for v in VIDEOS))
    link.append(spearmanr([pv(c, v) / GF[v] - 1 for v in VIDEOS],
                          [orc[(v, 1)]["P"] / GO[v] - 1 for v in VIDEOS]).statistic)
out["G_between_vine_sd_median"] = {k: round(st.median(x), 3) for k, x in spread.items()}
out["G_real_error_vs_oracle_error_spearman"] = dict(median=round(st.median(link), 3), q10=round(q(link, 0.1), 3),
                                                    q90=round(q(link, 0.9), 3),
                                                    share_ge_0786=round(sum(r >= 0.786 for r in link) / len(link), 3))

# ------------------------------------------------------------------ H: which term a factor cannot absorb
# Counterfactual counts built from the measured terms of the same arm: a count with only the missed
# trajectories (G - M), only the surplus (G + U + D), and the real count (G + U + D - M).
def lovo_ratio_err(counts):
    errs = []
    for v in VIDEOS:
        others = [u for u in VIDEOS if u != v]
        k = sum(GF[u] for u in others) / sum(counts[u] for u in others)
        errs.append(abs(k * counts[v] / GF[v] - 1))
    return errs


H = {"miss_only": [], "surplus_only": [], "real": [], "dup_only": [], "unowned_only": []}
for c in CONFIGS:
    f, sd, s, d = c
    rows = {v: T[(f, sd, v, s, d)] for v in VIDEOS}
    mfull = {v: rows[v]["M"] + GF[v] - rows[v]["G"] for v in VIDEOS}
    H["miss_only"] += lovo_ratio_err({v: GF[v] - mfull[v] for v in VIDEOS})
    H["surplus_only"] += lovo_ratio_err({v: GF[v] + rows[v]["U"] + rows[v]["D"] for v in VIDEOS})
    H["dup_only"] += lovo_ratio_err({v: GF[v] + rows[v]["D"] for v in VIDEOS})
    H["unowned_only"] += lovo_ratio_err({v: GF[v] + rows[v]["U"] for v in VIDEOS})
    H["real"] += lovo_ratio_err({v: pv(c, v) for v in VIDEOS})
out["H_lovo_ratio_by_term"] = {k: summ(x) for k, x in H.items()}

# per-vine means over all 120 configurations, and the oracle at interval 1
pvm = {}
for v in VIDEOS:
    rs = [T[(f, sd, v, s, d)] for (f, sd, s, d) in CONFIGS]
    pvm[v] = dict(G=GF[v], e_mean=round(st.fmean(r["P"] / GF[v] - 1 for r in rs), 3),
                  U=round(st.fmean(r["U"] / GF[v] for r in rs), 3), D=round(st.fmean(r["D"] / GF[v] for r in rs), 3),
                  M=round(st.fmean((r["M"] + GF[v] - r["G"]) / GF[v] for r in rs), 3),
                  oracle_e_d1=round(orc[(v, 1)]["P"] / GO[v] - 1, 3))
out["per_vine_means"] = pvm

# ------------------------------------------------------------------ I: robustness and the vine-level coupling of terms
import math  # noqa: E402

# I1 across vines, inside a configuration: do surplus and misses move together?
cpl = {"UD_vs_M": [], "D_vs_M": [], "U_vs_M": []}
for c in CONFIGS:
    f, sd, s, d = c
    rows = {v: T[(f, sd, v, s, d)] for v in VIDEOS}
    m = [(rows[v]["M"] + GF[v] - rows[v]["G"]) / GF[v] for v in VIDEOS]
    cpl["UD_vs_M"].append(spearmanr([(rows[v]["U"] + rows[v]["D"]) / GF[v] for v in VIDEOS], m).statistic)
    cpl["D_vs_M"].append(spearmanr([rows[v]["D"] / GF[v] for v in VIDEOS], m).statistic)
    cpl["U_vs_M"].append(spearmanr([rows[v]["U"] / GF[v] for v in VIDEOS], m).statistic)
out["I1_spearman_across_vines"] = {k: dict(median=round(st.median(x), 3), q10=round(q(x, 0.1), 3), q90=round(q(x, 0.9), 3),
                                           share_le_minus_0786=round(sum(r <= -0.786 for r in x) / len(x), 3))
                                   for k, x in cpl.items()}

# I2 variance shares of log(P/G) (symmetric in over- and under-count), balanced design as in journal_numbers.py
rows = [(f, sd, v, s, d, math.log(max(P[(f, sd, v, s, d)], 0.5) / GF[v])) for (f, sd, v, s, d) in T]
mean = st.fmean(r[5] for r in rows)
tot = sum((r[5] - mean) ** 2 for r in rows)


def share(idx, rs=rows, mu=mean, tt=tot):
    g = {}
    for r in rs:
        g.setdefault(tuple(r[i] for i in idx), []).append(r[5])
    return sum(len(x) * (st.fmean(x) - mu) ** 2 for x in g.values()) / tt


out["I2_variance_share_logratio"] = {"vine": round(share((2,)), 3), "interval": round(share((4,)), 3),
                                     "scale": round(share((3,)), 3), "detector_set": round(share((0, 1)), 3),
                                     "vine_x_detector_set": round(share((0, 1, 2)) - share((2,)) - share((0, 1)), 3)}
r6 = [r for r in rows if r[2] != "PathPlanning_6"]
mu6 = st.fmean(r[5] for r in r6)
tt6 = sum((r[5] - mu6) ** 2 for r in r6)
out["I2_variance_share_logratio_without_PP6"] = {
    "vine": round(share((2,), r6, mu6, tt6), 3), "interval": round(share((4,), r6, mu6, tt6), 3),
    "scale": round(share((3,), r6, mu6, tt6), 3), "detector_set": round(share((0, 1), r6, mu6, tt6), 3)}

# I3 leave-one-vine-out ratio calibration without PP6 (the most over-counted vine)
V6 = [v for v in VIDEOS if v != "PathPlanning_6"]
e6 = []
for c in CONFIGS:
    for v in V6:
        others = [u for u in V6 if u != v]
        k = sum(GF[u] for u in others) / sum(pv(c, u) for u in others)
        e6.append(abs(k * pv(c, v) / GF[v] - 1))
out["I3_lovo_ratio_without_PP6"] = summ(e6)

# I4 counterfactual counts: no duplicates (G + U - M) and no surplus at all (G - M); ranking of vines
cf = {"no_duplicates": [], "no_surplus": []}
rk = {"real": [], "no_duplicates": [], "no_surplus": []}
for c in CONFIGS:
    f, sd, s, d = c
    rows_c = {v: T[(f, sd, v, s, d)] for v in VIDEOS}
    mfull = {v: rows_c[v]["M"] + GF[v] - rows_c[v]["G"] for v in VIDEOS}
    nd = {v: GF[v] + rows_c[v]["U"] - mfull[v] for v in VIDEOS}
    ns = {v: GF[v] - mfull[v] for v in VIDEOS}
    cf["no_duplicates"] += lovo_ratio_err(nd)
    cf["no_surplus"] += lovo_ratio_err(ns)
    g = [GF[v] for v in VIDEOS]
    rk["real"].append(spearmanr([pv(c, v) for v in VIDEOS], g).statistic)
    rk["no_duplicates"].append(spearmanr([nd[v] for v in VIDEOS], g).statistic)
    rk["no_surplus"].append(spearmanr([ns[v] for v in VIDEOS], g).statistic)
out["I4_lovo_ratio_counterfactual"] = {k: summ(x) for k, x in cf.items()}
out["I4_spearman_counterfactual"] = {k: dict(median=round(st.median(x), 3), q90=round(q(x, 0.9), 3)) for k, x in rk.items()}

# ------------------------------------------------------------------ J: configurations whose total is right
near = [c for c in CONFIGS if abs(sum(pv(c, v) for v in VIDEOS) / G_TOT - 1) <= 0.10]
jv = [abs(pv(c, v) / GF[v] - 1) for c in near for v in VIDEOS]
out["J_total_within_10pct"] = dict(
    configs=len(near), per_vine_abs_error=summ(jv) if jv else None,
    configs_with_vines_of_both_signs=sum(len({pv(c, v) > GF[v] for v in VIDEOS if pv(c, v) != GF[v]}) > 1 for c in near),
    median_spearman=round(st.median(spearmanr([pv(c, v) for v in VIDEOS], [GF[v] for v in VIDEOS]).statistic
                                    for c in near), 3) if near else None)

OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
