#!/usr/bin/env python3
"""Every number the journal manuscript quotes, from frozen per-frame outputs, in one file.

Families (all on PathPlanning_2..8, plant-disjoint folds, seeds 0-2, BoT-SORT unless named):
  crop   archived retrained detectors (1280 crops), grapemots-protocol cadence2026_0919/raw/retrain.tar.gz
  whole  whole-frame detectors (imgsz 3072), runs/fullframe_pponly_0922/surface_seed{0,1,2}
Controls (whole, seed 0): phase / ret / cu / rt / bytetrack from runs/fullframe_pponly_0922.
Optional inputs, used when present: oracle (runs/oracle_cadence_1119) and StrongSORT
(runs/wholeframe_strongsort_1119/strongsort), identity metrics (results/identity_metrics.json).

Conventions: G_full over all annotated frames; M_full adds trajectories never shown;
e = (P - G_full)/G_full; coverage = 1 - M_full/G_full; brackets without interpolation;
matched design scores the interval-1 arm at the interval-k arm's frames.

    .venv/bin/python cadence_journal_1119/tools/journal_numbers.py [--check]
"""
from __future__ import annotations

import io
import json
import math
import statistics as st
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

assert HAVE_SCIPY, "SciPy is required"
HERE = ROOT / "cadence_journal_1119"
RUN = ROOT / "runs/fullframe_pponly_0922"
TAR = ROOT / "grapemots-protocol/cadence2026_0919/raw/retrain.tar.gz"
ORACLE = ROOT / "runs/oracle_cadence_1119"
STRONG = ROOT / "runs/wholeframe_strongsort_1119/strongsort"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
ALL11 = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
FAMS = ("crop", "whole")
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
PER_FRAME = ("frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes")


def sign_test(k: int, n: int) -> float:
    """Exact two-sided binomial sign test, ties already removed."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


# ------------------------------------------------------------------ load per-video records
REC = {}                                   # (fam, seed, v, s, d) -> per-video record
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
assert len(T) == 2 * 3 * 7 * 5 * 4
GF = {v: T[("whole", 0, v, 3072, 1)]["G"] for v in VIDEOS}
assert all(T[k]["G"] == GF[k[2]] for k in T if k[4] == 1)
assert all(t["identity_holds"] for t in T.values())


def err(t, v):
    return (t["P"] - GF[v]) / GF[v]


def cov(t, v):
    return 1 - (t["M"] + GF[v] - t["G"]) / GF[v]


def bracket(es):
    if es[0] < 0:
        return "<1"
    for i in range(3):
        if es[i] >= 0 > es[i + 1]:
            return f"({DELTAS[i]},{DELTAS[i + 1]})"
    return ">8"


N = {"G_full": GF, "G_total": sum(GF.values())}

# ------------------------------------------------------------------ interval ordering
steps = {}
for f in FAMS:
    for sd in SEEDS:
        ss = [(err(T[(f, sd, v, s, a)], v), err(T[(f, sd, v, s, b)], v))
              for v in VIDEOS for s in SIGMAS for a, b in zip(DELTAS, DELTAS[1:])]
        steps[(f, sd)] = dict(n=len(ss), down=sum(y < x for x, y in ss), tie=sum(y == x for x, y in ss),
                              up=sum(y > x for x, y in ss))
N["interval_steps"] = {f"{f}|{sd}": x for (f, sd), x in steps.items()}
N["interval_steps_total"] = {k: sum(x[k] for x in steps.values()) for k in ("n", "down", "tie", "up")}
for f in FAMS:
    N[f"interval_steps_{f}"] = {k: sum(steps[(f, sd)][k] for sd in SEEDS) for k in ("n", "down", "tie", "up")}
# sequence level: median over scales of e(1) - e(8), per detector set
seq = {}
for f in FAMS:
    for sd in SEEDS:
        drops = {v: st.median(err(T[(f, sd, v, s, 1)], v) - err(T[(f, sd, v, s, 8)], v) for s in SIGMAS)
                 for v in VIDEOS}
        k = sum(x > 0 for x in drops.values())
        seq[f"{f}|{sd}"] = dict(sequences_dropping=k, sign_test_p=round(sign_test(k, 7), 4),
                                median_drop=round(st.median(drops.values()), 3))
N["sequence_level"] = seq
# scale
sig = {}
for f in FAMS:
    for sd in SEEDS:
        z = [err(T[(f, sd, v, b, d)], v) > err(T[(f, sd, v, a, d)], v)
             for v in VIDEOS for d in DELTAS for a, b in zip(SIGMAS, SIGMAS[1:])]
        sig[f"{f}|{sd}"] = f"{sum(z)}/{len(z)}"
N["scale_steps_raising"] = sig
N["scale_steps_raising_total"] = f"{sum(int(x.split('/')[0]) for x in sig.values())}/{sum(int(x.split('/')[1]) for x in sig.values())}"

# ------------------------------------------------------------------ sign changes and their instability
maps = {(f, sd): {(v, s): bracket([err(T[(f, sd, v, s, d)], v) for d in DELTAS]) for v in VIDEOS for s in SIGMAS}
        for f in FAMS for sd in SEEDS}
N["sign_change"] = {f"{f}|{sd}": dict(rows=sum(b.startswith("(") for b in m.values()),
                                       sequences=len({v for (v, s), b in m.items() if b.startswith("(")}),
                                       always_over=sum(b == ">8" for b in m.values()),
                                       always_under=sum(b == "<1" for b in m.values()),
                                       ordered_sequences=sum(all(ORDER[m[(v, a)]] <= ORDER[m[(v, b)]]
                                                                 for a, b in zip(SIGMAS, SIGMAS[1:])) for v in VIDEOS))
                    for (f, sd), m in maps.items()}
N["crossing_map"] = {f"{f}|{sd}|{v}|{s}": b for (f, sd), m in maps.items() for (v, s), b in m.items()}
cells = [(v, s, d) for v in VIDEOS for s in SIGMAS for d in DELTAS]
for f in FAMS:
    flips = sum(len({err(T[(f, sd, *c)], c[0]) < 0 for sd in SEEDS}) > 1 for c in cells)
    spread = [max(err(T[(f, sd, *c)], c[0]) for sd in SEEDS) - min(err(T[(f, sd, *c)], c[0]) for sd in SEEDS) for c in cells]
    same_row = sum(len({maps[(f, sd)][(v, s)] for sd in SEEDS}) == 1 for v in VIDEOS for s in SIGMAS)
    N[f"seed_effect_{f}"] = dict(cells=len(cells), sign_flips=flips, spread_median=round(st.median(spread), 3),
                                 spread_max=round(max(spread), 3), rows_same_bracket=same_row)
pairs = [(sd, *c) for sd in SEEDS for c in cells]
N["family_effect"] = dict(
    cells=len(pairs),
    sign_flips=sum((err(T[("crop", *p)], p[1]) < 0) != (err(T[("whole", *p)], p[1]) < 0) for p in pairs),
    abs_diff_median=round(st.median(abs(err(T[("crop", *p)], p[1]) - err(T[("whole", *p)], p[1])) for p in pairs), 3),
    rows_same_bracket=sum(maps[("crop", sd)][(v, s)] == maps[("whole", sd)][(v, s)]
                          for sd in SEEDS for v in VIDEOS for s in SIGMAS))
any_flip = sum(len({err(T[(f, sd, *c)], c[0]) < 0 for f in FAMS for sd in SEEDS}) > 1 for c in cells)
N["any_detector_set_flips"] = f"{any_flip}/{len(cells)}"

# variance attribution (balanced factorial: main effects orthogonal)
rows = [(f, sd, v, s, d, err(T[(f, sd, v, s, d)], v)) for (f, sd, v, s, d) in T]
mean = st.fmean(r[5] for r in rows)
tot = sum((r[5] - mean) ** 2 for r in rows)


def share(idx):
    g = {}
    for r in rows:
        g.setdefault(tuple(r[i] for i in idx), []).append(r[5])
    return sum(len(x) * (st.fmean(x) - mean) ** 2 for x in g.values()) / tot


va = {"sequence": share((2,)), "interval": share((3 + 1,)), "scale": share((3,)),
      "detector_set": share((0, 1)), "family": share((0,))}
va["sequence_x_detector_set"] = share((0, 1, 2)) - va["sequence"] - va["detector_set"]
va["sequence_x_scale"] = share((2, 3)) - va["sequence"] - va["scale"]
va["sequence_x_interval"] = share((2, 4)) - va["sequence"] - va["interval"]
va["scale_x_interval"] = share((3, 4)) - va["scale"] - va["interval"]
va["detector_set_x_scale"] = share((0, 1, 3)) - va["detector_set"] - va["scale"]
va["detector_set_x_interval"] = share((0, 1, 4)) - va["detector_set"] - va["interval"]
va["main_effects_and_two_factor_interactions"] = sum(
    va[k] for k in ("sequence", "interval", "scale", "detector_set", "sequence_x_detector_set", "sequence_x_scale",
                    "sequence_x_interval", "scale_x_interval", "detector_set_x_scale", "detector_set_x_interval"))
N["variance_share"] = {k: round(x, 3) for k, x in va.items()}
fit = {}
for p in RUN.glob("PathPlanning_*_s?_verified.json"):
    v, sd = p.name.replace("_verified.json", "").rsplit("_s", 1)
    fit.setdefault(v, []).append(json.loads(p.read_text())["budget"]["best_ap"])
N["validation_map50_95_whole"] = dict(min=min(min(x) for x in fit.values()), max=max(max(x) for x in fit.values()),
                                      max_within_fold_range=round(max(max(x) - min(x) for x in fit.values()), 4))

# ------------------------------------------------------------------ decomposition and cancellation
dec = {}
for f in FAMS:
    for sd in SEEDS:
        g = lambda v, s, d: T[(f, sd, v, s, d)]
        dec[f"{f}|{sd}"] = dict(
            U_falls=sum(g(v, s, 8)["U"] < g(v, s, 1)["U"] for v in VIDEOS for s in SIGMAS),
            D_falls=sum(g(v, s, 8)["D"] < g(v, s, 1)["D"] for v in VIDEOS for s in SIGMAS),
            M_rises=sum(g(v, s, 8)["M"] + GF[v] - g(v, s, 8)["G"] > g(v, s, 1)["M"] for v in VIDEOS for s in SIGMAS),
            surplus_drop_exceeds_coverage_loss=sum(
                (g(v, s, 1)["U"] + g(v, s, 1)["D"]) - (g(v, s, 8)["U"] + g(v, s, 8)["D"])
                > (g(v, s, 8)["M"] + GF[v] - g(v, s, 8)["G"]) - g(v, s, 1)["M"] for v in VIDEOS for s in SIGMAS))
N["decomposition_1to8"] = dec
N["decomposition_1to8_total"] = {k: sum(x[k] for x in dec.values()) for k in next(iter(dec.values()))}
pooled = {}
for f in FAMS:
    for d in DELTAS:
        tt = Counter()
        for sd in SEEDS:
            for v in VIDEOS:
                for s in SIGMAS:
                    t = T[(f, sd, v, s, d)]
                    tt["P"] += t["P"]; tt["U"] += t["U"]; tt["D"] += t["D"]; tt["M"] += t["M"] + GF[v] - t["G"]
                    tt["G"] += GF[v]
        pooled[f"{f}|{d}"] = dict(U_per_G=round(tt["U"] / tt["G"], 3), D_per_G=round(tt["D"] / tt["G"], 3),
                                  M_per_G=round(tt["M"] / tt["G"], 3), e=round((tt["P"] - tt["G"]) / tt["G"], 3))
N["pooled_terms_by_interval"] = pooled
bc = []
for (f, sd), m in maps.items():
    for (v, s), b in m.items():
        if b.startswith("("):
            lo, hi = (int(x) for x in b[1:-1].split(","))
            bc += [cov(T[(f, sd, v, s, lo)], v), cov(T[(f, sd, v, s, hi)], v)]
near = [(cov(t, k[2]), k) for k, t in T.items() if abs(err(t, k[2])) <= 0.10]
N["coverage_at_sign_change"] = dict(n=len(bc), max=round(max(bc), 4), median=round(st.median(bc), 3))
N["near_zero"] = dict(arms=len(T), within_0p1=len(near), max=round(max(c for c, _ in near), 4),
                      median=round(st.median(c for c, _ in near), 3),
                      over_half=sum(c > 0.5 for c, _ in near))
hi_cov = [cov(t, k[2]) for k, t in T.items() if k[4] == 1]
N["coverage_delta1"] = dict(median=round(st.median(hi_cov), 3), max=round(max(hi_cov), 4))

# ------------------------------------------------------------------ matched design
mt = {}
for f in FAMS:
    for sd in SEEDS:
        rise = tie = opp = n = 0
        for v in VIDEOS:
            for s in SIGMAS:
                dense = REC[(f, sd, v, s, 1)]
                for k in DELTAS[1:]:
                    sub = dict(dense)
                    for key in PER_FRAME:
                        sub[key] = dense[key][0::k]
                    sparse = REC[(f, sd, v, s, k)]
                    assert sub["frame_gt_ids"] == sparse["frame_gt_ids"]
                    a, b = decompose(sub, 0.5, 1), T[(f, sd, v, s, k)]
                    n += 1; rise += a["P"] > b["P"]; tie += a["P"] == b["P"]
                    opp += a["P"] > a["G"] > b["P"]
        mt[f"{f}|{sd}"] = dict(n=n, rises=rise, ties=tie, opposite=opp)
N["matched"] = mt
N["matched_total"] = {k: sum(x[k] for x in mt.values()) for k in ("n", "rises", "ties", "opposite")}

# ------------------------------------------------------------------ controls (whole, seed 0)
base = lambda v, s, d: T[("whole", 0, v, s, d)]
_m0 = maps[("whole", 0)]


def ctl(fam):
    data = {(v, s, d): decompose(json.loads((RUN / fam / f"{v}_s{s}_d{d}.json").read_text())["videos"][0], 0.5, 1)
            for v in VIDEOS for s in SIGMAS for d in (DELTAS if fam == "bytetrack" else DELTAS[1:])}
    get = lambda v, s, d: data[(v, s, d)] if (v, s, d) in data else base(v, s, d)
    m = {(v, s): bracket([err(get(v, s, d), v) for d in DELTAS]) for v in VIDEOS for s in SIGMAS}
    ss = [(err(get(v, s, a), v), err(get(v, s, b), v)) for v in VIDEOS for s in SIGMAS for a, b in zip(DELTAS, DELTAS[1:])]
    out = dict(down=sum(y < x for x, y in ss), tie=sum(y == x for x, y in ss), up=sum(y > x for x, y in ss),
               sign_change_rows=sum(b.startswith("(") for b in m.values()),
               sequences=len({v for (v, s), b in m.items() if b.startswith("(")}),
               moved=sum(m[k] != _m0[k] for k in m))
    if fam != "bytetrack":
        out["median_error_change"] = round(st.median(err(data[(v, s, d)], v) - err(base(v, s, d), v)
                                                     for v in VIDEOS for s in SIGMAS for d in DELTAS[1:]), 3)
    return out


N["controls"] = {fam: ctl(fam) for fam in ("ret", "cu", "rt", "bytetrack")}
ph = {}
for v in VIDEOS:
    for s in SIGMAS:
        for d in DELTAS[1:]:
            es = [err(base(v, s, d), v)] + [err(decompose(json.loads((RUN / "phase" / f"{v}_s{s}_d{d}_o{k}.json").read_text())
                                                          ["videos"][0], 0.5, 1), v) for k in range(1, d)]
            ph[(v, s, d)] = es
N["controls"]["phase"] = dict(
    cells=len(ph), sign_flip_cells=sum((min(x) < 0) != (max(x) < 0) for x in ph.values()),
    median_span={d: round(st.median(max(x) - min(x) for (v, s, dd), x in ph.items() if dd == d), 3) for d in DELTAS[1:]},
    steps_down_lowest=sum(
        min(ph[(v, s, b)]) < (err(base(v, s, 1), v) if a == 1 else min(ph[(v, s, a)]))
        for v in VIDEOS for s in SIGMAS for a, b in zip(DELTAS, DELTAS[1:])),
    steps_down_highest=sum(
        max(ph[(v, s, b)]) < (err(base(v, s, 1), v) if a == 1 else max(ph[(v, s, a)]))
        for v in VIDEOS for s in SIGMAS for a, b in zip(DELTAS, DELTAS[1:])),
    steps_down_mean=sum(
        st.fmean(ph[(v, s, b)]) < (err(base(v, s, 1), v) if a == 1 else st.fmean(ph[(v, s, a)]))
        for v in VIDEOS for s in SIGMAS for a, b in zip(DELTAS, DELTAS[1:])))

# ------------------------------------------------------------------ detection output vs seed
rec = {}
for v in VIDEOS:
    for sd in SEEDS:
        ov = json.loads((RUN / f"surface_seed{sd}" / f"{v}_s3072_d1.json").read_text())["overall"]
        fit = json.loads((RUN / f"{v}_s{sd}_verified.json").read_text())["budget"]["best_ap"]
        rec[(v, sd)] = (fit, ov["recall"], err(T[("whole", sd, v, 3072, 1)], v))
agree_fit = agree_rec = n = 0
for v in VIDEOS:
    for a in SEEDS:
        for b in SEEDS:
            if a < b:
                n += 1
                agree_fit += (rec[(v, a)][0] - rec[(v, b)][0]) * (rec[(v, a)][2] - rec[(v, b)][2]) > 0
                agree_rec += (rec[(v, a)][1] - rec[(v, b)][1]) * (rec[(v, a)][2] - rec[(v, b)][2]) > 0
N["seed_mechanism_3072_d1"] = dict(pairs=n, validation_fitness_orders_error=agree_fit, box_recall_orders_error=agree_rec,
                                  PP4={sd: dict(fitness=rec[("PathPlanning_4", sd)][0], recall=round(rec[("PathPlanning_4", sd)][1], 3),
                                                e=round(rec[("PathPlanning_4", sd)][2], 3)) for sd in SEEDS})
N["training"] = {k.replace("_verified.json", ""): json.loads((RUN / k).read_text())["budget"]
                 for k in sorted(p.name for p in RUN.glob("PathPlanning_*_s?_verified.json"))}

# ------------------------------------------------------------------ cost (whole family, queue timing)
cost = {}
for sd in SEEDS:
    for v in VIDEOS:
        for s in SIGMAS:
            ann = len(REC[("whole", sd, v, s, 1)]["frame_gt_ids"])
            for d in DELTAS:
                ms = list(REC[("whole", sd, v, s, d)]["frame_ms"])
                ms[0] = st.median(ms)
                cost.setdefault(f"{s}|{d}", []).append(sum(ms) / ann)
N["cost_ms_per_annotated_frame"] = {k: round(st.median(x), 1) for k, x in cost.items()}

# ------------------------------------------------------------------ identity metrics
IM = HERE / "results/identity_metrics.json"
if IM.exists():
    I = json.loads(IM.read_text())
    key = lambda sd, v, s, d: f"surface_seed{sd}|{v}|{s}|{d}"
    hot = {}
    for mode in ("processed", "common"):
        hot[mode] = {f"{s}|{d}": round(st.median(I[key(sd, v, s, d)][mode]["HOTA"] for sd in SEEDS for v in VIDEOS), 3)
                     for s in SIGMAS for d in DELTAS}
    N["hota_median"] = hot
    same = 0; loss = []; e_at_best = []; d_at_best = Counter()
    for sd in SEEDS:
        for v in VIDEOS:
            arms = [(s, d) for s in SIGMAS for d in DELTAS]
            bh = max(arms, key=lambda a: I[key(sd, v, *a)]["processed"]["HOTA"])
            be = min(arms, key=lambda a: abs(err(T[("whole", sd, v, *a)], v)))
            same += bh == be
            loss.append(1 - I[key(sd, v, *be)]["processed"]["HOTA"] / I[key(sd, v, *bh)]["processed"]["HOTA"])
            e_at_best.append(abs(err(T[("whole", sd, v, *bh)], v)))
            d_at_best[bh[1]] += 1
    N["hota_vs_count"] = dict(cases=21, same_arm=same, median_hota_loss_choosing_count=round(st.median(loss), 3),
                              median_abs_e_choosing_hota=round(st.median(e_at_best), 3), best_hota_interval=dict(d_at_best))
    mono = sum(all(I[key(sd, v, s, a)]["processed"]["HOTA"] > I[key(sd, v, s, b)]["processed"]["HOTA"]
                   for a, b in zip(DELTAS, DELTAS[1:])) for sd in SEEDS for v in VIDEOS for s in SIGMAS)
    N["hota_falls_with_interval_rows"] = f"{mono}/105"

# ------------------------------------------------------------------ oracle (association alone)
if (ORACLE / "status.json").exists() and json.loads((ORACLE / "status.json").read_text()).get("state") == "completed":
    orc = {}
    for arm in ("botsort_gmc", "bytetrack"):
        for v in ALL11:
            for d in DELTAS:
                for k in range(d):
                    r = json.loads((ORACLE / arm / f"{v}_d{d}_o{k}.json").read_text())["videos"][0]
                    orc[(arm, v, d, k)] = decompose(r, 0.5, 1)
    GO = {v: orc[("botsort_gmc", v, 1, 0)]["G"] for v in ALL11}
    oe = lambda t, v: (t["P"] - GO[v]) / GO[v]
    ocov = lambda t, v: 1 - (t["M"] + GO[v] - t["G"]) / GO[v]
    O = {}
    for arm in ("botsort_gmc", "bytetrack"):
        for grp, vids in (("multi-view PP2-8", VIDEOS), ("all eleven", ALL11)):
            ph0 = [(oe(orc[(arm, v, a, 0)], v), oe(orc[(arm, v, b, 0)], v)) for v in vids for a, b in zip(DELTAS, DELTAS[1:])]
            meanph = {(v, d): st.fmean(oe(orc[(arm, v, d, k)], v) for k in range(d)) for v in vids for d in DELTAS}
            O[f"{arm}|{grp}"] = dict(
                steps_down_phase0=sum(y < x for x, y in ph0), steps=len(ph0),
                steps_down_phase_mean=sum(meanph[(v, b)] < meanph[(v, a)] for v in vids for a, b in zip(DELTAS, DELTAS[1:])),
                error_delta1_median=round(st.median(oe(orc[(arm, v, 1, 0)], v) for v in vids), 3),
                error_delta8_median=round(st.median(meanph[(v, 8)] for v in vids), 3),
                D_per_G_delta1=round(sum(orc[(arm, v, 1, 0)]["D"] for v in vids) / sum(GO[v] for v in vids), 3),
                D_per_G_delta8=round(sum(st.fmean(orc[(arm, v, 8, k)]["D"] for k in range(8)) for v in vids) / sum(GO[v] for v in vids), 3),
                M_per_G_delta8=round(sum(st.fmean(orc[(arm, v, 8, k)]["M"] + GO[v] - orc[(arm, v, 8, k)]["G"] for k in range(8))
                                         for v in vids) / sum(GO[v] for v in vids), 3),
                U_total=sum(orc[(arm, v, d, k)]["U"] for v in vids for d in DELTAS for k in range(d)),
                coverage_delta1_min=round(min(ocov(orc[(arm, v, 1, 0)], v) for v in vids), 3),
                per_video={v: [round(meanph[(v, d)], 3) for d in DELTAS] for v in vids})
    N["oracle"] = O

# ------------------------------------------------------------------ StrongSORT (whole, seed 0)
if STRONG.exists() and len(list(STRONG.glob("*.json"))) == 140:
    so = {(v, s, d): decompose(json.loads((STRONG / f"{v}_s{s}_d{d}.json").read_text())["videos"][0], 0.5, 1)
          for v in VIDEOS for s in SIGMAS for d in DELTAS}
    ss = [(err(so[(v, s, a)], v), err(so[(v, s, b)], v)) for v in VIDEOS for s in SIGMAS for a, b in zip(DELTAS, DELTAS[1:])]
    N["strongsort"] = dict(down=sum(y < x for x, y in ss), tie=sum(y == x for x, y in ss), up=sum(y > x for x, y in ss),
                           up_by_step={f"{a}->{b}": sum(err(so[(v, s, b)], v) > err(so[(v, s, a)], v) for v in VIDEOS for s in SIGMAS)
                                       for a, b in zip(DELTAS, DELTAS[1:])})

out = HERE / "results/journal_numbers.json"
if "--check" in sys.argv:
    old = json.loads(out.read_text())
    new = json.loads(json.dumps(N, sort_keys=True, default=str))
    bad = [k for k in old if old[k] != new.get(k)]
    print("identical" if not bad else f"DIFFERS: {bad}")
    sys.exit(1 if bad else 0)
out.write_text(json.dumps(N, indent=1, sort_keys=True, default=str))
print(json.dumps({k: v for k, v in N.items() if k not in ("crossing_map", "training")}, indent=1, default=str)[:12000])
