#!/usr/bin/env python3
"""Three numbers of the PA manuscript that had no frozen result file (computed 2026-11-21).

1. Per-frame box recall of the six detector sets at sigma 3072: pooled TP / reference boxes over the seven test
   videos, per-video range, and the per-video median true-positive confidence (detection_stats of the
   substitution outputs, runs/error_sources_1120/substitution/*/<set>__<video>_s3072.json).
2. Direction of the interval effect under every one of the 32 combinations of fixes: for each detector set at
   sigma 3072 (and for the seed-0 whole-frame set over all scales), the combinations in which the count rose more
   often than it fell between consecutive intervals (same files, arms[d]["values"][subset]["P"]).
3. Validation mAP50:95 against the count error: within each fold, for each of the 3 seed pairs, whether the seed
   with the higher validation mAP50:95 (manuscript Table S2) has the larger absolute count error, under four
   summaries of the error (results/crop_vs_whole.json, whole-frame detectors).
Output: results/supplementary_checks_1121.json
    .venv/bin/python cadence_journal_1119/tools/supplementary_checks_1121.py
"""
from __future__ import annotations

import glob
import itertools as it
import json
import statistics as st
from pathlib import Path
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_batch8_1127  # noqa: E402  batch-8 retrain of crop PathPlanning_4_s2 (ledger 10.28.17, R rule)

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SUB = ROOT / "runs/error_sources_1120/substitution"
SETS = ("crop_s0", "crop_s1", "crop_s2", "whole_s0", "whole_s1", "whole_s2")
F = ("bkg", "loc", "jit", "miss", "score")
KEY = lambda S: "+".join(sorted(S)) or "none"
SUBSETS = [c for n in range(6) for c in it.combinations(F, n)]


def files(pattern):
    return [f for f in sorted(glob.glob(str(SUB / "*" / pattern))) if not f.endswith(("_none.json", "_all.json"))]


out = {"recall_s3072": {}, "reversal": {}, "validation_vs_count": {}}
for s in SETS:
    fs = files(f"{s}__PathPlanning_*_s3072.json")
    assert len(fs) == 7, (s, len(fs))
    st_ = [crop_batch8_1127.read_substitution(f)["detection_stats"] for f in fs]
    out["recall_s3072"][s] = dict(pooled=round(sum(x["tp"] for x in st_) / sum(x["reference_boxes"] for x in st_), 4),
                                  per_video_min=min(x["recall_iou50"] for x in st_),
                                  per_video_max=max(x["recall_iou50"] for x in st_),
                                  median_tp_conf_min=min(x["median_tp_score"] for x in st_),
                                  median_tp_conf_max=max(x["median_tp_score"] for x in st_))


def reversal(fs):
    res = {}
    for S in SUBSETS:
        k = KEY(S)
        down = up = tie = 0
        for f in fs:
            r = crop_batch8_1127.read_substitution(f)
            P = [r["arms"][d]["values"][k]["P"] for d in ("1", "2", "4", "8")]
            for a, b in zip(P, P[1:]):
                down += b < a; up += b > a; tie += b == a
        res[k] = dict(down=down, up=up, tie=tie)
    return dict(steps=res, more_rises_than_falls=sorted(k for k, v in res.items() if v["up"] > v["down"]))


for s in SETS:
    out["reversal"][f"{s}|s3072"] = reversal(files(f"{s}__PathPlanning_*_s3072.json"))
out["reversal"]["whole_s0|all_scales"] = reversal(files("whole_s0__PathPlanning_*_s*.json"))
rev = [v["more_rises_than_falls"] for v in out["reversal"].values()]
out["reversal"]["every_reversing_subset_has_miss_and_score"] = all("miss" in k and "score" in k for r in rev for k in r)
out["reversal"]["every_reversing_subset_also_fixes_loc_or_bkg"] = all(("loc" in k or "bkg" in k) for r in rev for k in r)
out["reversal"]["miss_plus_score_alone_reverses_anywhere"] = any("miss+score" in r for r in rev)

C = json.loads((HERE / "results/crop_vs_whole.json").read_text())["error"]
VAL = {"PathPlanning_2": (0.189, 0.188, 0.203), "PathPlanning_3": (0.172, 0.183, 0.186),
       "PathPlanning_4": (0.156, 0.150, 0.162), "PathPlanning_5": (0.103, 0.096, 0.107),
       "PathPlanning_6": (0.197, 0.209, 0.190), "PathPlanning_7": (0.161, 0.159, 0.149),
       "PathPlanning_8": (0.076, 0.073, 0.066)}   # validation mAP50:95 per seed, manuscript Table S2
S_, D_ = (1536, 2048, 2560, 3072, 3840), (1, 2, 4, 8)
SUMS = {"abs_e_sigma3072_15Hz": lambda v, sd: abs(C[f"whole|{sd}|{v}|3072|1"]),
        "median_abs_e_20_cells": lambda v, sd: st.median(abs(C[f"whole|{sd}|{v}|{s}|{d}"]) for s in S_ for d in D_),
        "mean_abs_e_sigma3072_4_intervals": lambda v, sd: st.mean(abs(C[f"whole|{sd}|{v}|3072|{d}"]) for d in D_),
        "mean_abs_e_20_cells": lambda v, sd: st.mean(abs(C[f"whole|{sd}|{v}|{s}|{d}"]) for s in S_ for d in D_)}
for name, f in SUMS.items():
    n = tot = 0
    for v, m in VAL.items():
        for a, b in it.combinations(range(3), 2):
            hi, lo = (a, b) if m[a] > m[b] else (b, a)
            tot += 1
            n += f(v, hi) > f(v, lo)
    out["validation_vs_count"][name] = dict(higher_map_has_larger_error=n, pairs=tot)
# 4. Unrounded values behind printed numbers whose frozen value ends in 5 at the third decimal
#    (journal_numbers.py run in --check mode, which compares and does not write; calibration_transfer_1120.py)
import runpy
import sys
jn_path = HERE / "tools/journal_numbers.py"
ns = {"__name__": "journal_numbers_exact", "__file__": str(jn_path)}
argv, sys.argv = sys.argv, [str(jn_path), "--check"]
try:
    exec(compile(jn_path.read_text(), str(jn_path), "exec"), ns)
except SystemExit as ex:
    assert ex.code in (0, None), "journal_numbers.py --check differs"
finally:
    sys.argv = argv
T, GFj = ns["T"], ns["GF"]
exact = {}
for f in ("crop", "whole"):
    for d in (1, 8):
        tt = dict(P=0, U=0, D=0, M=0, G=0)
        for sd in (0, 1, 2):
            for v in [f"PathPlanning_{i}" for i in range(2, 9)]:
                for sg in (1536, 2048, 2560, 3072, 3840):
                    t = T[(f, sd, v, sg, d)]
                    tt["P"] += t["P"]; tt["U"] += t["U"]; tt["D"] += t["D"]; tt["M"] += t["M"] + GFj[v] - t["G"]
                    tt["G"] += GFj[v]
        exact[f"pooled|{f}|{d}"] = dict(U_per_G=tt["U"] / tt["G"], D_per_G=tt["D"] / tt["G"], M_per_G=tt["M"] / tt["G"],
                                        e=(tt["P"] - tt["G"]) / tt["G"], U=tt["U"], D=tt["D"], M=tt["M"], G=tt["G"])
ct = runpy.run_path(str(HERE / "tools/calibration_transfer_1120.py"), run_name="calib_exact")
k_of, pv, G_TOT = ct["k_of"], ct["pv"], ct["G_TOT"]
pairs = [((a, sd, sg, d), (b, sd, sg, d)) for sd in (0, 1, 2) for sg in (1536, 2048, 2560, 3072, 3840) for d in (1, 2, 4, 8)
         for a, b in it.permutations(("crop", "whole"), 2)]
vals = [abs(k_of(c0) * sum(pv(c1, v) for v in ct["VIDEOS"]) / G_TOT - 1) for c0, c1 in pairs]
exact["transfer_representation_pooled_median"] = st.median(vals)
# exact values from the substitution counts (P, G) at sigma 3072: miss-alone ranges, miss Shapley shares, seed spread
import math
ARM = {}
for st_name in SETS:
    for f in files(f"{st_name}__PathPlanning_*_s3072.json"):
        r = crop_batch8_1127.read_substitution(f)
        ARM[(st_name, r["video"])] = r
def e_arm(r, d, k):
    return (r["arms"][d]["values"][k]["P"] - r["G_full"]) / r["G_full"]
def e_pool(st_name, d, k):
    rs = [ARM[(st_name, v)] for v in VIDEOS_ALL]
    G = sum(r["G_full"] for r in rs)
    return (sum(r["arms"][d]["values"][k]["P"] for r in rs) - G) / G
VIDEOS_ALL = [f"PathPlanning_{i}" for i in range(2, 9)]
ma = {}
for d in ("1", "2", "4", "8"):
    ma[d] = {st_name: dict(real=e_pool(st_name, d, "none"), miss_alone=e_pool(st_name, d, "miss")) for st_name in SETS}
exact["miss_alone_sigma3072"] = ma
def shap_miss(st_name, d):
    rest = [x for x in F if x != "miss"]; tot = 0.0
    for n in range(5):
        for S in it.combinations(rest, n):
            w = math.factorial(n) * math.factorial(4 - n) / math.factorial(5)
            tot += w * (e_pool(st_name, d, KEY(S + ("miss",))) - e_pool(st_name, d, KEY(S)))
    return tot
exact["shapley_miss_d1_sigma3072"] = {st_name: shap_miss(st_name, "1") for st_name in SETS}
h2 = {}
for rep in ("crop", "whole"):
    fams = [x for x in SETS if x.startswith(rep)]
    for k in ("none", "miss"):
        rng = [max(e_arm(ARM[(f_, v)], d, k) for f_ in fams) - min(e_arm(ARM[(f_, v)], d, k) for f_ in fams)
               for v in VIDEOS_ALL for d in ("1", "2", "4", "8")]
        h2[f"{rep}|{k}"] = dict(cells=len(rng), median_range=st.median(rng))
exact["seed_spread_sigma3072"] = h2
out["exact"] = exact
dest = HERE / "results/supplementary_checks_1121.json"
dest.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("recall_s3072", "validation_vs_count")}, indent=1))
print({k: v for k, v in out["reversal"].items() if not isinstance(v, dict)})
print({k: v["more_rises_than_falls"] for k, v in out["reversal"].items() if isinstance(v, dict)})
print(json.dumps(out["exact"], indent=1))
