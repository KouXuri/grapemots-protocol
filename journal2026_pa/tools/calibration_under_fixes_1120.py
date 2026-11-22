#!/usr/bin/env python3
"""Which detector errors keep a bunch count from being calibrated vine by vine?  (new, 2026-11-20)

Reads the error-substitution outputs (runs/error_sources_1120/substitution/<set>/<set>__<video>_s<sigma>.json,
written by tools/error_sources_1120.py substitute): for every subset S of the five TIDE-style fixes
(bkg, loc, jit, miss, score) and every interval, the per-video count P(S). For each (scale, interval) the
seven vines give:
  - the leave-one-vine-out ratio-calibration error |k P_v / G_v - 1|, k fitted on the other six vines;
  - Spearman rho between P_v and G_v (vine ranking).
F(S) = median calibration error over (scale, interval, vine); R(S) = median rho over (scale, interval).
Exact Shapley shares of F and R over the five fixes (Shapley 1953; Shorrocks 2013) say which error type
the vine-level failure belongs to. Nothing is re-run.
    .venv/bin/python cadence_journal_1119/tools/calibration_under_fixes_1120.py [--set whole_s0] [--sigma 3072]
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import math
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "runs/error_sources_1120/substitution"
FACTORS = ("bkg", "loc", "jit", "miss", "score")
DELTAS = ("1", "2", "4", "8")
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
KEY = lambda S: "+".join(sorted(S)) or "none"

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="whole_s0")
ap.add_argument("--sigma", type=int, default=None)
a = ap.parse_args()

R = {}
for p in sorted(SUB.glob(f"*/{a.set}__PathPlanning_*_s*.json")):   # whole_s0/ and other/
    if p.stem.endswith(("_none", "_all")):
        continue
    r = json.loads(p.read_text())
    if a.sigma and r["imgsz"] != a.sigma:
        continue
    R[(r["video"], r["imgsz"])] = r
sigmas = sorted({s for _, s in R})
assert all((v, s) in R for v in VIDEOS for s in sigmas), "incomplete substitution set"
G = {v: R[(v, sigmas[0])]["G_full"] for v in VIDEOS}
SUBSETS = [tuple(c) for n in range(6) for c in it.combinations(FACTORS, n)]


def stats(S):
    errs, rhos, skipped = [], [], 0
    for s in sigmas:
        for d in DELTAS:
            P = {v: R[(v, s)]["arms"][d]["values"][KEY(S)]["P"] for v in VIDEOS}
            for v in VIDEOS:
                den = sum(P[u] for u in VIDEOS if u != v)
                if den == 0:
                    skipped += 1
                    continue
                k = sum(G[u] for u in VIDEOS if u != v) / den
                errs.append(abs(k * P[v] / G[v] - 1))
            rho = spearmanr([P[v] for v in VIDEOS], [G[v] for v in VIDEOS]).statistic
            if not math.isnan(rho):
                rhos.append(rho)
    return dict(F=st.median(errs), R=st.median(rhos), n_err=len(errs), n_rho=len(rhos), skipped=skipped)


val = {KEY(S): stats(S) for S in SUBSETS}


def shapley(metric):
    m = len(FACTORS)
    out = {}
    for f in FACTORS:
        rest = [x for x in FACTORS if x != f]
        tot = 0.0
        for n in range(m):
            for S in it.combinations(rest, n):
                w = math.factorial(n) * math.factorial(m - 1 - n) / math.factorial(m)
                tot += w * (val[KEY(S + (f,))][metric] - val[KEY(S)][metric])
        out[f] = round(tot, 4)
    assert abs(sum(out.values()) - (val[KEY(FACTORS)][metric] - val["none"][metric])) < 1e-3
    return out


res = dict(set=a.set, sigmas=sigmas, G=G,
           none={k: (round(x, 4) if isinstance(x, float) else x) for k, x in val["none"].items()},
           all={k: (round(x, 4) if isinstance(x, float) else x) for k, x in val[KEY(FACTORS)].items()},
           single={f: dict(F=round(val[f]["F"], 4), R=round(val[f]["R"], 4)) for f in FACTORS},
           all_but={f: dict(F=round(val[KEY(tuple(x for x in FACTORS if x != f))]["F"], 4),
                            R=round(val[KEY(tuple(x for x in FACTORS if x != f))]["R"], 4)) for f in FACTORS},
           shapley_F=shapley("F"), shapley_R=shapley("R"))
dest = ROOT / "cadence_journal_1119/results" / f"calibration_under_fixes_{a.set}{'_s' + str(a.sigma) if a.sigma else ''}.json"
dest.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
