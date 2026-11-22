#!/usr/bin/env python3
"""Two checks a reviewer would ask of the vine-level claims (2026-11-21; frozen outputs only).

1. Best case, not the median configuration: per configuration (set, seed, scale, interval) the median
   over the seven vines of the leave-one-vine-out ratio-calibration error; the five best configurations;
   and a nested choice that does not peek at the held-out vine: for each held-out vine v the configuration
   is chosen by its median inner leave-one-out error on the other six vines (each inner fit on five), then
   calibrated on those six and applied to v.
2. A seed ensemble: the count averaged over the three training seeds of one recipe (same set, scale,
   interval), calibrated leave-one-vine-out, against single-seed counts of the same recipes.
Reuses the records and helpers of calibration_transfer_1120.py (run first, unchanged).
    .venv/bin/python cadence_journal_1119/tools/calibration_robustness_1121.py
"""
from __future__ import annotations

import json
import runpy
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NS = runpy.run_path(str(ROOT / "cadence_journal_1119/tools/calibration_transfer_1120.py"), run_name="calib")
P, GF, CONFIGS, VIDEOS, summ = NS["P"], NS["GF"], NS["CONFIGS"], NS["VIDEOS"], NS["summ"]
SEEDS, SIGMAS, DELTAS, FAMS = NS["SEEDS"], NS["SIGMAS"], NS["DELTAS"], NS["FAMS"]


def count(c, v):
    f, sd, s, d = c
    return P[(f, sd, v, s, d)]


def lovo_err(cnt, v, pool):
    """|k cnt[v]/G_v - 1| with k fitted on pool (v not in pool)."""
    den = sum(cnt[u] for u in pool)
    if den == 0:
        return float("inf")
    return abs(sum(GF[u] for u in pool) / den * cnt[v] / GF[v] - 1)


out = {}
# 1a. per-configuration median over vines
per_cfg = {}
for c in CONFIGS:
    cnt = {v: count(c, v) for v in VIDEOS}
    per_cfg[c] = st.median(lovo_err(cnt, v, [u for u in VIDEOS if u != v]) for v in VIDEOS)
best = sorted(per_cfg.items(), key=lambda kv: kv[1])[:5]
out["best5_configs_median_lovo"] = [dict(config="|".join(map(str, c)), median_lovo=round(x, 3)) for c, x in best]
out["config_median_lovo_distribution"] = dict(min=round(min(per_cfg.values()), 3),
                                              q25=round(sorted(per_cfg.values())[len(per_cfg) // 4], 3),
                                              median=round(st.median(per_cfg.values()), 3))
# 1b. nested choice
outer, chosen = [], []
for v in VIDEOS:
    rest = [u for u in VIDEOS if u != v]
    inner = {}
    for c in CONFIGS:
        cnt = {u: count(c, u) for u in VIDEOS}
        inner[c] = st.median(lovo_err(cnt, u, [w for w in rest if w != u]) for u in rest)
    cstar = min(inner, key=inner.get)
    cnt = {u: count(cstar, u) for u in VIDEOS}
    outer.append(lovo_err(cnt, v, rest))
    chosen.append("|".join(map(str, cstar)))
out["nested_choice"] = dict(per_vine=summ(outer), chosen=dict(zip(VIDEOS, chosen)))

# 2. seed ensemble
ens, single = [], []
for f in FAMS:
    for s in SIGMAS:
        for d in DELTAS:
            cnt = {v: st.fmean(P[(f, sd, v, s, d)] for sd in SEEDS) for v in VIDEOS}
            ens += [lovo_err(cnt, v, [u for u in VIDEOS if u != v]) for v in VIDEOS]
            for sd in SEEDS:
                c1 = {v: P[(f, sd, v, s, d)] for v in VIDEOS}
                single += [lovo_err(c1, v, [u for u in VIDEOS if u != v]) for v in VIDEOS]
out["seed_ensemble_lovo"] = summ(ens)
out["single_seed_lovo_same_recipes"] = summ(single)

dest = ROOT / "cadence_journal_1119/results/calibration_robustness_1121.json"
dest.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
