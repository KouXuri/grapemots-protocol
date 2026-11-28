#!/usr/bin/env python3
"""Summarise the error-substitution runs (runs/error_sources_1120/substitution).

Reads the per-arm JSON written by tools/error_sources_1120.py substitute (values of
P, U, D, M, e for all 32 fix subsets, and the exact Shapley shares), nothing else.
Reports, per detector set and interval:
  - pooled Shapley shares of each fix in U, D, M and e (sums over videos and scales,
    divided by the summed reference count, so they add up to the pooled change);
  - the effect of each fix applied alone;
  - H1 (pre-registered 2026-11-20, ledger 10.28.5): the number of interval steps that
    lower / raise the count with no fix, with background+localisation+confidence fixed,
    and with everything fixed;
  - H2 (when several seeds exist): the median range of e over seeds with no fix and
    with the miss fix alone.
    .venv/bin/python cadence_journal_1119/tools/error_sources_summary.py [--sigma 3072]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_batch8_1127  # noqa: E402  batch-8 retrain of crop PathPlanning_4_s2 (ledger 10.28.17, R rule)

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "runs/error_sources_1120/substitution"
FACTORS = ("bkg", "loc", "jit", "miss", "score")
DELTAS = ("1", "2", "4", "8")
KEY = lambda S: "+".join(sorted(S)) or "none"

ap = argparse.ArgumentParser()
ap.add_argument("--sigma", type=int, default=None, help="restrict to one inference scale")
ap.add_argument("--sets", nargs="*", default=None,
                help="restrict to these detector sets; results/error_sources_summary.json was written with --sets whole_s0")
a = ap.parse_args()

arms = {}
for p in sorted(SUB.glob("*/*.json")):
    if p.stem.endswith(("_none", "_all")) or "_d" in p.stem.split("__")[-1].split("_s")[-1]:
        continue
    r = crop_batch8_1127.read_substitution(p)
    if "arms" not in r:
        continue
    fam = p.stem.split("__")[0]
    if a.sigma and r["imgsz"] != a.sigma:
        continue
    if a.sets and fam not in a.sets:
        continue
    arms[(fam, r["video"], r["imgsz"])] = r

out = {"n_arms": len(arms), "sigma": a.sigma}
sets = sorted({k[0] for k in arms})
for fam in sets:
    rows = {k: r for k, r in arms.items() if k[0] == fam}
    G = sum(r["G_full"] for r in rows.values())
    res = {}
    for d in DELTAS:
        sh = {q: {f: sum(r["arms"][d]["shapley"][q][f] for r in rows.values()) / G for f in FACTORS}
              for q in ("U", "D", "M")}
        sh["e"] = {f: sum(r["arms"][d]["shapley"]["e"][f] * r["G_full"] for r in rows.values()) / G
                   for f in FACTORS}
        val = lambda S, q: sum(r["arms"][d]["values"][KEY(S)][q] for r in rows.values()) / G
        base = {q: val((), q) for q in ("U", "D", "M")}
        base["e"] = sum(r["arms"][d]["values"]["none"]["e"] * r["G_full"] for r in rows.values()) / G
        alone = {f: {q: (val((f,), q) - base[q]) if q != "e" else
                     (sum(r["arms"][d]["values"][f]["e"] * r["G_full"] for r in rows.values()) / G - base["e"])
                     for q in ("U", "D", "M", "e")} for f in FACTORS}
        allf = {q: val(FACTORS, q) for q in ("U", "D", "M")}
        res[d] = dict(none={q: round(x, 3) for q, x in base.items()},
                      all={q: round(x, 3) for q, x in allf.items()},
                      shapley={q: {f: round(x, 3) for f, x in v.items()} for q, v in sh.items()},
                      alone={f: {q: round(x, 3) for q, x in v.items()} for f, v in alone.items()})
    # H1: direction of the interval effect under fix subsets
    h1 = {}
    for S in [(), ("bkg",), ("loc",), ("jit",), ("miss",), ("score",), ("bkg", "loc", "score"),
              ("bkg", "jit", "loc", "score"), FACTORS]:
        down = up = tie = 0
        for r in rows.values():
            es = [r["arms"][d]["values"][KEY(S)]["e"] for d in DELTAS]
            for x, y in zip(es, es[1:]):
                down += y < x; up += y > x; tie += y == x
        h1[KEY(S)] = dict(down=down, up=up, tie=tie)
    out[fam] = dict(G_total=G, arms=len(rows), by_interval=res, h1_interval_steps=h1)

# H2: seed spread with and without the miss fix, per representation, same (video, sigma, delta)
for rep in ("crop", "whole"):
    fams = [f for f in sets if f.startswith(rep + "_s")]
    if len(fams) < 2:
        continue
    cells = {(v, s) for (f, v, s) in arms if f in fams}
    for S in [(), ("miss",), ("bkg", "loc", "score"), FACTORS]:
        rng = []
        for v, s in cells:
            for d in DELTAS:
                es = [arms[(f, v, s)]["arms"][d]["values"][KEY(S)]["e"] for f in fams if (f, v, s) in arms]
                if len(es) == len(fams):
                    rng.append(max(es) - min(es))
        out.setdefault(f"h2_{rep}", {})[KEY(S)] = dict(cells=len(rng), median_range=round(st.median(rng), 3) if rng else None)

dest = ROOT / "cadence_journal_1119/results" / f"error_sources_summary{'_s' + str(a.sigma) if a.sigma else ''}.json"
dest.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:6000])
