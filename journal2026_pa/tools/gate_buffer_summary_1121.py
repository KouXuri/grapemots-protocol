#!/usr/bin/env python3
"""H5: do the two configuration-level interventions remove the duplicate class they target? (2026-11-21)

Reads the per-frame tracks kept by the replay (seed-0 whole-frame dumps, seven vines, five scales, four intervals):
  default  runs/error_sources_1120/substitution/whole_s0/whole_s0__{v}_s{s}_d{d}_{none,all}.json
  gate095  runs/error_sources_1120/gate_buffer_1121/gate095/gate095__{v}_s{s}_d{d}_{none,all}.json
  buf150   runs/error_sources_1120/gate_buffer_1121/buf150/buf150__{v}_s{s}_d{d}_{none,all}.json
'all' = reference boxes at 0.9 (identical at every scale, so sigma 3072 is read); 'none' = real detections.
Oracle: U, D, M per reference trajectory and the duplicate classes of oracle_trackers_summary_1120.py with the
same definition for every config (absence of >= 30 processed frames = re-entry). Real: pooled terms over the five
scales and the vine-level reading (leave-one-vine-out ratio calibration, Spearman rho) of calibration_transfer_1120.py.
Pre-registered readings (ledger 10.28.10): H5a switch-in-view duplicates at interval 1 fall >= 50% under gate095;
H5b re-entry duplicates fall >= 50% under buf150 (< 20% -> not re-linked).
    .venv/bin/python cadence_journal_1119/tools/gate_buffer_summary_1121.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
sys.path.insert(0, str(ROOT / "cadence_journal_1119/tools"))
from decompose_count_error import decompose  # noqa: E402

VINES = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SRC = {"default": lambda v, s, d, k: ROOT / f"runs/error_sources_1120/substitution/whole_s0/whole_s0__{v}_s{s}_d{d}_{k}.json",
       "gate095": lambda v, s, d, k: ROOT / f"runs/error_sources_1120/gate_buffer_1121/gate095/gate095__{v}_s{s}_d{d}_{k}.json",
       "buf150": lambda v, s, d, k: ROOT / f"runs/error_sources_1120/gate_buffer_1121/buf150/buf150__{v}_s{s}_d{d}_{k}.json"}

# the classification code of oracle_trackers_summary_1120.py, imported without running its main body
SUMMARY = ROOT / "cadence_journal_1119/tools/oracle_trackers_summary_1120.py"
src = SUMMARY.read_text()
ns: dict = {"__file__": str(SUMMARY), "__name__": "oracle_trackers_head"}
exec(src.split("GF = {v:")[0], ns)
classify = ns["classify"]

rec = lambda p: (lambda r: (r.setdefault("video", "x"), r)[1])(json.loads(p.read_text())["videos"][0])
GF = {v: len({g for f in rec(SRC["default"](v, 3072, 1, "all"))["frame_gt_ids"] for g in f}) for v in VINES}
G = sum(GF.values())
out = {"G_full": GF}
for cfg, src_ in SRC.items():
    if not all(src_(v, s, d, k).exists() for v in VINES for s in SIGMAS for d in DELTAS for k in ("none", "all")):
        out[cfg] = "incomplete"
        continue
    res = {}
    for d in DELTAS:
        c = Counter(); terms = Counter()
        for v in VINES:
            r = rec(src_(v, 3072, d, "all"))
            t = decompose(r, 0.5, 1)
            terms["P"] += t["P"]; terms["U"] += t["U"]; terms["D"] += t["D"]; terms["M"] += t["M"] + GF[v] - t["G"]
            c.update(classify(r, 30))
        real = Counter(); lovo = []; rhos = []
        for s in SIGMAS:
            P = {}
            for v in VINES:
                t = decompose(rec(src_(v, s, d, "none")), 0.5, 1)
                P[v] = t["P"]
                real["P"] += t["P"]; real["U"] += t["U"]; real["D"] += t["D"]; real["M"] += t["M"] + GF[v] - t["G"]
            for v in VINES:
                den = sum(P[u] for u in VINES if u != v)
                if den:
                    lovo.append(abs(sum(GF[u] for u in VINES if u != v) / den * P[v] / GF[v] - 1))
            rhos.append(spearmanr([P[v] for v in VINES], [GF[v] for v in VINES]).statistic)
        n = len(SIGMAS)
        res[str(d)] = dict(
            oracle=dict(e=round(terms["P"] / G - 1, 3), U=round(terms["U"] / G, 3), D=round(terms["D"] / G, 3),
                        M=round(terms["M"] / G, 3), dup_classes=dict(c)),
            real=dict(e=round(real["P"] / (n * G) - 1, 3), U=round(real["U"] / (n * G), 3),
                      D=round(real["D"] / (n * G), 3), M=round(real["M"] / (n * G), 3),
                      vine_lovo_median=round(st.median(lovo), 3), vine_rho_median=round(st.median(rhos), 3)))
    out[cfg] = res
if all(isinstance(out[c], dict) for c in SRC):
    sw = {c: out[c]["1"]["oracle"]["dup_classes"].get("switch_in_view", 0) for c in SRC}
    re_ = {c: out[c]["1"]["oracle"]["dup_classes"].get("reentry_beyond_buffer", 0) for c in SRC}
    red_a = 1 - sw["gate095"] / sw["default"]
    red_b = 1 - re_["buf150"] / re_["default"] if re_["default"] else float("nan")
    out["H5a"] = dict(switch_in_view=sw, reduction=round(red_a, 3),
                      reading="gate failures" if red_a >= 0.5 else ("not the gate value" if red_a < 0.2 else "partial"))
    out["H5b"] = dict(reentry_beyond_buffer=re_, reduction=round(red_b, 3),
                      reading="memory re-links" if red_b >= 0.5 else ("not re-linked" if red_b < 0.2 else "partial"))
dest = ROOT / "cadence_journal_1119/results/gate_buffer_summary_1121.json"
dest.write_text(json.dumps(out, indent=1))
for c in SRC:
    if not isinstance(out[c], dict):
        print(c, out[c]); continue
    for d in ("1", "8"):
        o, r = out[c][d]["oracle"], out[c][d]["real"]
        print(f"{c:8s} d{d} oracle e {o['e']:+.3f} D {o['D']:.3f} M {o['M']:.3f} {o['dup_classes']} | real e {r['e']:+.3f} "
              f"U {r['U']:.3f} D {r['D']:.3f} M {r['M']:.3f} vine {r['vine_lovo_median']:.3f} rho {r['vine_rho_median']:.3f}")
print(out.get("H5a"), out.get("H5b"))
