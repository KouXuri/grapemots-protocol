#!/usr/bin/env python3
"""H4b: does the vine-level failure hold for other associators on the same real detections? (2026-11-21)

Same 35 seed-0 whole-frame detection dumps (conf 0.25), four associators:
  botsort     runs/fullframe_pponly_0922/surface_seed0/{v}_s{s}_d{d}.json     (main grid)
  strongsort  runs/wholeframe_strongsort_1119/strongsort/{v}_s{s}_d{d}.json   (BoxMOT, OSNet)
  ocsort      runs/error_sources_1120/real_ocsort/tracks/{v}_s{s}_d{d}.json   (BoxMOT 23.0.0)
  deepocsort  runs/error_sources_1120/real_deepocsort/tracks/{v}_s{s}_d{d}.json
Per (scale, interval): per-vine count P (eligibility 1, the main grid's rule), leave-one-vine-out ratio
calibration error, Spearman rho over the seven vines; pooled e, U, D, M per reference trajectory.
Pre-registered reading (H4b, ledger 10.28.8): median calibration error <= 0.243 (BoT-SORT with perfect
detection) -> the vine-level failure is specific to the associator; otherwise it holds across families.
    .venv/bin/python cadence_journal_1119/tools/real_trackers_vine_1121.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import decompose  # noqa: E402

VINES = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SRC = {"botsort": lambda v, s, d: ROOT / f"runs/fullframe_pponly_0922/surface_seed0/{v}_s{s}_d{d}.json",
       "strongsort": lambda v, s, d: ROOT / f"runs/wholeframe_strongsort_1119/strongsort/{v}_s{s}_d{d}.json",
       "ocsort": lambda v, s, d: ROOT / f"runs/error_sources_1120/real_ocsort/tracks/{v}_s{s}_d{d}.json",
       "deepocsort": lambda v, s, d: ROOT / f"runs/error_sources_1120/real_deepocsort/tracks/{v}_s{s}_d{d}.json"}
T = {}
for trk, src in SRC.items():
    for v in VINES:
        for s in SIGMAS:
            for d in DELTAS:
                r = json.loads(src(v, s, d).read_text())["videos"][0]
                r.setdefault("video", v)
                T[(trk, v, s, d)] = decompose(r, 0.5, 1)
GF = {v: T[("botsort", v, 3072, 1)]["G"] for v in VINES}
G_TOT = sum(GF.values())
out = {"G_full": GF}
for trk in SRC:
    errs, rhos, pooled = [], [], {d: [] for d in DELTAS}
    terms = {d: {"U": 0, "D": 0, "M": 0, "P": 0, "G": 0} for d in DELTAS}
    for s in SIGMAS:
        for d in DELTAS:
            P = {v: T[(trk, v, s, d)]["P"] for v in VINES}
            for v in VINES:
                den = sum(P[u] for u in VINES if u != v)
                if den:
                    errs.append(abs(sum(GF[u] for u in VINES if u != v) / den * P[v] / GF[v] - 1))
            rho = spearmanr([P[v] for v in VINES], [GF[v] for v in VINES]).statistic
            if not math.isnan(rho):
                rhos.append(rho)
            pooled[d].append(sum(P.values()) / G_TOT - 1)
            for v in VINES:
                t = T[(trk, v, s, d)]
                terms[d]["U"] += t["U"]; terms[d]["D"] += t["D"]; terms[d]["M"] += t["M"] + GF[v] - t["G"]
                terms[d]["P"] += t["P"]; terms[d]["G"] += GF[v]
    out[trk] = dict(
        lovo_median=round(st.median(errs), 3), lovo_share_gt_030=round(sum(e > 0.3 for e in errs) / len(errs), 3),
        rho_median=round(st.median(rhos), 3), rho_share_ge_0786=round(sum(r >= 0.786 for r in rhos) / len(rhos), 3),
        pooled_e_median_by_interval={d: round(st.median(x), 3) for d, x in pooled.items()},
        terms_per_G_by_interval={d: {k: round(x[k] / x["G"], 3) for k in ("U", "D", "M")} for d, x in terms.items()})
out["H4b_reading"] = {trk: ("associator-specific (<= 0.243)" if out[trk]["lovo_median"] <= 0.243 else
                            "holds across families (> 0.243)") for trk in SRC}
(ROOT / "cadence_journal_1119/results/real_trackers_vine_1121.json").write_text(json.dumps(out, indent=1))
for trk in SRC:
    x = out[trk]
    print(f"{trk:10s} lovo {x['lovo_median']:.3f} (>0.3: {x['lovo_share_gt_030']:.2f})  rho {x['rho_median']:.3f} "
          f"(sig {x['rho_share_ge_0786']:.2f})  pooled e by d {x['pooled_e_median_by_interval']}  terms d1 {x['terms_per_G_by_interval'][1]}")
print(out["H4b_reading"])
