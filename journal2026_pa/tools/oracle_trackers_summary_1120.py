#!/usr/bin/env python3
"""H4: is the surplus under perfect detection a property of the associator? (2026-11-20)

Oracle runs (reference boxes as detections, phase 0) of five associators:
  botsort     runs/oracle_cadence_1119/botsort_gmc/{v}_d{d}_o0.json       (Ultralytics BoT-SORT + GMC)
  bytetrack   runs/oracle_cadence_1119/bytetrack/{v}_d{d}_o0.json         (Ultralytics ByteTrack)
  strongsort  runs/error_sources_1120/oracle_strongsort/tracks/{v}_s0_d{d}.json   (BoxMOT, OSNet)
  ocsort      runs/error_sources_1120/oracle_ocsort/tracks/{v}_s0_d{d}.json       (BoxMOT 23.0.0)
  deepocsort  runs/error_sources_1120/oracle_deepocsort/tracks/{v}_s0_d{d}.json   (BoxMOT 23.0.0)
For each: U, D, M per reference trajectory on the multi-view (PP2-8) and frontal (NP1-3) videos, at
eligibility tau = 1 and tau = 3 (a track counts only if it owns >= tau scored frames; OC-SORT and Deep
OC-SORT already withhold a track until 3 consecutive hits, so tau = 3 is the like-for-like control for
the other three), the duplicate classes of reentry_analysis_1120.py (code copied from
oracle_strongsort_summary.py unchanged), and the vine-level reading of calibration_transfer_1120.py:
per-vine P/G, leave-one-vine-out ratio-calibration error, Spearman rho over the seven vines.
Pre-registered reading (H4, same thresholds as H3, ledger 10.28.8): multi-view D/G >= 0.30 at interval 1
-> that associator does not keep identity through a change of viewpoint; <= 0.10 -> it does, and the
association limit is specific to the trackers tested before.
    .venv/bin/python cadence_journal_1119/tools/oracle_trackers_summary_1120.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import decompose, frame_matches  # noqa: E402

GROUP = lambda v: "NP1-3" if v.startswith("No") else ("PP1" if v == "PathPlanning_1" else "PP2-8")
VIDEOS = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]
VINES = [f"PathPlanning_{i}" for i in range(2, 9)]
SRC = {"botsort": lambda v, d: ROOT / f"runs/oracle_cadence_1119/botsort_gmc/{v}_d{d}_o0.json",
       "bytetrack": lambda v, d: ROOT / f"runs/oracle_cadence_1119/bytetrack/{v}_d{d}_o0.json",
       "strongsort": lambda v, d: ROOT / f"runs/error_sources_1120/oracle_strongsort/tracks/{v}_s0_d{d}.json",
       "ocsort": lambda v, d: ROOT / f"runs/error_sources_1120/oracle_ocsort/tracks/{v}_s0_d{d}.json",
       "deepocsort": lambda v, d: ROOT / f"runs/error_sources_1120/oracle_deepocsort/tracks/{v}_s0_d{d}.json"}
BUFFER = 30   # every associator here keeps a lost track for 30 processed frames (BoxMOT max_age default 30)


def classify(rec, buffer):
    overlap = defaultdict(Counter); first = {}
    for fi, (pb, pi, gb, gi) in enumerate(zip(rec["frame_predicted_boxes"], rec["frame_predicted_ids"],
                                              rec["frame_gt_boxes"], rec["frame_gt_ids"])):
        if pi:
            for p, g in frame_matches(pb, pi, gb, gi, 0.5):
                overlap[p][g] += 1; first.setdefault((p, g), fi)
    own = {p: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for p, c in overlap.items() if c}
    vis = defaultdict(set); matched = defaultdict(set)
    for fi, ids in enumerate(rec["frame_gt_ids"]):
        for g in ids:
            vis[g].add(fi)
    for fi, (pb, pi, gb, gi) in enumerate(zip(rec["frame_predicted_boxes"], rec["frame_predicted_ids"],
                                              rec["frame_gt_boxes"], rec["frame_gt_ids"])):
        if pi:
            for p, g in frame_matches(pb, pi, gb, gi, 0.5):
                if own.get(p) == g:
                    matched[g].add(fi)
    by_gt = defaultdict(list)
    for p, g in own.items():
        by_gt[g].append(first[(p, g)])
    c = Counter()
    for g, starts in by_gt.items():
        for s0 in sorted(starts)[1:]:
            prev = [f for f in matched[g] if f < s0]
            L = max(prev) if prev else -1
            run = best = 0
            for f in range(L + 1, s0):
                run = run + 1 if f not in vis[g] else 0
                best = max(best, run)
            c["reentry_beyond_buffer" if best >= buffer else ("short_occlusion" if best > 0 else "switch_in_view")] += 1
    return c


GF = {v: len({g for f in json.loads(SRC["botsort"](v, 1).read_text())["videos"][0]["frame_gt_ids"] for g in f})
      for v in VIDEOS}
out = {"G_full": GF}
for trk, src in SRC.items():
    if not all(src(v, d).exists() for v in VIDEOS for d in (1, 2, 4, 8)):
        out[trk] = "incomplete"
        continue
    res = {}
    for d in (1, 2, 4, 8):
        recs = {v: json.loads(src(v, d).read_text())["videos"][0] for v in VIDEOS}
        for v, r in recs.items():
            r.setdefault("video", v)
        for tau in (1, 3):
            agg = defaultdict(Counter)
            per = {}
            for v, r in recs.items():
                t = decompose(r, 0.5, tau)
                a = agg[GROUP(v)]
                a["G"] += GF[v]; a["P"] += t["P"]; a["U"] += t["U"]; a["D"] += t["D"]
                a["M"] += t["M"] + GF[v] - t["G"]; a["videos"] += 1
                if tau == 1:
                    a.update(classify(r, BUFFER))
                per[v] = t["P"]
            grp = {g: dict(x, e=round((x["P"] - x["G"]) / x["G"], 3), U_per_G=round(x["U"] / x["G"], 3),
                           D_per_G=round(x["D"] / x["G"], 3), M_per_G=round(x["M"] / x["G"], 3))
                   for g, x in agg.items()}
            lovo = []
            for v in VINES:
                oth = [u for u in VINES if u != v]
                k = sum(GF[u] for u in oth) / max(sum(per[u] for u in oth), 1e-9)
                lovo.append(abs(k * per[v] / GF[v] - 1))
            vine = dict(ratio={v: round(per[v] / GF[v], 3) for v in VINES},
                        lovo_median=round(st.median(lovo), 3), lovo_max=round(max(lovo), 3),
                        spearman=round(spearmanr([per[v] for v in VINES], [GF[v] for v in VINES]).statistic, 3))
            res[f"d{d}|tau{tau}"] = dict(groups=grp, vines=vine)
    out[trk] = res
dup = {trk: out[trk]["d1|tau1"]["groups"]["PP2-8"]["D_per_G"] for trk in SRC if isinstance(out[trk], dict)}
out["H4_reading_d1_tau1"] = {trk: ("keeps identity (<=0.10)" if x <= 0.10 else
                                   "does not keep identity (>=0.30)" if x >= 0.30 else "partial (0.10-0.30)")
                             for trk, x in dup.items()}
dest = ROOT / "cadence_journal_1119/results/oracle_trackers_summary_1120.json"
dest.write_text(json.dumps(out, indent=1))
for trk in SRC:
    if not isinstance(out[trk], dict):
        print(trk, out[trk]); continue
    for key, x in out[trk].items():
        g = x["groups"]
        print(f"{trk:10s} {key:9s} PP2-8 e {g['PP2-8']['e']:+.3f} U {g['PP2-8']['U_per_G']:.3f} D {g['PP2-8']['D_per_G']:.3f} "
              f"M {g['PP2-8']['M_per_G']:.3f} | NP e {g['NP1-3']['e']:+.3f} | vine lovo {x['vines']['lovo_median']:.3f} "
              f"rho {x['vines']['spearman']:.3f}")
print(out["H4_reading_d1_tau1"])
