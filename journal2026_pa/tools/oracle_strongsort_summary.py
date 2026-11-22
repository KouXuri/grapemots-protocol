#!/usr/bin/env python3
"""H3: does appearance association remove the duplicates of perfect detections?

Compares, at phase 0, the oracle runs of BoT-SORT (runs/oracle_cadence_1119/botsort_gmc) and
StrongSORT (runs/error_sources_1120/oracle_strongsort/tracks, reference boxes + OSNet-x0.25):
U, D, M per reference trajectory and the duplicate classes of reentry_analysis_1120.py
(same owner rule, same classification code, imported from it).
Pre-registered reading (ledger 10.28.5): multi-view D/G >= 0.30 with StrongSORT -> 2-D
appearance does not keep identity through a change of viewpoint; <= 0.10 -> it does.
    .venv/bin/python cadence_journal_1119/tools/oracle_strongsort_summary.py
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import decompose, frame_matches  # noqa: E402

GROUP = lambda v: "NP1-3" if v.startswith("No") else ("PP1" if v == "PathPlanning_1" else "PP2-8")
BUFFER = {"botsort": 30, "strongsort": 30}      # BoxMOT StrongSORT max_age default is 30 as well
SRC = {"botsort": lambda v, d: ROOT / f"runs/oracle_cadence_1119/botsort_gmc/{v}_d{d}_o0.json",
       "strongsort": lambda v, d: ROOT / f"runs/error_sources_1120/oracle_strongsort/tracks/{v}_s0_d{d}.json"}
VIDEOS = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]


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


out = {}
for trk in ("botsort", "strongsort"):
    for d in (1, 2, 4, 8):
        agg = defaultdict(Counter)
        for v in VIDEOS:
            p = SRC[trk](v, d)
            if not p.exists():
                continue
            rec = json.loads(p.read_text())["videos"][0]
            rec.setdefault("video", v)
            t = decompose(rec, 0.5, 1)
            G_full = len({g for f in json.loads(SRC["botsort"](v, 1).read_text())["videos"][0]["frame_gt_ids"] for g in f})
            a = agg[GROUP(v)]
            a["G"] += G_full; a["P"] += t["P"]; a["U"] += t["U"]; a["D"] += t["D"]; a["M"] += t["M"] + G_full - t["G"]
            a["videos"] += 1
            a.update(classify(rec, BUFFER[trk]))
        out[f"{trk}|d{d}"] = {g: dict(x, e=round((x["P"] - x["G"]) / x["G"], 3), D_per_G=round(x["D"] / x["G"], 3),
                                      M_per_G=round(x["M"] / x["G"], 3)) for g, x in agg.items() if x["G"]}
(ROOT / "cadence_journal_1119/results/oracle_strongsort_summary.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, {g: {kk: x[kk] for kk in ("videos", "e", "D_per_G", "M_per_G", "switch_in_view", "reentry_beyond_buffer", "short_occlusion") if kk in x} for g, x in v.items()})
