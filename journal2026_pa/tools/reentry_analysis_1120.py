#!/usr/bin/env python3
"""Where oracle duplicates come from: re-entry after invisibility vs. switches while visible.

Dataset side (no tracker): for every reference trajectory, its visible segments over the
annotated frames and the gaps between them.
Tracker side (oracle runs, reference boxes as detections): for every duplicate identity
(a trajectory owning m_g > 1 tracks), the number of processed frames the trajectory had
been invisible just before the new identity first matched it: >= track buffer (re-entry
the tracker could not bridge), 1..buffer-1 (short occlusion), 0 (switch while visible).
Ownership: the decomposition's rule (decompose_count_error.py, IoU 0.5, tau 1), rebuilt
here with its own frame_matches(); U, D, M are checked against decompose() for every arm.
    .venv/bin/python cadence_journal_1119/tools/reentry_analysis_1120.py
"""
from __future__ import annotations
import json, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import decompose, frame_matches  # noqa: E402
ORC = ROOT / "runs/oracle_cadence_1119"
ALL11 = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]
GROUP = {v: ("PP2-8" if v not in ("PathPlanning_1",) and v.startswith("Path") else ("PP1" if v == "PathPlanning_1" else "NP1-3")) for v in ALL11}
HZ = {v: (15.0 if GROUP[v] == "PP2-8" else 30.0) for v in ALL11}
BUFFER = 30


def owners(video, threshold=0.5):
    overlap = defaultdict(Counter)
    first_match = {}
    for fi, (pb, pi, gb, gi) in enumerate(zip(video["frame_predicted_boxes"], video["frame_predicted_ids"],
                                              video["frame_gt_boxes"], video["frame_gt_ids"])):
        if not pi:
            continue
        for pred, truth in frame_matches(pb, pi, gb, gi, threshold):
            overlap[pred][truth] += 1
            first_match.setdefault((pred, truth), fi)
    own = {p: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for p, c in overlap.items() if c}
    return own, first_match


res = {"dataset": {}, "oracle": {}}
for v in ALL11:
    rec = json.loads((ORC / "botsort_gmc" / f"{v}_d1_o0.json").read_text())["videos"][0]
    pres = defaultdict(list)
    for fi, ids in enumerate(rec["frame_gt_ids"]):
        for g in ids:
            pres[g].append(fi)
    gaps = {g: [b - a - 1 for a, b in zip(fr, fr[1:]) if b - a > 1] for g, fr in pres.items()}
    res["dataset"][v] = dict(group=GROUP[v], hz=HZ[v], frames=len(rec["frame_gt_ids"]), trajectories=len(pres),
                             reappear_any=sum(bool(x) for x in gaps.values()),
                             reappear_after_ge_2s=sum(any(y / HZ[v] >= 2.0 for y in x) for x in gaps.values()),
                             median_life_s=round(st.median((fr[-1] - fr[0] + 1) / HZ[v] for fr in pres.values()), 2))
for arm in ("botsort_gmc", "bytetrack"):
    for d in (1, 2, 4, 8):
        agg = defaultdict(lambda: Counter())
        for v in ALL11:
            for k in range(d):
                rec = json.loads((ORC / arm / f"{v}_d{d}_o{k}.json").read_text())["videos"][0]
                ref = decompose(rec, 0.5, 1)
                own, fm = owners(rec)
                per_gt = Counter(own.values())
                allp = {t for fr in rec["frame_predicted_ids"] for t in fr}
                U = sum(1 for p in allp if p not in own)
                D = sum(n - 1 for n in per_gt.values())
                assert (U, D) == (ref["U"], ref["D"]), (v, d, k)
                vis = defaultdict(set)
                for fi, ids in enumerate(rec["frame_gt_ids"]):
                    for g in ids:
                        vis[g].add(fi)
                matched = defaultdict(set)          # frames in which g is matched by each owned track
                for fi, (pb, pi, gb, gi) in enumerate(zip(rec["frame_predicted_boxes"], rec["frame_predicted_ids"],
                                                          rec["frame_gt_boxes"], rec["frame_gt_ids"])):
                    if pi:
                        for pred, truth in frame_matches(pb, pi, gb, gi, 0.5):
                            if own.get(pred) == truth:
                                matched[truth].add(fi)
                by_gt = defaultdict(list)
                for p, g in own.items():
                    by_gt[g].append(fm[(p, g)])
                c = agg[GROUP[v]]
                for g, starts in by_gt.items():
                    for s0 in sorted(starts)[1:]:
                        prev = [f for f in matched[g] if f < s0]
                        L = max(prev) if prev else -1
                        run = best = 0
                        for f in range(L + 1, s0):
                            run = run + 1 if f not in vis[g] else 0
                            best = max(best, run)
                        if L < 0:
                            c["no_earlier_match"] += 1
                        elif best >= BUFFER:
                            c["reentry_beyond_buffer"] += 1
                        elif best > 0:
                            c["reentry_within_buffer"] += 1
                        else:
                            c["switch_while_visible"] += 1
                c["D"] += D
                c["arms"] += 1
        res["oracle"][f"{arm}|d{d}"] = {g: dict(x) for g, x in agg.items()}
print(json.dumps(res, indent=1))
(ROOT / "cadence_journal_1119/results/reentry_analysis_1120.json").write_text(json.dumps(res, indent=1))
