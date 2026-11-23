#!/usr/bin/env python3
"""Summarise the external checks on APPLE MOTS and LettuceMOTS (ledger 10.28.16).

Oracle (tools/oracle_external_1122.py): per dataset, sequence group, associator and interval,
the pooled count error e = (P - G)/G with its terms U, D, M (decompose_count_error.decompose,
IoU 0.5, tau 1, the GrapeMOTS rule), and, for every duplicate identity, where it began, with the
classification of cadence_journal_1119/tools/reentry_analysis_1120.py (buffer 30 processed frames):
switch while visible / re-entry within the buffer / re-entry beyond the buffer.
LettuceMOTS groups are defined from the reference alone: "return" = sequences in which at least
one plant reappears after more than 10 annotated frames out of view (the robot passes the plants
again), "single" = the others. APPLE MOTS sequences are single passes.

Substitution (tools/error_sources_external_1122.py): per inference scale and interval, pooled over
the three test sequences, e with no fix, with the miss fix alone and with all fixes, and the
pooled Shapley shares of each fix in e (sum of per-sequence shares of P over the summed G, as
cadence_journal_1119/tools/error_sources_summary.py pools them).

    .venv/bin/python tools/external_summary_1122.py      # on raptor, writes runs/external_1122/summary.json
    python3 external_summary_1122.py --root WORKSPACE --only apple --out FILE   # archive smoke test
(LettuceMOTS per-frame outputs carry boxes derived from its masks and are not redistributed, so the archive
rebuilds the APPLE MOTS parts only.)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, default=Path("/home/kou/my_env/yolo26"))
ap.add_argument("--only", nargs="*", default=["apple", "lettuce"])
ap.add_argument("--out", type=Path, default=None)
A = ap.parse_args()
ROOT = A.root.resolve()
for tools in (ROOT / "tools", ROOT / "grapemots-protocol/cadence2026_0919/tools"):
    if (tools / "decompose_count_error.py").is_file():
        sys.path.insert(0, str(tools))
        break
from decompose_count_error import decompose, frame_matches  # noqa: E402

EXT = ROOT / "runs/external_1122"
BUFFER = 30
TRACKERS = ("botsort_gmc", "bytetrack", "strongsort", "ocsort", "deepocsort")
DELTAS = (1, 2, 4, 8)
FACTORS = ("bkg", "loc", "jit", "miss", "score")


def load(ds, trk, video, d):
    name = f"{video}_d{d}.json" if trk in ("botsort_gmc", "bytetrack") else f"{video}_s0_d{d}.json"
    p = EXT / ds / trk / name
    return json.loads(p.read_text())["videos"][0] if p.exists() else None


def reference_structure(ds):
    out = {}
    for dp in sorted((EXT / ds / "dets").glob("*_s0.json")):
        d = json.loads(dp.read_text())
        pres = defaultdict(list)
        for fi, f in enumerate(d["frames"]):
            for g in f["gt_ids"]:
                pres[g].append(fi)
        gaps = {g: max([b - a - 1 for a, b in zip(fr, fr[1:])] or [0]) for g, fr in pres.items()}
        out[d["video"]] = dict(frames=len(d["frames"]), G=len(pres),
                               reappear_after_gt10=sum(x > 10 for x in gaps.values()),
                               absent_ge30=sum(x >= BUFFER for x in gaps.values()),
                               reappear_any=sum(x > 0 for x in gaps.values()))
    return out


def classify(rec):
    """Duplicate origins, reentry_analysis_1120.py's rule."""
    overlap, first = defaultdict(Counter), {}
    for fi, (pb, pi, gb, gi) in enumerate(zip(rec["frame_predicted_boxes"], rec["frame_predicted_ids"],
                                              rec["frame_gt_boxes"], rec["frame_gt_ids"])):
        if pi:
            for pred, truth in frame_matches(pb, pi, gb, gi, 0.5):
                overlap[pred][truth] += 1
                first.setdefault((pred, truth), fi)
    own = {p: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for p, c in overlap.items() if c}
    vis, matched = defaultdict(set), defaultdict(set)
    for fi, ids in enumerate(rec["frame_gt_ids"]):
        for g in ids:
            vis[g].add(fi)
    for fi, (pb, pi, gb, gi) in enumerate(zip(rec["frame_predicted_boxes"], rec["frame_predicted_ids"],
                                              rec["frame_gt_boxes"], rec["frame_gt_ids"])):
        if pi:
            for pred, truth in frame_matches(pb, pi, gb, gi, 0.5):
                if own.get(pred) == truth:
                    matched[truth].add(fi)
    by_gt = defaultdict(list)
    for p, g in own.items():
        by_gt[g].append(first[(p, g)])
    c = Counter()
    for g, starts in by_gt.items():
        for s0 in sorted(starts)[1:]:
            prev = [f for f in matched[g] if f < s0]
            last = max(prev) if prev else -1
            run = best = 0
            for f in range(last + 1, s0):
                run = run + 1 if f not in vis[g] else 0
                best = max(best, run)
            if last < 0:
                c["no_earlier_match"] += 1
            elif best >= BUFFER:
                c["reentry_beyond_buffer"] += 1
            elif best > 0:
                c["reentry_within_buffer"] += 1
            else:
                c["switch_while_visible"] += 1
    return c


def oracle(ds, groups):
    res = {}
    for trk in TRACKERS:
        for d in DELTAS:
            for gname, vids in groups.items():
                tot, cls, n = Counter(), Counter(), 0
                for v in vids:
                    rec = load(ds, trk, v, d)
                    if rec is None:
                        continue
                    rec["video"] = v
                    t = decompose(rec, 0.5, 1)
                    for k in ("P", "G", "U", "D", "M"):
                        tot[k] += t[k]
                    if d == 1:
                        cls.update(classify(rec))
                    n += 1
                if n == 0:
                    continue
                row = dict(sequences=n, **tot, e=round((tot["P"] - tot["G"]) / tot["G"], 4))
                if d == 1:
                    row["duplicate_origin"] = dict(cls)
                res[f"{trk}|d{d}|{gname}"] = row
    return res


def substitution():
    res = {}
    files = sorted((EXT / "apple/substitution").glob("*_s*.json"))
    files = [p for p in files if not p.stem.endswith(("_none", "_all"))]
    by_scale = defaultdict(list)
    for p in files:
        by_scale[p.stem.split("_s")[-1]].append(json.loads(p.read_text()))
    for s, runs in sorted(by_scale.items(), key=lambda kv: int(kv[0])):
        for d in DELTAS:
            G = sum(r["G_full"] for r in runs)
            if not all(str(d) in r["arms"] for r in runs):
                continue
            tp = sum(r["detection_stats"]["tp"] for r in runs)
            ref = sum(r["detection_stats"]["reference_boxes"] for r in runs)
            row = dict(sequences=len(runs), G=G, tp=tp, reference_boxes=ref, recall_iou50=round(tp / ref, 4))
            for cond in ("none", "miss", "miss+score", "bkg+jit+loc+miss+score"):
                P = sum(r["arms"][str(d)]["values"][cond]["P"] for r in runs)
                row[f"e_{cond}"] = round((P - G) / G, 4)
                row[f"P_{cond}"] = P
                for k in ("U", "D", "M"):
                    row[f"{k}_{cond}"] = sum(r["arms"][str(d)]["values"][cond][k] for r in runs)
            row["abs_e_miss_minus_none"] = round(abs(row["e_miss"]) - abs(row["e_none"]), 4)
            row["shapley_e"] = {f: round(sum(r["arms"][str(d)]["shapley"]["P"][f] for r in runs) / G, 4)
                                for f in FACTORS}
            row["per_sequence"] = {r["video"]: dict(e_none=round(r["arms"][str(d)]["values"]["none"]["e"], 4),
                                                    e_miss=round(r["arms"][str(d)]["values"]["miss"]["e"], 4))
                                   for r in runs}
            res[f"s{s}|d{d}"] = row
    return res


def main():
    out = {"reference": {}, "oracle": {}}
    for ds in [d for d in ("apple", "lettuce") if d in A.only]:
        ref = reference_structure(ds)
        out["reference"][ds] = ref
        if ds == "apple":
            groups = {"all": sorted(ref)}
        else:
            groups = {"return": sorted(v for v, x in ref.items() if x["reappear_after_gt10"] > 0),
                      "single": sorted(v for v, x in ref.items() if x["reappear_after_gt10"] == 0)}
        out["groups_" + ds] = groups
        out["oracle"][ds] = oracle(ds, groups)
    if "apple" in A.only:
        out["substitution_apple"] = substitution()
    (A.out or EXT / "summary.json").write_text(json.dumps(out, indent=1))
    for ds in [d for d in ("apple", "lettuce") if d in A.only]:
        print(f"== oracle {ds}")
        for k, r in out["oracle"][ds].items():
            print(f"  {k:28s} n={r['sequences']:2d} G={r['G']:5d} P={r['P']:5d} U={r['U']:4d} D={r['D']:4d} "
                  f"M={r['M']:4d} e={r['e']:+.3f} {r.get('duplicate_origin', '')}")
    print("== substitution apple")
    for k, r in out.get("substitution_apple", {}).items():
        print(f"  {k:10s} recall={r['recall_iou50']:.3f} e none={r['e_none']:+.3f} miss={r['e_miss']:+.3f} "
              f"|e| change={r['abs_e_miss_minus_none']:+.3f} all={r['e_bkg+jit+loc+miss+score']:+.3f} "
              f"shapley={r['shapley_e']}")


if __name__ == "__main__":
    main()
