"""Checks asked for by the Track 3 review of 1 October 2026, from archived files only.

Run from the archive root:
    python3 cadence2026_1001/tools/review_checks.py LOVO_RAW_DIR OUT.json

LOVO_RAW_DIR is cadence2026_0919/raw/lovo_surface.tar.gz extracted (the GrapeMOTS
replication's per-frame outputs). Everything else is read from the archive.

1. Ownership versus contact. Coverage in the paper is 1 - M/G: the share of
   reference trajectories that own at least one predicted track, a track being
   owned by the trajectory it covers in most scored frames. A trajectory can be
   matched by a track that another trajectory owns. Here the share matched at
   least once (by any eligible track, IoU >= 0.5, same one-to-one matching) is
   reported beside it, with the number of tracks matched to more than one
   trajectory. Same eligibility rule and matcher as decompose_count_error.py.
2. The GrapeMOTS replication summarised per video, so the 150 comparisons are
   read as repeated settings on ten videos, not as 150 independent units.
3. The 2021 intervention with one flight left out at a time (pooled e per arm,
   rises, median paired difference).
"""
import json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cadence2026_0919/tools"), str(ROOT / "cadence2026_0813/tools")]
from decompose_count_error import frame_matches  # noqa: E402
from definition_sensitivity_0815 import per_video  # noqa: E402

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
J = lambda p: json.loads(Path(p).read_text())


def contact(rec, thr=0.5, tau=1):
    """Ownership coverage, contact coverage and multi-trajectory tracks for one arm."""
    life = Counter(t for f in rec["frame_predicted_ids"] for t in f)
    kept = {t for t, n in life.items() if n >= tau}          # eligibility first
    touched, by_track = set(), defaultdict(set)
    overlap = defaultdict(Counter)
    for pb, pi, gb, gi in zip(rec["frame_predicted_boxes"], rec["frame_predicted_ids"],
                              rec["frame_gt_boxes"], rec["frame_gt_ids"]):
        live = [(b, t) for b, t in zip(pb, pi) if t in kept]
        if not live:
            continue
        # one-to-one by maximum total IoU, then pairs below thr dropped
        for p, g in frame_matches([b for b, _ in live], [t for _, t in live], gb, gi, thr):
            touched.add(g); by_track[p].add(g); overlap[p][g] += 1
    G = len({t for f in rec["frame_gt_ids"] for t in f})
    owners = set()
    for p, c in overlap.items():
        best = max(c.values())
        owners.add(min(g for g, n in c.items() if n == best))
    return {"G": G, "owned": len(owners), "touched": len(touched),
            "multi_trajectory_tracks": sum(1 for s in by_track.values() if len(s) > 1),
            "matched_tracks": len(by_track), "P": len(kept)}


def pool(rows):
    t = Counter()
    for r in rows:
        t.update(r)
    return {"G": t["G"], "P": t["P"], "ownership_coverage": round(t["owned"] / t["G"], 4),
            "contact_coverage": round(t["touched"] / t["G"], 4),
            "multi_trajectory_tracks": t["multi_trajectory_tracks"], "matched_tracks": t["matched_tracks"]}


out = {}
# ---- 2021, 17 held-out sequences, both arms (the frozen arms of Table I)
runs = []
for f in ("arms_fold1_six.json", "arms_fold2_eleven.json"):
    runs += [r for r in J(ROOT / "cadence2026_0813/results/adaptive_0813" / f)["runs"] if r["arm"] in ("rel", "src")]
c21 = {a: pool([contact(r) for r in runs if r["arm"] == a]) for a in ("rel", "src")}
for a, r in zip(("rel", "src"), (c21["rel"], c21["src"])):
    own = sum(per_video(x, 0.5, 1)["G"] - per_video(x, 0.5, 1)["M"] for x in runs if x["arm"] == a)
    assert round(own / r["G"], 4) == r["ownership_coverage"], "ownership disagrees with per_video"
out["2021_held_out"] = c21

# ---- GrapeMOTS: arms within +-0.10 of zero error, scored as in gm_matched.py
def video(name, s, d):
    v = J(RAW / f"{name}_s{s}_d{d}.json")["videos"]
    assert len(v) == 1
    return v[0]

def subset(v, k):
    o = dict(v)
    for key in ("frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes"):
        o[key] = v[key][0::k]
    return o

GM = J(ROOT / "cadence2026_1001/results/gm_matched.json")
near = []
for c in GM["cells"]:
    for arm, rec in (("dense", lambda c: subset(video(c["video"], c["sigma"], 1), c["k"])),
                     ("sparse", lambda c: video(c["video"], c["sigma"], c["k"]))):
        if abs((c[arm]["P"] - c["G"]) / c["G"]) <= 0.10:
            x = contact(rec(c))
            assert x["G"] == c["G"] and c["G"] - x["owned"] == c[arm]["M"], (c["video"], arm)
            near.append({"video": c["video"], "sigma": c["sigma"], "k": c["k"], "arm": arm,
                         "ownership": round(x["owned"] / x["G"], 4), "contact": round(x["touched"] / x["G"], 4)})
own = [n["ownership"] for n in near]; con = [n["contact"] for n in near]
out["grapemots_near_zero"] = {"arms": len(near), "ownership_median": statistics.median(own),
                              "ownership_max": max(own), "contact_median": statistics.median(con),
                              "contact_max": max(con), "rows": near}

# ---- GrapeMOTS per video
pv = {}
for v in GM["videos"]:
    rows = [c for c in GM["cells"] if c["video"] == v]
    d = [c["e_dense"] - c["e_sparse"] for c in rows]
    pv[v] = {"n": len(rows), "rises": sum(c["dense"]["P"] > c["sparse"]["P"] for c in rows),
             "opposite": sum(c["e_dense"] > 0 > c["e_sparse"] for c in rows),
             "median_delta_e": round(statistics.median(d), 3), "min_delta_e": round(min(d), 3),
             "max_delta_e": round(max(d), 3)}
out["grapemots_per_video"] = pv

# ---- 2021 leave one flight out (flight = vineyard row, as cluster_bootstrap.json)
per = {}
for r in runs:
    v = per_video(r, 0.5, 1)
    per.setdefault(r["video"], {})[r["arm"]] = v
flight = lambda s: "row " + s.split("_")[1].split(".")[0]
lofo = {}
for f in sorted({flight(s) for s in per}):
    keep = [s for s in per if flight(s) != f]
    P = {a: sum(per[s][a]["P"] for s in keep) for a in ("rel", "src")}
    G = sum(per[s]["rel"]["G"] for s in keep)
    d = [(per[s]["src"]["P"] - per[s]["rel"]["P"]) / per[s]["rel"]["G"] for s in keep]
    lofo[f] = {"sequences_left": len(keep), "pooled_e_sparse": round((P["rel"] - G) / G, 3),
               "pooled_e_source": round((P["src"] - G) / G, 3),
               "rises": sum(x > 0 for x in d), "median_delta": round(statistics.median(d), 3)}
out["2021_leave_one_flight_out"] = lofo

OUT.write_text(json.dumps(out, indent=1) + "\n")
print("2021 held-out:", json.dumps(c21))
g = out["grapemots_near_zero"]
print(f"GrapeMOTS near-zero arms {g['arms']}: ownership median {g['ownership_median']:.3f} max {g['ownership_max']:.3f}; "
      f"contact median {g['contact_median']:.3f} max {g['contact_max']:.3f}")
for v, x in pv.items():
    print(f"  {v:17} rises {x['rises']}/{x['n']} opp {x['opposite']} median de {x['median_delta_e']:+.3f} [{x['min_delta_e']:+.3f},{x['max_delta_e']:+.3f}]")
for f, x in lofo.items():
    print("  without", f, x)
