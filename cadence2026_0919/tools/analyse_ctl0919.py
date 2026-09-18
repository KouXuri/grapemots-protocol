#!/usr/bin/env python3
"""Read the out-of-fold scale-cadence surface and its three controls.

Inputs (all frozen JSON written by tools/track_grapemots_mot.py):
  runs/lovo_surface_0918/results/{video}_s{sigma}_d{delta}.json   phase 0, buffer 30
  runs/ctl_0919/phase/{video}_s{sigma}_d{delta}_o{k}.json         phases k = 1..delta-1
  runs/ctl_0919/{ret,cu,rt}/{video}_s{sigma}_d{delta}.json        retention / clock arms

Count terms come from tools/decompose_count_error.decompose (tau = 1, IoU 0.5).
The denominator is G_full, the number of reference trajectories in ALL annotated
frames of the sequence (the Delta = 1 phase-0 reference), not the trajectories
visible in the processed frames: a trajectory the sampling never shows is a
missed bunch and is added to M. The processed-frame G is kept as G_proc.

Crossing bracket per (sequence, sigma): '<1' if the error is already negative at
Delta = 1, '(a,b)' for the first sign change between measured intervals, '>8' if
still positive at Delta = 8. No interpolation.

Writes runs/ctl_0919/ctl_summary.json and prints the tables the paper uses.
"""
from __future__ import annotations
import json, os, sys, statistics
sys.path.insert(0, "tools")
from decompose_count_error import decompose

BASE = "runs/lovo_surface_0918/results"
CTL = "runs/ctl_0919"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
_cache: dict[str, dict | None] = {}


def terms(path: str) -> dict | None:
    if path in _cache:
        return _cache[path]
    if not os.path.exists(path):
        _cache[path] = None
        return None
    v = json.load(open(path))["videos"][0]
    r = decompose({k: v[k] for k in ("video", "frame_predicted_ids", "frame_predicted_boxes",
                                     "frame_gt_ids", "frame_gt_boxes")}, 0.5, 1)
    _cache[path] = {k: r[k] for k in ("P", "G", "U", "D", "M")}
    return _cache[path]


def base(v, s, d):
    return terms(f"{BASE}/{v}_s{s}_d{d}.json")


GFULL = {v: base(v, 3072, 1)["G"] for v in VIDEOS}


def err(t, v):
    return None if t is None else (t["P"] - GFULL[v]) / GFULL[v]


def cov(t, v):
    return None if t is None else 1 - (t["M"] + GFULL[v] - t["G"]) / GFULL[v]


def bracket(es):
    if any(x is None for x in es):
        return None
    if es[0] < 0:
        return "<1"
    for i in range(3):
        if es[i] >= 0 and es[i + 1] < 0:
            return f"({DELTAS[i]},{DELTAS[i + 1]})"
    return ">8"


def surface(getter):
    """{(v, s): [e at Delta 1,2,4,8]} with Delta = 1 always the base arm."""
    out = {}
    for v in VIDEOS:
        for s in SIGMAS:
            out[(v, s)] = [err(base(v, s, 1), v)] + [err(getter(v, s, d), v) for d in DELTAS[1:]]
    return out


def summarise(name, surf, ref_map=None):
    complete = [k for k, es in surf.items() if all(x is not None for x in es)]
    steps = viol = 0
    for k in complete:
        es = surf[k]
        for a, b in zip(es, es[1:]):
            steps += 1
            viol += b > a
    cmap = {k: bracket(surf[k]) for k in complete}
    crossing_rows = sum(1 for b in cmap.values() if b not in ("<1", ">8"))
    mono_seq = 0; seq_complete = 0
    for v in VIDEOS:
        row = [cmap.get((v, s)) for s in SIGMAS]
        if None in row:
            continue
        seq_complete += 1
        mono_seq += all(ORDER[a] <= ORDER[b] for a, b in zip(row, row[1:]))
    moved = later = earlier = 0
    if ref_map:
        for k, b in cmap.items():
            if ref_map.get(k) and b != ref_map[k]:
                moved += 1
                later += ORDER[b] > ORDER[ref_map[k]]
                earlier += ORDER[b] < ORDER[ref_map[k]]
    res = dict(arm=name, rows_complete=len(complete), steps=steps, monotone_violations=viol,
               crossing_rows=crossing_rows, sequences_complete=seq_complete,
               sequences_with_ordered_crossing=mono_seq, cells_moved=moved,
               moved_later=later, moved_earlier=earlier,
               crossing_map={f"{v}|{s}": b for (v, s), b in cmap.items()})
    print(f"{name:10s} rows {len(complete):2d}/50  monotone viol {viol}/{steps}  sign-change rows {crossing_rows}"
          f"  ordered seqs {mono_seq}/{seq_complete}  moved {moved} (later {later}, earlier {earlier})")
    return res


def main():
    summary = {"G_full": GFULL}
    b = surface(lambda v, s, d: base(v, s, d))
    rb = summarise("base", b)
    ref = {tuple([k.split("|")[0], int(k.split("|")[1])]): x for k, x in rb["crossing_map"].items()}
    summary["base"] = rb
    for arm in ("ret", "cu", "rt"):
        surf = surface(lambda v, s, d, arm=arm: terms(f"{CTL}/{arm}/{v}_s{s}_d{d}.json"))
        summary[arm] = summarise(arm, surf, ref)
    # Phase: every (v, s, d) read at all d phases; phase 0 is the base arm.
    ph = {}
    for v in VIDEOS:
        for s in SIGMAS:
            for d in DELTAS[1:]:
                ts = [base(v, s, d)] + [terms(f"{CTL}/phase/{v}_s{s}_d{d}_o{k}.json") for k in range(1, d)]
                if any(t is None for t in ts):
                    continue
                es = [err(t, v) for t in ts]
                ph[(v, s, d)] = dict(e_min=min(es), e_max=max(es), e_phase0=es[0],
                                     G_proc_min=min(t["G"] for t in ts),
                                     sign_stable=(min(es) >= 0) == (max(es) >= 0),
                                     cov_max=max(cov(t, v) for t in ts))
    if ph:
        spans = [x["e_max"] - x["e_min"] for x in ph.values()]
        flips = [k for k, x in ph.items() if not x["sign_stable"]]
        print(f"phase      cells {len(ph)}/150  median span {statistics.median(spans):.3f}  max span {max(spans):.3f}"
              f"  sign flips across phase {len(flips)}")
        # crossing map under the most and least favourable phase per cell
        lo = surface(lambda v, s, d: None)
        hi = surface(lambda v, s, d: None)
        for (v, s) in lo:
            for i, d in enumerate(DELTAS[1:], start=1):
                x = ph.get((v, s, d))
                lo[(v, s)][i] = None if x is None else x["e_min"]
                hi[(v, s)][i] = None if x is None else x["e_max"]
        summary["phase_low"] = summarise("phase-low", lo, ref)
        summary["phase_high"] = summarise("phase-high", hi, ref)
        summary["phase_cells"] = {f"{v}|{s}|{d}": x for (v, s, d), x in ph.items()}
        summary["phase_flips"] = [f"{v}|{s}|{d}" for (v, s, d) in flips]
    # coverage at every observed sign change, base arm
    cs = []
    for (v, s), es in b.items():
        br = bracket(es)
        if br and br.startswith("("):
            a, c = (int(x) for x in br[1:-1].split(","))
            cs += [cov(base(v, s, a), v), cov(base(v, s, c), v)]
    summary["coverage_at_sign_change"] = dict(max=max(cs), median=statistics.median(cs), n=len(cs))
    print(f"coverage at the bracketing intervals: max {max(cs):.3f} median {statistics.median(cs):.3f} (n={len(cs)})")
    json.dump(summary, open(f"{CTL}/ctl_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
