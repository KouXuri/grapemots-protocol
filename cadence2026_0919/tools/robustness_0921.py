#!/usr/bin/env python3
"""Recompute the robustness table of the camera-ready paper from the frozen stubs.

Reads raw/{lovo_surface,phase,ret,cu,rt,bytetrack,strongsort,retrain,apple_val0000,timing}.tar.gz
and rebuilds, with the paper's conventions (tau = 1, matching IoU 0.5, e computed
against the reference trajectories over all annotated frames):

  * one row per arm of Table I: steps that lowered the error (ties in brackets),
    sign-change rows and sequences, sequences whose sign change never moves to a
    denser interval, and how many of the 50 brackets differ from the published map,
    split into moves to a sparser and to a denser interval;
  * coverage at the arms bracketing each sign change, (G - M) / G_full, with the
    exact numerator, denominator and arm of the maximum. Phase rows take the terms
    of the phase arm they pick (the lowest phase among ties); the phase mean
    averages them;
  * the most coverage an arm reaches at a sign change under any combination of phases;
  * the same coverage with ownership read at IoU 0.3 and 0.2 (the error is unchanged),
    and over the 1,200 arms of the surface and its three controls, the arms whose
    error lies within +-0.1 and the most coverage any of them reached;
  * the two mask-cutoff rows, from results/cutoff_sensitivity.json: the per-arm
    terms of the frozen predictions scored against references rebuilt from the
    release's masks (tools/cutoff_sensitivity_0920.py, which needs the imagery);
  * the seed spread of the retrained detectors and how many cells change sign with
    the seed;
  * the AppleMOT grid read with the detector validated on sequence 0000, with the
    coverage of each arm and the grid scored on one of each identically annotated pair;
  * the held-out cost frontiers: arms non-dominated in HOTA against cost per annotated
    frame, for processed-frame HOTA (results/hota_timing_session.json) and for HOTA on
    the common annotated timeline (results/common_timeline_hota.json).

    python3 tools/robustness_0921.py            # recompute and print
    python3 tools/robustness_0921.py --check    # and compare with results/expected_robustness.json

Requires numpy and scipy (scipy for the one-to-one matching). No imagery or GPU.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

MV = [f"PathPlanning_{i}" for i in range(2, 9)]
FR = [f"NoPathPlanning_{i}" for i in range(1, 4)]
V = MV + FR
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
SHORT = {**{f"PathPlanning_{i}": f"PP{i}" for i in range(2, 9)},
         **{f"NoPathPlanning_{i}": f"NP{i}" for i in range(1, 4)}}
APPLE_UNIQUE3 = ("0006", "0007", "0008")  # 0010-0012 repeat their annotations


def members(family: str):
    with tarfile.open(ROOT / "raw" / f"{family}.tar.gz") as tar:
        for m in tar.getmembers():
            yield Path(m.name).name, json.load(io.TextIOWrapper(tar.extractfile(m)))


def load(family: str, per_video: bool = False, iou: float = 0.5) -> dict[str, dict]:
    out = {}
    for name, stub in members(family):
        terms = [dict(decompose(v, iou, 1), video=v["video"]) for v in stub["videos"]]
        out[name] = {k: sum(t[k] for t in terms) for k in ("P", "G", "U", "D", "M")}
        if per_video:
            out[name]["videos"] = terms
    return out


def bracket(es):
    if es[0] < 0:
        return "<1"
    for i in range(3):
        if es[i] >= 0 > es[i + 1]:
            return f"({DELTAS[i]},{DELTAS[i + 1]})"
    return ">8"


def row(get, GF, base_map=None):
    """get(v, sigma, delta) -> terms with P, G, M (and 'arm', a label)."""
    strict = ties = rev = 0
    cmap, cov = {}, []
    for v in V:
        for s in SIGMAS:
            t = {d: get(v, s, d) for d in DELTAS}
            es = [(t[d]["P"] - GF[v]) / GF[v] for d in DELTAS]
            for a, b in zip(es, es[1:]):
                strict += b < a; ties += b == a; rev += b > a
            cmap[(v, s)] = bracket(es)
            if cmap[(v, s)].startswith("("):
                lo, hi = (int(x) for x in cmap[(v, s)][1:-1].split(","))
                for d in (lo, hi):
                    reached = t[d]["G"] - t[d]["M"]
                    label = t[d].get("arm", f"s{s} d{d}")
                    cov.append((reached / GF[v], reached, GF[v], f"{SHORT[v]} {label}"))
    ordered = sum(all(ORDER[cmap[(v, a)]] <= ORDER[cmap[(v, b)]] for a, b in zip(SIGMAS, SIGMAS[1:])) for v in V)
    top = max(cov)
    num = round(top[1], 2) if isinstance(top[1], float) else top[1]
    out = dict(steps_down=strict, ties=ties, reversals=rev,
               sign_change_rows=sum(1 for x in cmap.values() if x.startswith("(")),
               sign_change_sequences=len({v for (v, s), x in cmap.items() if x.startswith("(")}),
               ordered_sequences=ordered,
               coverage_max=round(top[0], 4), coverage_max_at=f"{num}/{top[2]} {top[3]}",
               coverage_median=round(statistics.median(c[0] for c in cov), 3))
    if base_map is not None:
        out["brackets_differing"] = sum(cmap[k] != base_map[k] for k in cmap)
        out["brackets_sparser"] = sum(ORDER[cmap[k]] > ORDER[base_map[k]] for k in cmap)
        out["brackets_denser"] = sum(ORDER[cmap[k]] < ORDER[base_map[k]] for k in cmap)
    return out, cmap


def frontier(points: dict[str, tuple[float, float]]) -> list[str]:
    """Arms no other arm matches or beats on both cost (lower) and HOTA (higher)."""
    return sorted((k for k, (c, h) in points.items()
                   if not any(c2 <= c and h2 >= h and (c2 < c or h2 > h)
                              for k2, (c2, h2) in points.items() if k2 != k)),
                  key=lambda k: points[k][0])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not HAVE_SCIPY:
        sys.exit("scipy is required for the one-to-one matching the paper uses")

    base = load("lovo_surface")
    GF = {v: base[f"{v}_s3072_d1.json"]["G"] for v in V}
    B = lambda v, s, d: dict(base[f"{v}_s{s}_d{d}.json"], arm=f"s{s} d{d}")
    N = {}
    N["surface"], base_map = row(B, GF)
    controls = {}
    for fam in ("ret", "cu", "rt"):
        data = controls[fam] = load(fam)
        N[fam], _ = row(lambda v, s, d, data=data, fam=fam:
                        dict(data[f"{v}_s{s}_d{d}.json"], arm=f"{fam} s{s} d{d}") if d > 1 else B(v, s, d),
                        GF, base_map)

    phase = load("phase")
    def arms(v, s, d):
        return [B(v, s, d)] + [dict(phase[f"{v}_s{s}_d{d}_o{k}.json"], arm=f"s{s} d{d} phase {k}") for k in range(1, d)]
    for tag, pick in (("phase_low", min), ("phase_high", max)):
        def get(v, s, d, pick=pick):
            if d == 1:
                return B(v, s, d)
            a = arms(v, s, d)
            target = pick(x["P"] for x in a)
            return next(x for x in a if x["P"] == target)
        N[tag], _ = row(get, GF, base_map)
    N["phase_mean"], _ = row(lambda v, s, d: B(v, s, d) if d == 1 else dict(
        {k: statistics.mean(x[k] for x in arms(v, s, d)) for k in ("P", "G", "M")}, arm=f"s{s} d{d} phase mean"),
        GF, base_map)

    # any combination of phases: every pair of arms at adjacent intervals, one at or above
    # zero and the next below it, whatever their phases (a superset of every bracket
    # a phase combination can produce)
    sup = max(((x["G"] - x["M"]) / GF[v], x["G"] - x["M"], GF[v], f"{SHORT[v]} {x['arm']}")
              for v in V for s in SIGMAS for i in range(3)
              for hi in arms(v, s, DELTAS[i]) if hi["P"] >= GF[v]
              for lo in arms(v, s, DELTAS[i + 1]) if lo["P"] < GF[v]
              for x in (hi, lo))
    N["phase_any_combination"] = dict(coverage_max=round(sup[0], 4), coverage_max_at=f"{sup[1]}/{sup[2]} {sup[3]}")

    for iou in (0.3, 0.2):
        own = load("lovo_surface", iou=iou)
        N[f"surface_ownership_iou{iou}"], m = row(lambda v, s, d, own=own: dict(own[f"{v}_s{s}_d{d}.json"], arm=f"s{s} d{d}"), GF)
        assert m == base_map  # P does not depend on ownership, so neither does the map
    pool = [(v, B(v, s, d)) for v in V for s in SIGMAS for d in DELTAS]
    pool += [(v, x) for v in V for s in SIGMAS for d in DELTAS[1:] for x in arms(v, s, d)[1:]]
    pool += [(v, dict(ctl[f"{v}_s{s}_d{d}.json"], arm=f"{fam} s{s} d{d}"))
             for fam, ctl in controls.items() for v in V for s in SIGMAS for d in DELTAS[1:]]
    near = [((x["G"] - x["M"]) / GF[v], x["G"] - x["M"], GF[v], f"{SHORT[v]} {x['arm']}")
            for v, x in pool if abs((x["P"] - GF[v]) / GF[v]) <= 0.1]
    top = max(near)
    N["near_zero_arms"] = dict(arms=len(pool), within_0p1=len(near), coverage_max=round(top[0], 4),
                               coverage_max_at=f"{top[1]}/{top[2]} {top[3]}",
                               coverage_median=round(statistics.median(c[0] for c in near), 3))

    cut = json.loads((ROOT / "results" / "cutoff_sensitivity.json").read_text())
    for c in (1, 20, 100, 400):
        GFc = {v: cut[v][f"{c}|3072|1"]["G_full"] for v in V}
        N[f"cutoff_{c}px"], _ = row(lambda v, s, d, c=c: dict(cut[v][f"{c}|{s}|{d}"], arm=f"s{s} d{d}"), GFc, base_map)
        N[f"cutoff_{c}px"]["reference_trajectories"] = sum(GFc.values())

    for fam in ("bytetrack", "strongsort"):
        data = load(fam)
        N[fam], _ = row(lambda v, s, d, data=data, fam=fam: dict(data[f"{v}_s{s}_d{d}.json"], arm=f"{fam} s{s} d{d}"), GF)

    retr = load("retrain")
    def rp(v, s, d, seed):
        key = f"{v}_s{seed}__{v}_s{s}_d{d}.json"
        if key in retr:
            return dict(retr[key], arm=f"seed {seed} s{s} d{d}")
        # by design the frontal folds keep their published seed-0 checkpoint
        # (tools/build_rev_lovo_0920.py); any other missing arm is an error
        if v in FR and seed == 0:
            return B(v, s, d)
        raise KeyError(key)
    for seed in (0, 1, 2):
        N[f"retrain_seed{seed}"], _ = row(lambda v, s, d, seed=seed: rp(v, s, d, seed), GF, base_map)
    e = lambda v, s, d, seed: (rp(v, s, d, seed)["P"] - GF[v]) / GF[v]
    sds = [statistics.pstdev([e(v, s, d, k) for k in (0, 1, 2)]) for v in V for s in SIGMAS for d in DELTAS]
    N["retrain_seed_spread"] = dict(
        median_sd=round(statistics.median(sds), 3), max_sd=round(max(sds), 3),
        sign_flip_cells=sum(1 for v in V for s in SIGMAS for d in DELTAS
                            if len({e(v, s, d, k) >= 0 for k in (0, 1, 2)}) > 1),
        frontal_median_sd=round(statistics.median(
            [statistics.pstdev([e(v, s, d, k) for k in (0, 1, 2)]) for v in FR for s in SIGMAS for d in DELTAS]), 3))

    apple = load("apple_val0000", per_video=True)
    a_e, detail = {}, {}
    for s in (640, 960, 1280):
        g1 = apple[f"apple_s{s}_d1.json"]["G"]
        g3 = sum(x["G"] for x in apple[f"apple_s{s}_d1.json"]["videos"] if x["video"] in APPLE_UNIQUE3)
        es, e3, cv = [], [], []
        for d in DELTAS:
            t = apple[f"apple_s{s}_d{d}.json"]
            a_e[f"{s}/{d}"] = round((t["P"] - g1) / g1, 3)
            es.append(a_e[f"{s}/{d}"])
            e3.append(round((sum(x["P"] for x in t["videos"] if x["video"] in APPLE_UNIQUE3) - g3) / g3, 3))
            cv.append(f"{t['G'] - t['M']}/{g1}")
        detail[str(s)] = dict(P=[apple[f"apple_s{s}_d{d}.json"]["P"] for d in DELTAS], G=g1,
                              bracket=bracket(es), coverage=cv, e_unique3=e3, bracket_unique3=bracket(e3))
    N["apple_val0000"] = a_e
    N["apple_val0000_detail"] = detail

    cost = {}
    for name, stub in members("timing"):
        arm, d, rep = name[:-5].rsplit("_", 2)
        tot = 0.0; ann = 0
        for i, v in enumerate(stub["videos"]):
            ms = list(v["frame_ms"])
            if i == 0 and ms:
                ms[0] = statistics.median(ms)  # the first frame carries initialisation
            tot += sum(ms); ann += v["annotated_frames"]
        cost.setdefault(f"{arm}_{d}", []).append(tot / ann)
    cost = {k: statistics.median(v) for k, v in cost.items()}
    hp = json.loads((ROOT / "results" / "hota_timing_session.json").read_text())
    hc = json.loads((ROOT / "results" / "common_timeline_hota.json").read_text())
    for tag, H in (("processed", lambda k: hp[f"{k}_r1"]["HOTA"]),
                   ("common_timeline", lambda k: hc[f"{k}_r1"]["common"]["HOTA"])):
        pts = {k: (cost[k], H(k)) for k in cost}
        nd = frontier(pts)
        N[f"frontier_{tag}"] = dict(arms=len(pts), non_dominated=len(nd),
                                    tiled_on_frontier=[k for k in nd if k.startswith("tiled")],
                                    members={k: [round(pts[k][0], 1), round(pts[k][1], 4)] for k in nd})

    (ROOT / "results" / "robustness_numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True))
    for k, v in N.items():
        print(f"{k:22s} {v}")
    if args.check:
        exp = json.loads((ROOT / "results" / "expected_robustness.json").read_text())
        bad = [k for k, v in exp.items() if json.loads(json.dumps(N.get(k))) != v]
        missing = sorted(set(N) - set(exp))
        if bad or missing:
            sys.exit(f"DRIFT in {bad}; not in the expected file: {missing}")
        print(f"all {len(exp)} expected entries match")


if __name__ == "__main__":
    main()
