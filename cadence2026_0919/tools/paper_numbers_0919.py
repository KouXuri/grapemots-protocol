#!/usr/bin/env python3
"""Recompute every number the camera-ready paper reports, from the frozen stubs.

Reads raw/<family>.tar.gz (written by build_release_0919.py), decomposes every
arm with decompose_count_error.decompose (tau = 1, IoU 0.5, Hungarian matching),
and writes results/paper_numbers.json. With --check it compares that file with
results/expected_numbers.json, which holds the values as the paper typesets them,
and fails on any drift.

    python3 tools/paper_numbers_0919.py            # recompute, write, print
    python3 tools/paper_numbers_0919.py --check    # and compare with the paper

Requires Python 3.9+, numpy and scipy (scipy is needed for the one-to-one
matching; without it the decomposition falls back to a greedy match and the
numbers can differ). No imagery, weights or GPU.

Conventions (paper, Section II-C):
  * e = (P - G) / G with G the reference trajectories over ALL annotated frames
    of a sequence, i.e. G at interval 1 and phase 0; M adds the trajectories the
    processed frames never show.
  * Sign-change bracket per (sequence, scale): '<1', '(a,b)', '>8'. No
    interpolation.
  * Cost per annotated frame = timed span over all processed frames, with the
    first processed frame of each run (model initialisation) replaced by the
    run's median frame time, divided by the annotated frames; median of three
    repeats.
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

VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
CIRCLING, FRONTAL = VIDEOS[:7], VIDEOS[7:]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}


def load(family: str) -> dict[str, dict]:
    """name -> {'terms': per-video decomposition list, 'stub': stub}"""
    out = {}
    with tarfile.open(ROOT / "raw" / f"{family}.tar.gz") as tar:
        for m in tar.getmembers():
            stub = json.load(io.TextIOWrapper(tar.extractfile(m)))
            # timing repeats 2-3 keep ids only; they are read for cost, not counts
            terms = ([decompose(v, 0.5, 1) for v in stub["videos"]]
                     if all("frame_predicted_boxes" in v for v in stub["videos"]) else None)
            out[Path(m.name).name] = {"terms": terms, "stub": stub}
    return out


def pooled(entry):
    t = {k: sum(x[k] for x in entry["terms"]) for k in ("P", "G", "U", "D", "M")}
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not HAVE_SCIPY:
        sys.exit("scipy is required for the one-to-one matching the paper uses")

    base = load("lovo_surface")
    tb = lambda v, s, d: pooled(base[f"{v}_s{s}_d{d}.json"])
    GF = {v: tb(v, 3072, 1)["G"] for v in VIDEOS}
    err = lambda t, v: (t["P"] - GF[v]) / GF[v]
    cov = lambda t, v: 1 - (t["M"] + GF[v] - t["G"]) / GF[v]

    def bracket(es):
        if es[0] < 0:
            return "<1"
        for i in range(3):
            if es[i] >= 0 > es[i + 1]:
                return f"({DELTAS[i]},{DELTAS[i + 1]})"
        return ">8"

    def surface_stats(get):
        """get(v, s, d) -> terms for d >= 2; d = 1 is always the base arm."""
        surf = {(v, s): [err(tb(v, s, 1), v)] + [err(get(v, s, d), v) for d in DELTAS[1:]]
                for v in VIDEOS for s in SIGMAS}
        steps = sum(1 for es in surf.values() for a, b in zip(es, es[1:]) if b <= a)
        strict = sum(1 for es in surf.values() for a, b in zip(es, es[1:]) if b < a)
        ties = sum(1 for es in surf.values() for a, b in zip(es, es[1:]) if b == a)
        cmap = {k: bracket(es) for k, es in surf.items()}
        rows = sum(1 for b in cmap.values() if b.startswith("("))
        seqs = len({v for (v, s), b in cmap.items() if b.startswith("(")})
        ordered = sum(all(ORDER[cmap[(v, a)]] <= ORDER[cmap[(v, b)]] for a, b in zip(SIGMAS, SIGMAS[1:]))
                      for v in VIDEOS)
        return surf, cmap, dict(monotone_steps=steps, strict_steps=strict, tie_steps=ties,
                                reversed_steps=len(surf) * 3 - steps, sign_change_rows=rows,
                                sign_change_sequences=seqs, ordered_sequences=ordered)

    N = {}
    surf0, cmap0, N["surface"] = surface_stats(tb)
    N["crossing_map"] = {f"{v}|{s}": b for (v, s), b in cmap0.items()}
    N["U_falls_rows"] = sum(tb(v, s, 8)["U"] < tb(v, s, 1)["U"] for v in VIDEOS for s in SIGMAS)
    N["D_falls_rows"] = sum(tb(v, s, 8)["D"] < tb(v, s, 1)["D"] for v in VIDEOS for s in SIGMAS)
    N["M_rises_rows"] = sum(tb(v, s, 8)["M"] + GF[v] - tb(v, s, 8)["G"] > tb(v, s, 1)["M"]
                            for v in VIDEOS for s in SIGMAS)
    sig_steps = [(err(tb(v, b, d), v) > err(tb(v, a, d), v)) for v in VIDEOS for d in DELTAS
                 for a, b in zip(SIGMAS, SIGMAS[1:])]
    N["sigma_steps_raising"] = f"{sum(sig_steps)}/{len(sig_steps)}"
    grp_mono = {}
    for name, grp in (("circling", CIRCLING), ("frontal", FRONTAL)):
        G = sum(GF[v] for v in grp); ok_e = ok_c = 0
        for d in DELTAS:
            es = [(sum(tb(v, s, d)["P"] for v in grp) - G) / G for s in SIGMAS]
            cs = [1 - sum(tb(v, s, d)["M"] + GF[v] - tb(v, s, d)["G"] for v in grp) / G for s in SIGMAS]
            ok_e += all(a < b for a, b in zip(es, es[1:])); ok_c += all(a < b for a, b in zip(cs, cs[1:]))
        grp_mono[name] = dict(error_columns=ok_e, coverage_columns=ok_c)
    N["pooled_sigma_monotone"] = grp_mono
    e1 = {v: [err(tb(v, s, 1), v) for s in SIGMAS] for v in VIDEOS}
    N["delta1_error_range"] = [round(min(min(x) for x in e1.values()), 3), round(max(max(x) for x in e1.values()), 3)]
    cs = []
    for (v, s), b in cmap0.items():
        if b.startswith("("):
            lo, hi = (int(x) for x in b[1:-1].split(","))
            cs += [cov(tb(v, s, lo), v), cov(tb(v, s, hi), v)]
    N["coverage_at_sign_change"] = dict(max=round(max(cs), 4), median=round(statistics.median(cs), 3), n=len(cs))
    N["G_full"] = GF

    # controls
    fam = {a: load(a) for a in ("ret", "cu", "rt")}
    for a, data in fam.items():
        _, cm, st = surface_stats(lambda v, s, d, data=data: pooled(data[f"{v}_s{s}_d{d}.json"]))
        st["cells_moved_sparser"] = sum(ORDER[cm[k]] > ORDER[cmap0[k]] for k in cm)
        st["cells_moved_denser"] = sum(ORDER[cm[k]] < ORDER[cmap0[k]] for k in cm)
        de = [err(pooled(data[f"{v}_s{s}_d{d}.json"]), v) - err(tb(v, s, d), v)
              for v in VIDEOS for s in SIGMAS for d in DELTAS[1:]]
        st["median_error_change"] = round(statistics.median(de), 3)
        N[a] = st
    phase = load("phase")
    cells = {}
    for v in VIDEOS:
        for s in SIGMAS:
            for d in DELTAS[1:]:
                ts = [tb(v, s, d)] + [pooled(phase[f"{v}_s{s}_d{d}_o{k}.json"]) for k in range(1, d)]
                cells[(v, s, d)] = ([err(t, v) for t in ts], min(t["G"] for t in ts))
    spans = {d: statistics.median(max(es) - min(es) for (v, s, dd), (es, _) in cells.items() if dd == d)
             for d in DELTAS[1:]}
    N["phase"] = dict(median_span={d: round(x, 3) for d, x in spans.items()},
                      max_span=round(max(max(es) - min(es) for es, _ in cells.values()), 3),
                      sign_flip_cells=sum((min(es) < 0) != (max(es) < 0) for es, _ in cells.values()),
                      max_G_proc_drop=max(GF[v] - g for (v, s, d), (_, g) in cells.items()))
    import itertools
    def phase_err(v, s, d, k):
        return err(tb(v, s, d) if k == 0 else pooled(phase[f"{v}_s{s}_d{d}_o{k}.json"]), v)
    def ordered_under(v, k2, k4, k8):
        m = [bracket([err(tb(v, s, 1), v), phase_err(v, s, 2, k2), phase_err(v, s, 4, k4),
                      phase_err(v, s, 8, k8)]) for s in SIGMAS]
        return all(ORDER[a] <= ORDER[b] for a, b in zip(m, m[1:]))
    combos = [(v, c) for v in VIDEOS for c in itertools.product(range(2), range(4), range(8))]
    bad = [(v, c) for v, c in combos if not ordered_under(v, *c)]
    uni = [(v, k) for v in VIDEOS for k in range(8) if not ordered_under(v, k % 2, k % 4, k % 8)]
    N["phase_combinations"] = dict(combinations=len(combos), ordering_violations=len(bad),
                                   sequences_with_violations=sorted({v for v, _ in bad}),
                                   uniform_remainder=f"{len(uni)}/{len(VIDEOS) * 8}",
                                   PP2_remainder2_counts={s: [tb("PathPlanning_2", s, 1)["P"]] + [
                                       (tb("PathPlanning_2", s, d)["P"] if 2 % d == 0 else
                                        pooled(phase[f"PathPlanning_2_s{s}_d{d}_o{2 % d}.json"])["P"])
                                       for d in DELTAS[1:]] for s in (2560, 3072)})
    for tag, pick in (("phase_low", min), ("phase_high", max)):
        lookup = {k: pick(es) for k, (es, _) in cells.items()}
        surf = {(v, s): [err(tb(v, s, 1), v)] + [lookup[(v, s, d)] for d in DELTAS[1:]]
                for v in VIDEOS for s in SIGMAS}
        cm = {k: bracket(es) for k, es in surf.items()}
        N[tag] = dict(monotone_steps=sum(1 for es in surf.values() for a, b in zip(es, es[1:]) if b <= a),
                      strict_steps=sum(1 for es in surf.values() for a, b in zip(es, es[1:]) if b < a),
                      tie_steps=sum(1 for es in surf.values() for a, b in zip(es, es[1:]) if b == a),
                      sign_change_rows=sum(1 for b in cm.values() if b.startswith("(")),
                      sign_change_sequences=len({v for (v, s), b in cm.items() if b.startswith("(")}),
                      ordered_sequences=sum(all(ORDER[cm[(v, a)]] <= ORDER[cm[(v, b)]]
                                                for a, b in zip(SIGMAS, SIGMAS[1:])) for v in VIDEOS),
                      cells_moved_sparser=sum(ORDER[cm[k]] > ORDER[cmap0[k]] for k in cm),
                      cells_moved_denser=sum(ORDER[cm[k]] < ORDER[cmap0[k]] for k in cm))
    near = []
    for fam_name, data in [("base", base), ("phase", phase)] + list(fam.items()):
        for name, entry in data.items():
            v = name.split("_s")[0]; t = pooled(entry)
            if abs(err(t, v)) <= 0.1:
                near.append(cov(t, v))
    N["near_zero_arms"] = dict(n_arms=len(base) + len(phase) + sum(len(d) for d in fam.values()),
                               within_0p1=len(near), max_coverage=round(max(near), 4),
                               median_coverage=round(statistics.median(near), 3))

    # held-out pair: count terms against G_full = 138
    held = load("heldout")
    hrows = {}
    for name, entry in held.items():
        t = pooled(entry); cfg = entry["stub"]["config"]
        G = 138
        hrows[f"{cfg['detector_mode']}{cfg['imgsz']}/{cfg['frame_step']}"] = dict(
            U=t["U"], D=t["D"], coverage=round(1 - (t["M"] + G - t["G"]) / G, 2),
            e=round((t["P"] - G) / G, 3))
    N["heldout"] = hrows

    # same-session timing
    timing = load("timing")
    cost = {}
    for name, entry in timing.items():
        arm, d, rep = name[:-5].rsplit("_", 2)
        tot = 0.0; ann = 0
        for v in entry["stub"]["videos"]:
            ms = list(v["frame_ms"])
            if v is entry["stub"]["videos"][0] and ms:
                ms[0] = statistics.median(ms)
            tot += sum(ms); ann += v["annotated_frames"]
        cost.setdefault(f"{arm}/{d[1:]}", []).append(tot / ann)
    N["cost_ms_per_annotated_frame"] = {k: round(statistics.median(v), 1) for k, v in sorted(cost.items())}
    N["cost_repeat_spread"] = {k: round(max(v) - min(v), 1) for k, v in sorted(cost.items())}
    t1 = {k: pooled(e) for k, e in timing.items() if e["terms"] is not None}
    N["timing_r1_terms"] = {k: dict(e=round((t["P"] - 138) / 138, 3), U=t["U"], D=t["D"]) for k, t in t1.items()}

    # AppleMOT, pooled over its six test sequences, G at interval 1
    apple = load("apple")
    ag = None
    a_e = {}
    for s in (640, 960, 1280):
        g1 = {x["video"]: x["G"] for x in apple[f"apple_s{s}_d1.json"]["terms"]}
        ag = sum(g1.values())
        for d in DELTAS:
            key = f"apple_s{s}_d{d}.json"
            if key in apple:
                P = sum(x["P"] for x in apple[key]["terms"])
                a_e[f"{s}/{d}"] = round((P - ag) / ag, 3)
    N["apple_error"] = a_e
    # The published split lists six test sequences; 0006/0010, 0007/0011 and
    # 0008/0012 carry identical annotations. Score one of each pair as a check.
    u3 = {}
    for s in (640, 960, 1280):
        keep = ("0006", "0007", "0008")
        g1 = sum(x["G"] for x in apple[f"apple_s{s}_d1.json"]["terms"] if x["video"] in keep)
        for d in DELTAS:
            P = sum(x["P"] for x in apple[f"apple_s{s}_d{d}.json"]["terms"] if x["video"] in keep)
            u3[f"{s}/{d}"] = round((P - g1) / g1, 3)
    N["apple_error_unique3"] = u3
    def apple_summary(tab):
        rows = {s: [tab[f"{s}/{d}"] for d in DELTAS] for s in (640, 960, 1280)}
        return dict(brackets={s: bracket(v) for s, v in rows.items()},
                    rows_monotone=all(all(b <= a for a, b in zip(v, v[1:])) for v in rows.values()),
                    columns_rising=all(rows[640][i] < rows[960][i] < rows[1280][i] for i in range(4)))
    N["apple_structure"] = {"six": apple_summary(a_e), "unique3": apple_summary(u3)}

    out = ROOT / "results" / "paper_numbers.json"
    out.write_text(json.dumps(N, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in N.items() if k not in ("crossing_map", "G_full")}, indent=1))
    if args.check:
        exp = json.loads((ROOT / "results" / "expected_numbers.json").read_text())
        bad = [k for k, v in exp.items() if json.loads(json.dumps(N.get(k))) != v]
        if bad:
            sys.exit(f"DRIFT in {bad}")
        print(f"all {len(exp)} expected entries match")


if __name__ == "__main__":
    main()
