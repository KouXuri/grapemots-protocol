#!/usr/bin/env python3
"""Scale x interval surface of the whole-frame, plant-disjoint GrapeMOTS study.

Input: the per-frame outputs of queue runs/fullframe_pponly_0922 (raptor, finished
2026-09-28): seven multi-view test videos (PathPlanning_2..8), each read by detectors
trained on the other multi-view videos only (six training vines, one validation vine;
PathPlanning_1 training-only), whole frames at imgsz 3072, seeds 0, 1, 2. Every arm is
BoT-SORT (cfg/trackers/botsort_gmc.yaml) at confidence 0.25 unless named.

Conventions, as in the September draft (paper_numbers_0919.py):
  * G_full = reference trajectories over ALL annotated frames (the interval-1 arm);
    M_full = M + (G_full - G_processed): a trajectory the processed frames never show
    is a missed one. e = (P - G_full) / G_full, coverage = 1 - M_full / G_full.
  * Sign-change bracket per (video, scale, seed) over the tested intervals 1, 2, 4, 8:
    '<1', '(a,b)', '>8'. No interpolation.
  * Matched design (CBDCom 2026): the interval-1 arm scored only at the frames the
    interval-k arm processed, against that same reference.

    .venv/bin/python cadence_journal_1119/tools/wholeframe_surface.py RUN_DIR OUT.json

SciPy is required (maximum-total-IoU matching).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

if not HAVE_SCIPY:
    sys.exit("SciPy is required: without it the decomposition falls back to greedy matching")

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
PER_FRAME = ("frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes")


def video(path: Path) -> dict:
    vids = json.loads(path.read_text())["videos"]
    assert len(vids) == 1, path
    return vids[0]


def terms(path: Path) -> dict:
    v = video(path)
    t = decompose(v, 0.5, 1)
    t["frames"] = len(v["frame_gt_ids"])
    t["ms"] = v.get("mean_ms_per_frame")
    return t


# ----------------------------------------------------------------- load everything
surface = {}            # (seed, v, s, d) -> terms
for seed in SEEDS:
    for v in VIDEOS:
        for s in SIGMAS:
            for d in DELTAS:
                surface[(seed, v, s, d)] = terms(RUN / f"surface_seed{seed}" / f"{v}_s{s}_d{d}.json")

GF = {}
for v in VIDEOS:
    gs = {surface[(seed, v, s, 1)]["G"] for seed in SEEDS for s in SIGMAS}
    assert len(gs) == 1, (v, gs)          # one reference per video, whatever the detector
    GF[v] = gs.pop()


def err(t, v):
    return (t["P"] - GF[v]) / GF[v]


def cov(t, v):
    return 1 - (t["M"] + GF[v] - t["G"]) / GF[v]


def bracket(es):
    if es[0] < 0:
        return "<1"
    for i in range(3):
        if es[i] >= 0 > es[i + 1]:
            return f"({DELTAS[i]},{DELTAS[i + 1]})"
    return ">8"


def surface_stats(get, videos=VIDEOS):
    """get(v, s, d) -> terms. Steps compare consecutive intervals within (v, s)."""
    surf = {(v, s): [err(get(v, s, d), v) for d in DELTAS] for v in videos for s in SIGMAS}
    steps = [(a, b) for es in surf.values() for a, b in zip(es, es[1:])]
    cmap = {k: bracket(es) for k, es in surf.items()}
    return surf, cmap, dict(
        steps=len(steps),
        steps_down=sum(b < a for a, b in steps), ties=sum(b == a for a, b in steps),
        steps_up=sum(b > a for a, b in steps),
        sign_change_rows=sum(b.startswith("(") for b in cmap.values()),
        sign_change_sequences=len({v for (v, s), b in cmap.items() if b.startswith("(")}),
        always_over=sum(b == ">8" for b in cmap.values()),
        always_under=sum(b == "<1" for b in cmap.values()),
        ordered_sequences=sum(all(ORDER[cmap[(v, a)]] <= ORDER[cmap[(v, b)]]
                                  for a, b in zip(SIGMAS, SIGMAS[1:])) for v in videos))


N = {"G_full": GF, "videos": VIDEOS, "sigmas": SIGMAS, "deltas": DELTAS, "seeds": SEEDS}

# ----------------------------------------------------------------- main surface, per seed
maps = {}
per_seed = {}
for seed in SEEDS:
    get = lambda v, s, d, seed=seed: surface[(seed, v, s, d)]
    surf, cmap, stats = surface_stats(get)
    maps[seed] = cmap
    sig = [err(get(v, b, d), v) > err(get(v, a, d), v)
           for v in VIDEOS for d in DELTAS for a, b in zip(SIGMAS, SIGMAS[1:])]
    stats["sigma_steps_raising"] = sum(sig)
    stats["sigma_steps"] = len(sig)
    stats["U_falls_1to8"] = sum(get(v, s, 8)["U"] < get(v, s, 1)["U"] for v in VIDEOS for s in SIGMAS)
    stats["D_falls_1to8"] = sum(get(v, s, 8)["D"] < get(v, s, 1)["D"] for v in VIDEOS for s in SIGMAS)
    stats["Mfull_rises_1to8"] = sum(get(v, s, 8)["M"] + GF[v] - get(v, s, 8)["G"] > get(v, s, 1)["M"]
                                    for v in VIDEOS for s in SIGMAS)
    stats["surplus_drop_exceeds_coverage_loss"] = sum(
        (get(v, s, 1)["U"] + get(v, s, 1)["D"]) - (get(v, s, 8)["U"] + get(v, s, 8)["D"])
        > (get(v, s, 8)["M"] + GF[v] - get(v, s, 8)["G"]) - get(v, s, 1)["M"]
        for v in VIDEOS for s in SIGMAS)
    cs = []
    for (v, s), b in cmap.items():
        if b.startswith("("):
            lo, hi = (int(x) for x in b[1:-1].split(","))
            cs += [cov(get(v, s, lo), v), cov(get(v, s, hi), v)]
    stats["coverage_at_sign_change"] = (dict(n=len(cs), max=round(max(cs), 4), median=round(st.median(cs), 3))
                                        if cs else None)
    near = [cov(get(v, s, d), v) for v in VIDEOS for s in SIGMAS for d in DELTAS
            if abs(err(get(v, s, d), v)) <= 0.10]
    stats["near_zero"] = dict(n=len(near), max=round(max(near), 4) if near else None,
                              median=round(st.median(near), 3) if near else None)
    stats["delta1_error_range"] = [round(min(surf[k][0] for k in surf), 3), round(max(surf[k][0] for k in surf), 3)]
    per_seed[seed] = stats
N["per_seed"] = per_seed
N["crossing_map"] = {f"{seed}|{v}|{s}": maps[seed][(v, s)] for seed in SEEDS for v in VIDEOS for s in SIGMAS}
N["error"] = {f"{seed}|{v}|{s}|{d}": round(err(surface[(seed, v, s, d)], v), 4)
              for seed in SEEDS for v in VIDEOS for s in SIGMAS for d in DELTAS}
N["coverage"] = {f"{seed}|{v}|{s}|{d}": round(cov(surface[(seed, v, s, d)], v), 4)
                 for seed in SEEDS for v in VIDEOS for s in SIGMAS for d in DELTAS}
N["terms"] = {f"{seed}|{v}|{s}|{d}": {k: surface[(seed, v, s, d)][k] for k in ("P", "G", "U", "D", "M")}
              for seed in SEEDS for v in VIDEOS for s in SIGMAS for d in DELTAS}

# ----------------------------------------------------------------- agreement across seeds
same_bracket = sum(len({maps[seed][(v, s)] for seed in SEEDS}) == 1 for v in VIDEOS for s in SIGMAS)
sign_by_seed = sum(len({err(surface[(seed, v, s, d)], v) < 0 for seed in SEEDS}) > 1
                   for v in VIDEOS for s in SIGMAS for d in DELTAS)
spread = [max(err(surface[(seed, v, s, d)], v) for seed in SEEDS) - min(err(surface[(seed, v, s, d)], v) for seed in SEEDS)
          for v in VIDEOS for s in SIGMAS for d in DELTAS]
step_agree = 0
step_all_down = 0
for v in VIDEOS:
    for s in SIGMAS:
        for a, b in zip(DELTAS, DELTAS[1:]):
            dirs = {(err(surface[(seed, v, s, b)], v) < err(surface[(seed, v, s, a)], v)) for seed in SEEDS}
            step_agree += len(dirs) == 1
            step_all_down += dirs == {True}
N["across_seeds"] = dict(cells=len(spread), cells_sign_differs_by_seed=sign_by_seed,
                         error_spread_median=round(st.median(spread), 3), error_spread_max=round(max(spread), 3),
                         rows_same_bracket_all_seeds=same_bracket, rows=len(VIDEOS) * len(SIGMAS),
                         interval_steps_down_in_all_seeds=step_all_down, interval_steps=len(VIDEOS) * len(SIGMAS) * 3)

# sequence level: median over scales and seeds of e(1) - e(8)
seq_drop = {v: st.median(err(surface[(seed, v, s, 1)], v) - err(surface[(seed, v, s, 8)], v)
                         for seed in SEEDS for s in SIGMAS) for v in VIDEOS}
N["sequence_level"] = dict(median_drop_1to8={v: round(x, 3) for v, x in seq_drop.items()},
                           sequences_dropping=sum(x > 0 for x in seq_drop.values()))

# ----------------------------------------------------------------- matched design
matched = []
for seed in SEEDS:
    for v in VIDEOS:
        for s in SIGMAS:
            dense = video(RUN / f"surface_seed{seed}" / f"{v}_s{s}_d1.json")
            for k in DELTAS[1:]:
                sparse = video(RUN / f"surface_seed{seed}" / f"{v}_s{s}_d{k}.json")
                sub = dict(dense)
                for key in PER_FRAME:
                    sub[key] = dense[key][0::k]
                assert sub["frame_gt_ids"] == sparse["frame_gt_ids"], (seed, v, s, k)
                a, b = decompose(sub, 0.5, 1), decompose(sparse, 0.5, 1)
                assert a["G"] == b["G"]
                G = a["G"]
                matched.append(dict(seed=seed, video=v, sigma=s, k=k, G=G, P_dense=a["P"], P_sparse=b["P"],
                                    rises=a["P"] > b["P"], tie=a["P"] == b["P"],
                                    opposite=(a["P"] - G) * (b["P"] - G) < 0 and a["P"] > G))
N["matched"] = dict(n=len(matched), rises=sum(m["rises"] for m in matched), ties=sum(m["tie"] for m in matched),
                    falls=sum(not m["rises"] and not m["tie"] for m in matched),
                    opposite_sign=sum(m["opposite"] for m in matched),
                    opposite_sequences=sorted({m["video"] for m in matched if m["opposite"]}),
                    by_seed={seed: dict(rises=sum(m["rises"] for m in matched if m["seed"] == seed),
                                        opposite=sum(m["opposite"] for m in matched if m["seed"] == seed))
                             for seed in SEEDS})

# ----------------------------------------------------------------- seed-0 controls
base = lambda v, s, d: surface[(0, v, s, d)]
_, cmap0, _ = surface_stats(base)
controls = {}
for fam in ("ret", "cu", "rt"):
    data = {(v, s, d): terms(RUN / fam / f"{v}_s{s}_d{d}.json") for v in VIDEOS for s in SIGMAS for d in DELTAS[1:]}
    get = lambda v, s, d, data=data: base(v, s, 1) if d == 1 else data[(v, s, d)]
    _, cm, stats = surface_stats(get)
    stats["cells_moved_sparser"] = sum(ORDER[cm[k]] > ORDER[cmap0[k]] for k in cm)
    stats["cells_moved_denser"] = sum(ORDER[cm[k]] < ORDER[cmap0[k]] for k in cm)
    stats["median_error_change"] = round(st.median(err(data[(v, s, d)], v) - err(base(v, s, d), v)
                                                   for v in VIDEOS for s in SIGMAS for d in DELTAS[1:]), 3)
    controls[fam] = stats
bt = {(v, s, d): terms(RUN / "bytetrack" / f"{v}_s{s}_d{d}.json") for v in VIDEOS for s in SIGMAS for d in DELTAS}
_, cmb, stats = surface_stats(lambda v, s, d: bt[(v, s, d)])
stats["delta1_error_range"] = [round(min(err(bt[(v, s, 1)], v) for v in VIDEOS for s in SIGMAS), 3),
                               round(max(err(bt[(v, s, 1)], v) for v in VIDEOS for s in SIGMAS), 3)]
controls["bytetrack"] = stats

phase = {}
for v in VIDEOS:
    for s in SIGMAS:
        for d in DELTAS[1:]:
            phase[(v, s, d)] = [base(v, s, d)] + [terms(RUN / "phase" / f"{v}_s{s}_d{d}_o{k}.json") for k in range(1, d)]
spans = {d: st.median(max(err(t, v) for t in ts) - min(err(t, v) for t in ts)
                      for (v, s, dd), ts in phase.items() if dd == d) for d in DELTAS[1:]}
pc = {}
for tag, pick in (("phase_low", min), ("phase_high", max), ("phase_mean", None)):
    def get(v, s, d, pick=pick):
        if d == 1:
            return base(v, s, 1)
        es = [err(t, v) for t in phase[(v, s, d)]]
        val = st.mean(es) if pick is None else pick(es)
        return {"P": GF[v] * (1 + val), "G": GF[v], "M": 0, "U": 0, "D": 0}
    _, cm, stats = surface_stats(get)
    stats["cells_moved_sparser"] = sum(ORDER[cm[k]] > ORDER[cmap0[k]] for k in cm)
    stats["cells_moved_denser"] = sum(ORDER[cm[k]] < ORDER[cmap0[k]] for k in cm)
    pc[tag] = stats
controls["phase"] = dict(median_span={d: round(x, 3) for d, x in spans.items()},
                         sign_flip_cells=sum((min(err(t, v) for t in ts) < 0) != (max(err(t, v) for t in ts) < 0)
                                             for (v, s, d), ts in phase.items()),
                         cells=len(phase), **pc)
N["controls_seed0"] = controls

# all arms within +-0.10, over the surface (3 seeds) and every control
near = []
for key, t in surface.items():
    if abs(err(t, key[1])) <= 0.10:
        near.append(cov(t, key[1]))
N["near_zero_all_seeds"] = dict(arms=len(surface), within_0p1=len(near),
                                max=round(max(near), 4), median=round(st.median(near), 3))

# ----------------------------------------------------------------- training record
train = {}
for f in sorted(RUN.glob("PathPlanning_*_s?_verified.json")):
    b = json.loads(f.read_text())["budget"]
    train[f.name.replace("_verified.json", "")] = dict(best_epoch=b["best_epoch"], final_epoch=b["final_epoch"],
                                                       fitness=b["best_ap"])
N["training"] = train

OUT.write_text(json.dumps(N, indent=1, sort_keys=True, default=str))
show = {k: N[k] for k in ("per_seed", "across_seeds", "sequence_level", "matched", "controls_seed0", "near_zero_all_seeds")}
print(json.dumps(show, indent=1, default=str))
