#!/usr/bin/env python3
"""Two detector families on identical plant-disjoint folds: crop-trained vs whole-frame.

Both families hold out one multi-view video (PathPlanning_2..8), train on the other
multi-view videos only and validate on the same one vine (rev_lovo_0920/index.json and
grapemots_fullframe_pponly_0922 list the same train/val/test videos per fold), at seeds
0, 1, 2, and read the test video whole, letterboxed to sigma, with the same BoT-SORT.
They differ in the training representation and budget only:
  crop   1280-pixel crops, 15 epochs, batch 8, two GPUs (archived arms, retrain.tar.gz);
  whole  whole frames at imgsz 3072 (rect), up to 60 epochs, patience 20, batch 2.

Same conventions as wholeframe_surface.py. Writes per-family surface statistics and
a cell-by-cell comparison.

    .venv/bin/python cadence_journal_1119/tools/crop_vs_whole.py RUN_DIR RETRAIN_TAR OUT.json
"""
from __future__ import annotations

import io
import json
import statistics as st
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

assert HAVE_SCIPY, "SciPy is required"
RUN, TAR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}

terms = {}
with tarfile.open(TAR) as tar:
    for m in tar.getmembers():
        if not m.isfile():
            continue
        name = Path(m.name).name                     # PathPlanning_2_s1__PathPlanning_2_s1536_d1.json
        fold, arm = name[:-5].split("__")
        v, seed = fold.rsplit("_s", 1)
        if v not in VIDEOS:
            continue
        s, d = arm.rsplit("_s", 1)[1].split("_d")
        stub = json.load(io.TextIOWrapper(tar.extractfile(m)))
        assert len(stub["videos"]) == 1 and stub["videos"][0]["video"] == v
        terms[("crop", int(seed), v, int(s), int(d))] = decompose(stub["videos"][0], 0.5, 1)
for seed in SEEDS:
    for v in VIDEOS:
        for s in SIGMAS:
            for d in DELTAS:
                x = json.loads((RUN / f"surface_seed{seed}" / f"{v}_s{s}_d{d}.json").read_text())["videos"][0]
                terms[("whole", seed, v, s, d)] = decompose(x, 0.5, 1)
missing = [(f, sd, v, s, d) for f in ("crop", "whole") for sd in SEEDS for v in VIDEOS for s in SIGMAS for d in DELTAS
           if (f, sd, v, s, d) not in terms]
assert not missing, missing[:5]

GF = {v: terms[("whole", 0, v, 3072, 1)]["G"] for v in VIDEOS}
for k, t in terms.items():
    if k[4] == 1:
        assert t["G"] == GF[k[2]], k
err = lambda t, v: (t["P"] - GF[v]) / GF[v]
cov = lambda t, v: 1 - (t["M"] + GF[v] - t["G"]) / GF[v]


def bracket(es):
    if es[0] < 0:
        return "<1"
    for i in range(3):
        if es[i] >= 0 > es[i + 1]:
            return f"({DELTAS[i]},{DELTAS[i + 1]})"
    return ">8"


N = {}
maps = {}
for fam in ("crop", "whole"):
    for seed in SEEDS:
        surf = {(v, s): [err(terms[(fam, seed, v, s, d)], v) for d in DELTAS] for v in VIDEOS for s in SIGMAS}
        cm = {k: bracket(es) for k, es in surf.items()}
        maps[(fam, seed)] = cm
        steps = [(a, b) for es in surf.values() for a, b in zip(es, es[1:])]
        cs = []
        for (v, s), b in cm.items():
            if b.startswith("("):
                lo, hi = (int(x) for x in b[1:-1].split(","))
                cs += [cov(terms[(fam, seed, v, s, lo)], v), cov(terms[(fam, seed, v, s, hi)], v)]
        near = [cov(terms[(fam, seed, v, s, d)], v) for v in VIDEOS for s in SIGMAS for d in DELTAS
                if abs(err(terms[(fam, seed, v, s, d)], v)) <= 0.10]
        sig = [err(terms[(fam, seed, v, b, d)], v) > err(terms[(fam, seed, v, a, d)], v)
               for v in VIDEOS for d in DELTAS for a, b in zip(SIGMAS, SIGMAS[1:])]
        N[f"{fam}|{seed}"] = dict(
            steps_down=sum(b < a for a, b in steps), ties=sum(b == a for a, b in steps),
            steps_up=sum(b > a for a, b in steps),
            sign_change_rows=sum(b.startswith("(") for b in cm.values()),
            sign_change_sequences=len({v for (v, s), b in cm.items() if b.startswith("(")}),
            always_over=sum(b == ">8" for b in cm.values()), always_under=sum(b == "<1" for b in cm.values()),
            ordered_sequences=sum(all(ORDER[cm[(v, a)]] <= ORDER[cm[(v, b)]] for a, b in zip(SIGMAS, SIGMAS[1:]))
                                  for v in VIDEOS),
            sigma_steps_raising=sum(sig),
            coverage_at_sign_change_max=round(max(cs), 4) if cs else None,
            near_zero_n=len(near), near_zero_max_coverage=round(max(near), 4) if near else None,
            median_error_delta1=round(st.median(surf[k][0] for k in surf), 3),
            median_coverage_delta1=round(st.median(cov(terms[(fam, seed, v, s, 1)], v)
                                                   for v in VIDEOS for s in SIGMAS), 3))

# cell by cell, same seed, video, scale, interval
cells = [(sd, v, s, d) for sd in SEEDS for v in VIDEOS for s in SIGMAS for d in DELTAS]
diff = [err(terms[("whole", *c)], c[1]) - err(terms[("crop", *c)], c[1]) for c in cells]
sign_same = sum((err(terms[("whole", *c)], c[1]) < 0) == (err(terms[("crop", *c)], c[1]) < 0) for c in cells)
bracket_same = sum(maps[("whole", sd)][(v, s)] == maps[("crop", sd)][(v, s)] for sd in SEEDS for v in VIDEOS for s in SIGMAS)
N["cell_by_cell"] = dict(cells=len(cells), sign_same=sign_same,
                         error_difference_median=round(st.median(diff), 3),
                         abs_error_difference_median=round(st.median(abs(x) for x in diff), 3),
                         rows_same_bracket=bracket_same, rows=len(SEEDS) * len(VIDEOS) * len(SIGMAS))
# per scale: does the whole-frame family read more or fewer bunches?
N["by_sigma"] = {s: dict(
    median_error_crop=round(st.median(err(terms[("crop", sd, v, s, d)], v) for sd in SEEDS for v in VIDEOS for d in DELTAS), 3),
    median_error_whole=round(st.median(err(terms[("whole", sd, v, s, d)], v) for sd in SEEDS for v in VIDEOS for d in DELTAS), 3),
    median_cov_crop=round(st.median(cov(terms[("crop", sd, v, s, d)], v) for sd in SEEDS for v in VIDEOS for d in DELTAS), 3),
    median_cov_whole=round(st.median(cov(terms[("whole", sd, v, s, d)], v) for sd in SEEDS for v in VIDEOS for d in DELTAS), 3))
    for s in SIGMAS}
# seed spread within each family, same cell
for fam in ("crop", "whole"):
    sp = [max(err(terms[(fam, sd, v, s, d)], v) for sd in SEEDS) - min(err(terms[(fam, sd, v, s, d)], v) for sd in SEEDS)
          for v in VIDEOS for s in SIGMAS for d in DELTAS]
    flips = sum(len({err(terms[(fam, sd, v, s, d)], v) < 0 for sd in SEEDS}) > 1
                for v in VIDEOS for s in SIGMAS for d in DELTAS)
    N[f"seed_spread|{fam}"] = dict(median=round(st.median(sp), 3), max=round(max(sp), 3), sign_flips=flips, cells=len(sp))
N["error"] = {"|".join(map(str, k)): round(err(t, k[2]), 4) for k, t in terms.items()}
N["coverage"] = {"|".join(map(str, k)): round(cov(t, k[2]), 4) for k, t in terms.items()}
OUT.write_text(json.dumps(N, indent=1, sort_keys=True))
print(json.dumps({k: v for k, v in N.items() if k not in ("error", "coverage")}, indent=1))
