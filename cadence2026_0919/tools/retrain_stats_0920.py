import json, os, sys, statistics
sys.path.insert(0, "tools")
from decompose_count_error import decompose
MV = [f"PathPlanning_{i}" for i in range(2, 9)]
FR = [f"NoPathPlanning_{i}" for i in range(1, 4)]
V = MV + FR
S = (1536, 2048, 2560, 3072, 3840); D = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
def path(v, s, d, seed):
    p = f"runs/rev_0920/retrain/{v}_s{seed}/{v}_s{s}_d{d}.json"
    return p if os.path.exists(p) else f"runs/lovo_surface_0918/results/{v}_s{s}_d{d}.json"
T = {}
for seed in (0, 1, 2):
    for v in V:
        for s in S:
            for d in D:
                T[(seed, v, s, d)] = decompose(json.load(open(path(v, s, d, seed)))["videos"][0], 0.5, 1)
GF = {v: decompose(json.load(open(f"runs/lovo_surface_0918/results/{v}_s3072_d1.json"))["videos"][0], 0.5, 1)["G"] for v in V}
def brk(es):
    if es[0] < 0: return "<1"
    for i in range(3):
        if es[i] >= 0 > es[i + 1]: return f"({D[i]},{D[i+1]})"
    return ">8"
base = {}
for v in V:
    for s in S:
        base[(v, s)] = brk([(decompose(json.load(open(f"runs/lovo_surface_0918/results/{v}_s{s}_d{d}.json"))["videos"][0], 0.5, 1)["P"] - GF[v]) / GF[v] for d in D])
out = {}
for seed in (0, 1, 2):
    strict = ties = rev = 0; cmap = {}; cs = []
    for v in V:
        for s in S:
            es = [(T[(seed, v, s, d)]["P"] - GF[v]) / GF[v] for d in D]
            for a, b in zip(es, es[1:]): strict += b < a; ties += b == a; rev += b > a
            cmap[(v, s)] = brk(es)
            if cmap[(v, s)].startswith("("):
                lo, hi = (int(x) for x in cmap[(v, s)][1:-1].split(","))
                for d in (lo, hi):
                    t = T[(seed, v, s, d)]
                    cs.append(1 - (t["M"] + GF[v] - t["G"]) / GF[v])
    rows = sum(1 for x in cmap.values() if x.startswith("("))
    seqs = len({v for (v, s), x in cmap.items() if x.startswith("(")})
    ordered = sum(all(ORDER[cmap[(v, a)]] <= ORDER[cmap[(v, b)]] for a, b in zip(S, S[1:])) for v in V)
    moved = sum(cmap[k] != base[k] for k in cmap)
    out[seed] = dict(strict=strict, ties=ties, rev=rev, rows=rows, seqs=seqs, ordered=ordered,
                     cells_vs_base=moved, cov_max=round(max(cs), 4), cov_med=round(statistics.median(cs), 3))
    print(f"seed {seed}: strict {strict}/150 (ties {ties}, rev {rev})  sign-change rows {rows} in {seqs} seqs  "
          f"ordered {ordered}/10  cells differing from the published map {moved}/50  coverage max {out[seed]['cov_max']} median {out[seed]['cov_med']}")
sds = []
for v in V:
    for s in S:
        for d in D:
            es = [(T[(seed, v, s, d)]["P"] - GF[v]) / GF[v] for seed in (0, 1, 2)]
            sds.append(statistics.pstdev(es))
print(f"seed spread of e per cell: median sd {statistics.median(sds):.3f}, max {max(sds):.3f}")
# sign stability across seeds
flips = sum(1 for v in V for s in S for d in D
            if len({(T[(seed, v, s, d)]["P"] - GF[v]) >= 0 for seed in (0, 1, 2)}) > 1)
print(f"cells whose sign depends on the seed: {flips}/200")
json.dump({"per_seed": {str(k): v for k, v in out.items()},
           "seed_sd_median": statistics.median(sds), "seed_sd_max": max(sds), "sign_flip_cells": flips},
          open("runs/rev_0920/cpu/retrain_stats.json", "w"), indent=1)
