import json, os, sys, statistics
sys.path.insert(0, "tools")
from decompose_count_error import decompose
MV = [f"PathPlanning_{i}" for i in range(2, 9)]; FR = [f"NoPathPlanning_{i}" for i in range(1, 4)]
V = MV + FR; S = (1536, 2048, 2560, 3072, 3840); D = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
def load(p):
    return decompose(json.load(open(p))["videos"][0], 0.5, 1)["P"]
P = {}
for v in V:
    for s in S:
        for d in D:
            pub = f"runs/lovo_surface_0918/results/{v}_s{s}_d{d}.json"
            P[("pub", v, s, d)] = load(pub)
            for seed in (0, 1, 2):
                r = f"runs/rev_0920/retrain/{v}_s{seed}/{v}_s{s}_d{d}.json"
                P[(seed, v, s, d)] = load(r) if os.path.exists(r) else P[("pub", v, s, d)]
GF = {v: decompose(json.load(open(f"runs/lovo_surface_0918/results/{v}_s3072_d1.json"))["videos"][0], 0.5, 1)["G"] for v in V}
e = lambda src, v, s, d: (P[(src, v, s, d)] - GF[v]) / GF[v]
def brk(src, v, s):
    es = [e(src, v, s, d) for d in D]
    return "<1" if es[0] < 0 else next((f"({D[i]},{D[i+1]})" for i in range(3) if es[i] >= 0 > es[i + 1]), ">8")
cells = sum(brk("pub", v, s) != brk(0, v, s) for v in MV for s in S)
med = statistics.median([abs(e(0, v, s, d) - e("pub", v, s, d)) for v in MV for s in S for d in D])
print(f"composition (published vs plant-disjoint, both seed 0, multi-view): brackets differing {cells}/35, median |delta e| {med:.3f}")
for name, grp, n in (("frontal, composition unchanged", FR, 60), ("multi-view, plant-disjoint", MV, 140)):
    sds = [statistics.pstdev([e(seed, v, s, d) for seed in (0, 1, 2)]) for v in grp for s in S for d in D]
    flips = sum(1 for v in grp for s in S for d in D if len({e(seed, v, s, d) >= 0 for seed in (0, 1, 2)}) > 1)
    print(f"seed spread, {name}: median sd {statistics.median(sds):.3f}, max {max(sds):.3f}, sign flips {flips}/{n}")
for seed in (0, 1, 2):
    strict = sum(1 for v in V for s in S for a, b in zip([e(seed, v, s, d) for d in D], [e(seed, v, s, d) for d in D][1:]) if b < a)
    bad = [v for v in V if not all(ORDER[brk(seed, v, a)] <= ORDER[brk(seed, v, b)] for a, b in zip(S, S[1:]))]
    print(f"seed {seed}: strict {strict}/150, ordering breaks on {bad}")
print("published ordering breaks on", [v for v in V if not all(ORDER[brk('pub', v, a)] <= ORDER[brk('pub', v, b)] for a, b in zip(S, S[1:]))])
