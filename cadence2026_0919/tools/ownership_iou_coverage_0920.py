import json, sys, statistics
sys.path.insert(0, "tools")
from decompose_count_error import decompose
V = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
S = (1536, 2048, 2560, 3072, 3840); D = (1, 2, 4, 8)
ent = {(v, s, d): json.load(open(f"runs/lovo_surface_0918/results/{v}_s{s}_d{d}.json"))["videos"][0] for v in V for s in S for d in D}
res = {}
for th in (0.2, 0.3, 0.5):
    T = {k: decompose(e, th, 1) for k, e in ent.items()}
    GF = {v: T[(v, 3072, 1)]["G"] for v in V}
    err = lambda t, v: (t["P"] - GF[v]) / GF[v]
    cov = lambda t, v: 1 - (t["M"] + GF[v] - t["G"]) / GF[v]
    cs = []
    for v in V:
        for s in S:
            es = [err(T[(v, s, d)], v) for d in D]
            for i in range(3):
                if es[i] >= 0 > es[i + 1]:
                    cs += [cov(T[(v, s, D[i])], v), cov(T[(v, s, D[i + 1])], v)]
    near = [cov(t, k[0]) for k, t in T.items() if abs(err(t, k[0])) <= 0.1]
    Uf = sum(T[(v, s, 1)]["U"] for v in V for s in S); Df = sum(T[(v, s, 1)]["D"] for v in V for s in S)
    res[th] = dict(max_cov_sign_change=round(max(cs), 4), median=round(statistics.median(cs), 3),
                   near_zero_max=round(max(near), 4), n_near=len(near), U_at_d1=Uf, D_at_d1=Df)
    print(th, res[th])
json.dump(res, open("runs/rev_0920/cpu/ownership_iou_coverage.json", "w"), indent=1)
