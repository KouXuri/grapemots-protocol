#!/usr/bin/env python3
"""Standard surface statistics for any results directory of {video}_s{sigma}_d{delta}.json.

Same conventions as the paper: tau = 1, matching IoU 0.5, e = (P - G_full)/G_full
with G_full the reference trajectories over all annotated frames (tracker
independent, read once from the frozen base surface).
"""
import json, sys, statistics
sys.path.insert(0, "tools")
from decompose_count_error import decompose
V = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
S = (1536, 2048, 2560, 3072, 3840); D = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
BASE = "runs/lovo_surface_0918/results"
GF = {v: decompose(json.load(open(f"{BASE}/{v}_s3072_d1.json"))["videos"][0], 0.5, 1)["G"] for v in V}


def stats(dirname, label):
    T = {}
    for v in V:
        for s in S:
            for d in D:
                f = f"{dirname}/{v}_s{s}_d{d}.json"
                T[(v, s, d)] = decompose(json.load(open(f))["videos"][0], 0.5, 1)
    err = lambda k: (T[k]["P"] - GF[k[0]]) / GF[k[0]]
    cov = lambda k: 1 - (T[k]["M"] + GF[k[0]] - T[k]["G"]) / GF[k[0]]
    strict = ties = rev = 0
    cmap = {}
    for v in V:
        for s in S:
            es = [err((v, s, d)) for d in D]
            for a, b in zip(es, es[1:]):
                strict += b < a; ties += b == a; rev += b > a
            cmap[(v, s)] = ("<1" if es[0] < 0 else next((f"({D[i]},{D[i+1]})" for i in range(3) if es[i] >= 0 > es[i + 1]), ">8"))
    rows = sum(1 for x in cmap.values() if x.startswith("("))
    seqs = len({v for (v, s), x in cmap.items() if x.startswith("(")})
    ordered = sum(all(ORDER[cmap[(v, a)]] <= ORDER[cmap[(v, b)]] for a, b in zip(S, S[1:])) for v in V)
    cs = []
    for (v, s), x in cmap.items():
        if x.startswith("("):
            lo, hi = (int(y) for y in x[1:-1].split(","))
            cs += [cov((v, s, lo)), cov((v, s, hi))]
    ud = sum(T[(v, s, 8)]["U"] < T[(v, s, 1)]["U"] for v in V for s in S)
    dd = sum(T[(v, s, 8)]["D"] < T[(v, s, 1)]["D"] for v in V for s in S)
    mu = sum(T[(v, s, 8)]["M"] + GF[v] - T[(v, s, 8)]["G"] > T[(v, s, 1)]["M"] for v in V for s in S)
    sig = [err((v, b, d)) > err((v, a, d)) for v in V for d in D for a, b in zip(S, S[1:])]
    out = dict(label=label, strict=strict, ties=ties, reversals=rev, sign_change_rows=rows, sign_change_sequences=seqs,
               ordered_sequences=ordered, coverage_max=round(max(cs), 4) if cs else None,
               coverage_median=round(statistics.median(cs), 3) if cs else None,
               U_falls=ud, D_falls=dd, M_rises=mu, sigma_steps_raising=f"{sum(sig)}/{len(sig)}",
               e_delta1_range=[round(min(err((v, s, 1)) for v in V for s in S), 3),
                               round(max(err((v, s, 1)) for v in V for s in S), 3)],
               crossing_map={f"{v}|{s}": x for (v, s), x in cmap.items()})
    print(f"{label:12s} strict {strict}/150 (ties {ties}, reversals {rev})  sign-change rows {rows} in {seqs} seqs  "
          f"ordered {ordered}/10  coverage at sign change max {out['coverage_max']} median {out['coverage_median']}  "
          f"U/D fall, M rise {ud}/{dd}/{mu}  sigma steps {out['sigma_steps_raising']}")
    for v in V:
        print("   ", v, [cmap[(v, s)] for s in S])
    return out


if __name__ == "__main__":
    res = [stats(d, l) for d, l in zip(sys.argv[1::2], sys.argv[2::2])]
    json.dump(res, open("runs/rev_0920/cpu/surface_stats.json", "w"), indent=1)
