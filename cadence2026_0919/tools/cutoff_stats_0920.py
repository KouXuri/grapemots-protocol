"""Summarise results/cutoff_sensitivity.json (per-arm terms of the frozen predictions
scored against references rebuilt at 1/20/100/400 mask pixels by
tools/cutoff_sensitivity_0920.py, which needs the GrapeMOTS masks).

    python3 tools/cutoff_stats_0920.py [path/to/cutoff_sensitivity.json]

With no argument it reads the archived file. tools/robustness_0921.py --check
recomputes the two cutoff rows of Table I from the same file.
"""
import json, statistics, sys
from pathlib import Path
V = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
S = (1536, 2048, 2560, 3072, 3840); D = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "cutoff_sensitivity.json"))
for c in (1, 20, 100, 400):
    strict = ties = rev = 0; cmap = {}; cs = []; gfull = {}
    for v in V:
        gfull[v] = res[v][f"{c}|3072|1"]["G_full"]
        for s in S:
            t = [res[v][f"{c}|{s}|{d}"] for d in D]
            es = [(x["P"] - gfull[v]) / gfull[v] for x in t]
            for a, b in zip(es, es[1:]):
                strict += b < a; ties += b == a; rev += b > a
            cmap[(v, s)] = ("<1" if es[0] < 0 else next((f"({D[i]},{D[i+1]})" for i in range(3) if es[i] >= 0 > es[i + 1]), ">8"))
            x = cmap[(v, s)]
            if x.startswith("("):
                lo, hi = (int(y) for y in x[1:-1].split(","))
                for d in (lo, hi):
                    r = res[v][f"{c}|{s}|{d}"]
                    cs.append(1 - (r["M"] + gfull[v] - r["G"]) / gfull[v])
    rows = sum(1 for x in cmap.values() if x.startswith("("))
    seqs = len({v for (v, s), x in cmap.items() if x.startswith("(")})
    ordered = sum(all(ORDER[cmap[(v, a)]] <= ORDER[cmap[(v, b)]] for a, b in zip(S, S[1:])) for v in V)
    print(f"cutoff {c:3d} px: G total {sum(gfull.values()):4d}  strict {strict}/150 (ties {ties}, rev {rev})  "
          f"sign-change rows {rows} in {seqs} seqs  ordered {ordered}/10  coverage at sign change max {max(cs):.3f} median {statistics.median(cs):.3f}")
