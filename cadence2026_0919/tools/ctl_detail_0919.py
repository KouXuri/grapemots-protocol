import json, statistics, sys
sys.path.insert(0, "tools")
import analyse_ctl0919 as A
S = json.load(open("runs/ctl_0919/ctl_summary.json"))
# 1. how far each clock arm moved e (Delta>=2 cells), and its monotone violations
for arm in ("ret", "cu", "rt"):
    d = []; viol = []
    for v in A.VIDEOS:
        for s in A.SIGMAS:
            es = [A.err(A.base(v, s, 1), v)]
            for dl in (2, 4, 8):
                t = A.terms(f"runs/ctl_0919/{arm}/{v}_s{s}_d{dl}.json"); b = A.base(v, s, dl)
                d.append(A.err(t, v) - A.err(b, v)); es.append(A.err(t, v))
            for a, c in zip(es, es[1:]):
                if c > a: viol.append((v, s, round(a, 3), round(c, 3), round((c - a) * A.GFULL[v], 1)))
    print(arm, "median de", round(statistics.median(d), 4), "median |de|", round(statistics.median(map(abs, d)), 4),
          "range", round(min(d), 3), round(max(d), 3), "positive share", round(sum(x > 0 for x in d) / len(d), 2))
    print("   violations (v, s, e_before, e_after, identities):", viol)
    moved = {k: b for k, b in S[arm]["crossing_map"].items() if b != S["base"]["crossing_map"][k]}
    print("   moved cells:", {k: (S["base"]["crossing_map"][k], b) for k, b in moved.items()})
    # coverage at the sign changes under this arm
    cs = []
    for v in A.VIDEOS:
        for s in A.SIGMAS:
            b = S[arm]["crossing_map"][f"{v}|{s}"]
            if b.startswith("("):
                lo, hi = (int(x) for x in b[1:-1].split(","))
                for dl in (lo, hi):
                    t = A.base(v, s, 1) if dl == 1 else A.terms(f"runs/ctl_0919/{arm}/{v}_s{s}_d{dl}.json")
                    cs.append(A.cov(t, v))
    print("   coverage at sign changes: max", round(max(cs), 3), "median", round(statistics.median(cs), 3))
# 2. phase
ph = S["phase_cells"]
spans = sorted(((x["e_max"] - x["e_min"], k) for k, x in ph.items()), reverse=True)
print("phase top spans:", [(round(a, 3), k) for a, k in spans[:6]])
by_d = {}
for k, x in ph.items():
    by_d.setdefault(int(k.split("|")[2]), []).append(x["e_max"] - x["e_min"])
print("phase median span by Delta:", {d: round(statistics.median(v), 3) for d, v in sorted(by_d.items())})
print("phase flips:", S["phase_flips"])
gdrop = [(k, A.GFULL[k.split("|")[0]] - x["G_proc_min"]) for k, x in ph.items()]
print("max G_proc drop across phases:", max(gdrop, key=lambda t: t[1]), "cells with drop>=3:", sum(1 for _, g in gdrop if g >= 3))
print("phase-low moved:", {k: (S["base"]["crossing_map"][k], b) for k, b in S["phase_low"]["crossing_map"].items() if b != S["base"]["crossing_map"][k]})
print("phase-high moved:", {k: (S["base"]["crossing_map"][k], b) for k, b in S["phase_high"]["crossing_map"].items() if b != S["base"]["crossing_map"][k]})
print("phase max coverage at any cell with |e|<0.1:", max((x["cov_max"] for x in ph.values() if min(abs(x["e_min"]), abs(x["e_max"])) < 0.1), default=None))
