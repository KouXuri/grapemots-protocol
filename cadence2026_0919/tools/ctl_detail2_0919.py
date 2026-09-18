import json, glob, os, sys
sys.path.insert(0, "tools")
import analyse_ctl0919 as A
S = json.load(open("runs/ctl_0919/ctl_summary.json"))
for arm in ("base", "phase_low", "phase_high", "ret", "cu", "rt"):
    seqs = {k.split("|")[0] for k, b in S[arm]["crossing_map"].items() if b.startswith("(")}
    print(arm, "rows", S[arm]["crossing_rows"], "sequences with a sign change", len(seqs))
arms = []
for p in glob.glob("runs/lovo_surface_0918/results/*_s*_d*.json"):
    arms.append(("base", p))
for fam in ("phase", "ret", "cu", "rt"):
    for p in glob.glob(f"runs/ctl_0919/{fam}/*.json"):
        arms.append((fam, p))
near = []; allc = []
for fam, p in arms:
    v = os.path.basename(p).split("_s")[0]
    t = A.terms(p); e = A.err(t, v); c = A.cov(t, v)
    allc.append(c)
    if abs(e) <= 0.1: near.append((c, fam, os.path.basename(p), round(e, 3)))
near.sort(reverse=True)
print("arms", len(arms), "with |e|<=0.1:", len(near), "max coverage", round(near[0][0], 3), near[:3])
print("median coverage of near-zero arms", round(sorted(x[0] for x in near)[len(near)//2], 3), "max coverage any arm", round(max(allc), 3))
