import json, sys, statistics
sys.path.insert(0, "tools")
from decompose_count_error import decompose
V = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
S = (1536, 2048, 2560, 3072, 3840); D = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}
base = lambda v, s, d: decompose(json.load(open(f"runs/lovo_surface_0918/results/{v}_s{s}_d{d}.json"))["videos"][0], 0.5, 1)
ph = lambda v, s, d, k: decompose(json.load(open(f"runs/ctl_0919/phase/{v}_s{s}_d{d}_o{k}.json"))["videos"][0], 0.5, 1)
GF = {v: base(v, 3072, 1)["G"] for v in V}
def brk(es):
    if es[0] < 0: return "<1"
    for i in range(3):
        if es[i] >= 0 > es[i + 1]: return f"({D[i]},{D[i+1]})"
    return ">8"
strict = ties = rev = 0; moved_s = moved_d = 0; ordered = 0; rows = 0; seqs = set(); cs = []
for v in V:
    rowmap = []
    for s in S:
        Pm = [base(v, s, 1)["P"]] + [statistics.mean([base(v, s, d)["P"]] + [ph(v, s, d, k)["P"] for k in range(1, d)]) for d in D[1:]]
        es = [(p - GF[v]) / GF[v] for p in Pm]
        for a, b in zip(es, es[1:]):
            strict += b < a; ties += b == a; rev += b > a
        x = brk(es); rowmap.append(x)
        b0 = brk([(base(v, s, d)["P"] - GF[v]) / GF[v] for d in D])
        moved_s += ORDER[x] > ORDER[b0]; moved_d += ORDER[x] < ORDER[b0]
        if x.startswith("("): rows += 1; seqs.add(v)
    ordered += all(ORDER[a] <= ORDER[b] for a, b in zip(rowmap, rowmap[1:]))
print(f"phase-mean: strict {strict} ties {ties} reversals {rev}; sign-change rows {rows} in {len(seqs)} seqs; "
      f"ordered {ordered}/10; cells moved sparser {moved_s} denser {moved_d}")
