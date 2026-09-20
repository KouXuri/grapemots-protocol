import json, sys
sys.path.insert(0, "tools")
from decompose_count_error import decompose
Q = "runs/rev_0920/apple_val0000"
U3 = ("0006", "0007", "0008")
res = {}
for use in ("six", "unique3"):
    rows = {}
    for s in (640, 960, 1280):
        g1 = None; es = []; cv = []
        for d in (1, 2, 4, 8):
            vids = [v for v in json.load(open(f"{Q}/apple_s{s}_d{d}.json"))["videos"] if use == "six" or v["video"] in U3]
            T = [decompose(v, 0.5, 1) for v in vids]
            P = sum(t["P"] for t in T); G = sum(t["G"] for t in T); M = sum(t["M"] for t in T)
            if d == 1: g1 = G
            es.append(round((P - g1) / g1, 3)); cv.append(round(1 - (M + g1 - G) / g1, 3))
        rows[s] = {"e": es, "coverage": cv}
    res[use] = rows
    print(use, {s: r["e"] for s, r in rows.items()})
    print("   coverage", {s: r["coverage"] for s, r in rows.items()})
json.dump(res, open("runs/rev_0920/cpu/apple_val0000_surface.json", "w"), indent=1)
