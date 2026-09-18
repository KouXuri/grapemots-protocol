import json, os, sys
sys.path.insert(0, "tools")
from decompose_count_error import decompose
def path(s, d):
    for p in (f"runs/caout_stage13_20260829/pa/apple_s{s}_d{d}.json", f"runs/ctl_0919/apple/apple_s{s}_d{d}.json"):
        if os.path.exists(p): return p
out = {}
for s in (640, 960, 1280):
    g1 = None; row = []
    for d in (1, 2, 4, 8):
        j = json.load(open(path(s, d)))
        T = {k: 0 for k in "PGUDM"}; gproc = 0
        for v in j["videos"]:
            r = decompose({k: v[k] for k in ("video", "frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes")}, 0.5, 1)
            for k in "PGUDM": T[k] += r[k]
        if d == 1: g1 = T["G"]
        e = (T["P"] - g1) / g1; eproc = (T["P"] - T["G"]) / T["G"]
        cov = 1 - (T["M"] + g1 - T["G"]) / g1
        row.append((d, round(e, 3), round(eproc, 3), T["G"], round(cov, 3), sum(v["frames"] for v in j["videos"])))
        out[f"{s}/{d}"] = dict(e=e, e_proc=eproc, G_proc=T["G"], G_full=g1, coverage=cov, **T)
    print(s, row)
json.dump(out, open("runs/ctl_0919/apple_surface.json", "w"), indent=1)
