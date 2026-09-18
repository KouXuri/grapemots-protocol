import json, glob, os, statistics, sys
sys.path.insert(0, "tools")
from decompose_count_error import decompose
cost = {}; raw = {}
for f in sorted(glob.glob("runs/ctl_0919/timing/*.json")):
    arm, d, rep = os.path.basename(f)[:-5].rsplit("_", 2)
    j = json.load(open(f)); tot = 0.0; ann = 0; proc = 0
    for i, v in enumerate(j["videos"]):
        ms = list(v["frame_ms"])
        if i == 0: ms[0] = statistics.median(ms)
        tot += sum(ms); ann += v["annotated_frames"]; proc += len(ms)
    cost.setdefault((arm, int(d[1:])), []).append(tot / ann)
    raw.setdefault((arm, int(d[1:])), []).append(tot / proc)
    if rep == "r1" and "frame_gt_boxes" in j["videos"][0]:
        P = G = U = D = M = 0
        for v in j["videos"]:
            r = decompose({k: v[k] for k in ("video", "frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes")}, 0.5, 1)
            P += r["P"]; G += r["G"]; U += r["U"]; D += r["D"]; M += r["M"]
        raw.setdefault(("terms", arm, int(d[1:])), (P, G, U, D, M, round((P - 138) / 138, 3)))
out = {}
for (arm, d), v in sorted(cost.items()):
    out[f"{arm}/{d}"] = dict(ms_per_annotated=[round(x, 1) for x in v], median=round(statistics.median(v), 1),
                             ms_per_processed_median=round(statistics.median(raw[(arm, d)]), 1))
    t = raw.get(("terms", arm, d))
    print(f"{arm:10s} d{d}  ms/ann {out[f'{arm}/{d}']['median']:6.1f}  reps {out[f'{arm}/{d}']['ms_per_annotated']}  ms/proc {out[f'{arm}/{d}']['ms_per_processed_median']:6.1f}  terms {t}")
json.dump(out, open("runs/ctl_0919/timing_summary.json", "w"), indent=1)
