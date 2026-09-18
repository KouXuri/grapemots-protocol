"""Re-read every arm of the paper's scale-cadence surface and price it per
annotated frame. Reads frozen JSONs only; e from tools/decompose_count_error."""
import json, os, sys
sys.path.insert(0, "tools")
from decompose_count_error import decompose
B = "runs/caout_cadence_budget_20260819/results"
H19 = json.load(open(f"{B}/hota_arms.json"))
H06 = json.load(open("runs/grid_0906/results/hota_arms.json"))
arms = {}
for s in (1536, 2048, 2560, 3072, 3840):
    for d in (1, 2, 4, 8):
        g = f"runs/grid_0906/results/grid_{s}_s{d}.json"
        if os.path.exists(g):
            arms[f"full{s}/{d}"] = (g, H06.get(f"grid_{s}_s{d}", {}).get("HOTA"))
arms["full1536/1"] = (f"{B}/full1536_s1.json", H19["full1536_s1"]["HOTA"])
arms["full2048/1"] = (f"{B}/full2048_s1.json", H19["full2048_s1"]["HOTA"])
arms["full2048/2"] = (f"{B}/full2048_s2.json", H19["full2048_s2"]["HOTA"])
arms["full2560/1"] = (f"{B}/full2560_s1.json", H19["full2560_s1"]["HOTA"])
arms["full3072/4"] = ("runs/caout_stage2_20260825/predicted/pred_full3072_s4.json", None)
for d in (1, 4, 8):
    arms[f"tiled1280/{d}"] = (f"{B}/tiled_s{d}.json", H19[f"tiled_s{d}"]["HOTA"])
out = {}
for name, (f, hota) in sorted(arms.items()):
    j = json.load(open(f)); c = j["config"]
    P = G = U = D = M = 0; proc = 0; el = 0.0
    for v in j["videos"]:
        stub = {k: v[k] for k in ("video", "frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes") if k in v}
        if "frame_gt_boxes" in stub:
            r = decompose(stub, 0.5, 1)
            P += r["P"]; G += r["G"]; U += r["U"]; D += r["D"]; M += r["M"]
        proc += v["frames"]; el += v["mean_ms_per_frame"] * v["frames"]
    out[name] = dict(file=f, weights=c.get("weights"), tracker=c.get("tracker"), mode=c.get("detector_mode"),
                     imgsz=c.get("imgsz"), tile=c.get("tile"), step=c.get("frame_step"), conf=c.get("conf"),
                     match_iou=c.get("match_iou"), processed=proc, ms_per_processed=el / proc,
                     ms_per_annotated=el / 583, HOTA=hota, P=P, G=G, U=U, D=D, M=M,
                     e=(P - G) / G if G else None)
    o = out[name]
    print(f"{name:13s} w={os.path.basename(os.path.dirname(os.path.dirname(str(o['weights']))))[:26]:26s} trk={os.path.basename(str(o['tracker'])):12s} "
          f"proc={proc:3d} ms/proc={o['ms_per_processed']:6.1f} ms/ann={o['ms_per_annotated']:6.1f} "
          f"HOTA={hota if hota is None else round(hota,4)} e={o['e'] if o['e'] is None else round(o['e'],3)} G={G} U={U} D={D} M={M}")
json.dump(out, open("/tmp/surface_cost_audit.json", "w"), indent=1)
