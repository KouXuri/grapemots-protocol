"""AppleMOT geometry and structural sign curve, on the definitions the vineyard and
pedestrian corpora used (tools/thinned_geometry.py, tools/sequence_structure_stats.py):
r = centre displacement / sqrt(area of the earlier box), over same-identity pairs of
consecutive kept frames; a median per sequence, then across sequences.
The curve is the sparse arm's pooled error at each thinning step, scored against the
reference at the kept frames -- the design the pedestrian corpora were read under."""
import json, math, statistics, sys
sys.path.insert(0, sys.argv[1])
from decompose_count_error import decompose
RAW, OUT = sys.argv[2], sys.argv[3]

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0

dense = {v["video"]: v for v in json.load(open(f"{RAW}/apple_s1280_d1.json"))["videos"]}
out = {"by_step": {}, "sequences": [], "curve_s1280": {}}
for k in (1, 2, 4, 8):
    seq_r, seq_iou, pairs = [], [], 0
    for name, v in sorted(dense.items()):
        kept = list(range(0, len(v["frame_gt_ids"]), k))
        rs, ious = [], []
        for i, j in zip(kept, kept[1:]):
            a = dict(zip(v["frame_gt_ids"][i], v["frame_gt_boxes"][i]))
            b = dict(zip(v["frame_gt_ids"][j], v["frame_gt_boxes"][j]))
            for t, ba in a.items():
                bb = b.get(t)
                if bb is None: continue
                area = (ba[2]-ba[0]) * (ba[3]-ba[1])
                if area <= 0: continue
                d = math.hypot((ba[0]+ba[2])/2 - (bb[0]+bb[2])/2, (ba[1]+ba[3])/2 - (bb[1]+bb[3])/2)
                rs.append(d / math.sqrt(area)); ious.append(iou(ba, bb))
        pairs += len(rs)
        if rs:
            seq_r.append(statistics.median(rs)); seq_iou.append(statistics.median(ious))
            if k == 1:
                out["sequences"].append({"corpus": "applemot", "video": name,
                    "step_over_size_median": statistics.median(rs), "consecutive_iou_median": statistics.median(ious)})
    out["by_step"][str(k)] = {"sequence_median_r": statistics.median(seq_r),
                              "sequence_median_iou": statistics.median(seq_iou), "pairs": pairs, "sequences": len(seq_r)}
    # structural curve: sparse arm processed and scored at the kept frames
    arm = dense if k == 1 else {v["video"]: v for v in json.load(open(f"{RAW}/apple_s1280_d{k}.json"))["videos"]}
    T = [decompose(v, 0.5, 1) for v in arm.values()]
    P = sum(t["P"] for t in T); G = sum(t["G"] for t in T)
    U = sum(t["U"] for t in T); D = sum(t["D"] for t in T); M = sum(t["M"] for t in T)
    out["curve_s1280"][str(k)] = {"P": P, "G": G, "U": U, "D": D, "M": M,
                                  "signed_error": (P - G) / G, "assigned_fraction": 1 - M / G}
one = out["curve_s1280"]["1"]
out["base_UD_over_G_s1280_k1"] = (one["U"] + one["D"]) / one["G"]
json.dump(out, open(OUT, "w"), indent=1)
for k, v in out["by_step"].items():
    c = out["curve_s1280"][k]
    print(f"k={k}: r={v['sequence_median_r']:.3f} IoU={v['sequence_median_iou']:.3f} pairs={v['pairs']}  "
          f"sparse e={c['signed_error']:+.3f} cov={c['assigned_fraction']:.3f} U={c['U']} D={c['D']} M={c['M']} G={c['G']}")
print("(U+D)/G at k=1, s1280:", round(out["base_UD_over_G_s1280_k1"], 3))
