"""AppleMOT re-scored under the design of the vineyard cadence intervention.

The archived AppleMOT surface scores each arm at the frames it processed, so the
dense and the sparse arm were read at different instants. The vineyard experiment
reads both arms at the same instants against the same reference. Here the dense
arm (every frame processed) is scored only at frames 0, k, 2k, ... -- the frames
the sparse arm processed -- so the two arms share pixels, detector, tracker,
reference and scoring instants and differ only in what the tracker saw between
them. Frozen per-frame outputs only; no GPU, no re-tracking."""
import json, statistics, sys
sys.path.insert(0, sys.argv[1])
from decompose_count_error import decompose

RAW = sys.argv[2]
PER_FRAME = ("frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes")
UNIQUE3 = ("0006", "0007", "0008")

def load(s, d):
    return {v["video"]: v for v in json.load(open(f"{RAW}/apple_s{s}_d{d}.json"))["videos"]}

def subset(v, k):
    n = len(v["frame_gt_ids"])
    out = dict(v)
    for key in PER_FRAME:
        out[key] = [v[key][i] for i in range(0, n, k)]
    return out

out = {}
for s in (640, 960, 1280):
    dense = load(s, 1)
    for k in (2, 4, 8):
        sparse = load(s, k)
        for use in ("six", "unique3"):
            vids = [v for v in sorted(dense) if use == "six" or v in UNIQUE3]
            pooled = {a: dict(P=0, G=0, U=0, D=0, M=0) for a in ("dense", "sparse")}
            deltas = []
            for v in vids:
                a = decompose(subset(dense[v], k), 0.5, 1)
                b = decompose(sparse[v], 0.5, 1)
                assert len(subset(dense[v], k)["frame_gt_ids"]) == len(sparse[v]["frame_gt_ids"]), v
                assert a["G"] == b["G"], (s, k, v, a["G"], b["G"])      # same reference
                for arm, t in (("dense", a), ("sparse", b)):
                    for key in pooled[arm]:
                        pooled[arm][key] += t[key]
                deltas.append((a["P"] - a["G"]) / a["G"] - (b["P"] - b["G"]) / b["G"])
            row = {}
            for arm, t in pooled.items():
                row[arm] = dict(t, e=round((t["P"] - t["G"]) / t["G"], 3),
                                coverage=round(1 - t["M"] / t["G"], 3),
                                identity=(t["P"] - t["G"] == t["U"] + t["D"] - t["M"]))
            row["paired_median"] = round(statistics.median(deltas), 3)
            row["up/down/tie"] = f"{sum(d>0 for d in deltas)}/{sum(d<0 for d in deltas)}/{sum(d==0 for d in deltas)}"
            out[f"s{s}_k{k}_{use}"] = row
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(f"{'cell':18} {'e dense':>8} {'e sparse':>8} {'cov d':>6} {'cov s':>6} {'G':>5} {'up/dn/tie':>9} {'median':>7}")
for key, r in out.items():
    print(f"{key:18} {r['dense']['e']:+8.3f} {r['sparse']['e']:+8.3f} {r['dense']['coverage']:6.3f} {r['sparse']['coverage']:6.3f} {r['dense']['G']:5d} {r['up/down/tie']:>9} {r['paired_median']:+7.3f}")
