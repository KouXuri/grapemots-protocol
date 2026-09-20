import json, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, "tools")
from decompose_count_error import frame_matches, iou_matrix
V = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
out = {}
tot = Counter()
for v in V:
    e = json.load(open(f"runs/lovo_surface_0918/results/{v}_s3072_d1.json"))["videos"][0]
    overlap = defaultdict(Counter); best = defaultdict(float)
    for pb, pi, gb, gi in zip(e["frame_predicted_boxes"], e["frame_predicted_ids"], e["frame_gt_boxes"], e["frame_gt_ids"]):
        for p, g in frame_matches(pb, pi, gb, gi, 0.5):
            overlap[p][g] += 1
        if pi and gi:
            m = iou_matrix(np.asarray(pb, float), np.asarray(gb, float)).max(axis=1)
            for p, x in zip(pi, m):
                best[p] = max(best[p], float(x))
    tracks = sorted({t for f in e["frame_predicted_ids"] for t in f})
    U = [t for t in tracks if not overlap[t]]
    bins = Counter("none" if best[t] == 0 else ("<0.2" if best[t] < 0.2 else "0.2-0.5") for t in U)
    out[v] = {"U": len(U), **bins}
    tot.update(bins); tot["U"] += len(U)
    print(f"{v:18s} U={len(U):3d}  max IoU with any reference box: none {bins['none']:3d}  (0,0.2) {bins['<0.2']:3d}  [0.2,0.5) {bins['0.2-0.5']:3d}")
print("all ten:", dict(tot))
json.dump(out, open("runs/rev_0920/cpu/ownerless_max_iou_s3072_d1.json", "w"), indent=1)
