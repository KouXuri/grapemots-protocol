#!/usr/bin/env python3
"""Contact sheet of ownerless tracks (the U term) for a visual audit.

Ownership is computed exactly as decompose_count_error.decompose does (per-frame
one-to-one IoU >= 0.5 matching, a track owned by the trajectory it covers most).
A fixed-seed random sample of the tracks that end up with no owner is cropped at
the frame where the track's box has its median area; the crop shows the track's
box in red and every reference box in view in blue. Output: PNG sheet + JSON key.
"""
from __future__ import annotations
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "tools")
from decompose_count_error import frame_matches  # noqa: E402

arm, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
v = json.load(open(arm))["videos"][0]
overlap = defaultdict(Counter)
for pb, pi, gb, gi in zip(v["frame_predicted_boxes"], v["frame_predicted_ids"], v["frame_gt_boxes"], v["frame_gt_ids"]):
    for p, g in frame_matches(pb, pi, gb, gi, 0.5):
        overlap[p][g] += 1
tracks = sorted({t for f in v["frame_predicted_ids"] for t in f})
ownerless = [t for t in tracks if not overlap[t]]
random.Random(0).shuffle(ownerless)
sample = ownerless[:n]
root = Path("datasets/grapemots_det_721/images/all")
tiles, key = [], []
for k, t in enumerate(sample):
    obs = [(fi, b) for fi, (ids, boxes) in enumerate(zip(v["frame_predicted_ids"], v["frame_predicted_boxes"])) for i, b in zip(ids, boxes) if i == t]
    obs.sort(key=lambda o: (o[1][2] - o[1][0]) * (o[1][3] - o[1][1]))
    fi, b = obs[len(obs) // 2]
    img = cv2.imread(str(root / v["frame_names"][fi]))
    x0, y0, x1, y1 = map(int, b); w, h = x1 - x0, y1 - y0; pad = max(w, h) * 2
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad); cx1, cy1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    crop = img[cy0:cy1, cx0:cx1].copy()
    for g in v["frame_gt_boxes"][fi]:
        gx0, gy0, gx1, gy1 = [int(c) for c in g]
        cv2.rectangle(crop, (gx0 - cx0, gy0 - cy0), (gx1 - cx0, gy1 - cy0), (255, 120, 0), 2)
    cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (0, 0, 255), 2)
    crop = cv2.resize(crop, (300, 300))
    cv2.putText(crop, str(k), (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(crop, str(k), (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    tiles.append(crop)
    key.append({"index": k, "track": t, "frame": v["frame_names"][fi], "box": b, "observations": len(obs)})
cols = 8
while len(tiles) % cols: tiles.append(np.full((300, 300, 3), 255, np.uint8))
sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
cv2.imwrite(str(out), sheet)
out.with_suffix(".json").write_text(json.dumps({"arm": arm, "ownerless": len(ownerless), "tracks": len(tracks),
                                                 "sample": key}, indent=1))
print(f"{len(ownerless)} ownerless of {len(tracks)} tracks; sheet of {len(key)} -> {out}")
