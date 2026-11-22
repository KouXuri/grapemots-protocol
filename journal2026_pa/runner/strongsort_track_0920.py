#!/usr/bin/env python3
"""StrongSORT over dumped detections, at each processing interval, in our result schema.

StrongSORT (Du et al., IEEE TMM 25, 2023) associates by appearance embeddings
(OSNet ReID, BoxMOT's default osnet_x0_25_msmt17) under a Mahalanobis motion gate,
and only falls back to IoU for leftovers, so it can re-associate a target whose
boxes no longer overlap -- the class the paper's mechanism does not cover.
BoxMOT defaults are kept except min_conf, set to 0.25 so that StrongSORT receives
exactly the detections BoT-SORT received. Output matches track_grapemots_mot.py
(frame_predicted_ids/boxes, frame_gt_ids/boxes), so decompose_count_error reads it.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from boxmot import ReIDModel
from boxmot.trackers.bbox.strongsort import StrongSort

ap = argparse.ArgumentParser()
ap.add_argument("--dets", type=Path, required=True); ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
d = json.loads(a.dets.read_text())
reid = ReIDModel("osnet_x0_25_msmt17.pt", device=a.device, half=False)
backend = getattr(reid, "model", reid)
for delta in a.deltas:
    out = a.out_dir / f"{d['video']}_s{d['imgsz']}_d{delta}.json"
    if out.exists():
        continue
    trk = StrongSort(reid_model=backend, min_conf=0.25)
    frames = d["frames"][::delta]
    fp_ids, fp_boxes, fg_ids, fg_boxes, names = [], [], [], [], []
    for f in frames:
        img = cv2.imread(f["path"])
        dets = np.array([[*b, s, 0] for b, s in zip(f["boxes"], f["scores"])], dtype=float).reshape(-1, 6)
        res = np.asarray(trk.update(dets, img))
        if res.size:
            fp_ids.append([int(r[4]) for r in res]); fp_boxes.append([[float(v) for v in r[:4]] for r in res])
        else:
            fp_ids.append([]); fp_boxes.append([])
        fg_ids.append(f["gt_ids"]); fg_boxes.append(f["gt_boxes"]); names.append(f["name"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {"tracker": "strongsort (boxmot, osnet_x0_25_msmt17, min_conf 0.25)",
                                          "weights": d["weights"], "imgsz": d["imgsz"], "frame_step": delta,
                                          "frame_offset": 0, "conf": d["conf"]},
                               "overall": None,
                               "videos": [{"video": d["video"], "frames": len(frames), "annotated_frames": len(d["frames"]),
                                           "frame_predicted_ids": fp_ids, "frame_predicted_boxes": fp_boxes,
                                           "frame_gt_ids": fg_ids, "frame_gt_boxes": fg_boxes, "frame_names": names}]}))
    print(f"{d['video']} s{d['imgsz']} d{delta}: {len(frames)} frames, {len({i for f in fp_ids for i in f})} identities")
