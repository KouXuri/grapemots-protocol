#!/usr/bin/env python3
"""A BoxMOT tracker over dumped detections, at each processing interval, in our result schema.

Same logic and output schema as tools/strongsort_track_0920.py; only the tracker changes.
Trackers (BoxMOT 23.0.0, raptor boxmot_venv):
  ocsort      OC-SORT (Cao et al., CVPR 2023): observation-centric re-update and momentum,
              built for non-linear motion and re-association after occlusion; no appearance.
  deepocsort  Deep OC-SORT (Maggiolino et al., ICIP 2023): OC-SORT plus appearance (the same
              OSNet-x0.25 MSMT17 weights as the StrongSORT runs) and camera-motion compensation.
BoxMOT defaults are kept except min_conf and det_thresh, both 0.25, so the tracker receives the
detections BoT-SORT received (new_track_thresh 0.25); with reference boxes at score 0.9 the two
settings change nothing. Output matches track_grapemots_mot.py, so decompose_count_error reads it.

    /home/kou/my_env/boxmot_venv/bin/python tools/boxmot_track_1120.py --tracker ocsort \
        --dets <dets.json> --out-dir <dir> [--deltas 1 2 4 8] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tracker", choices=["ocsort", "deepocsort"], required=True)
ap.add_argument("--dets", type=Path, required=True)
ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
d = json.loads(a.dets.read_text())
COMMON = dict(min_conf=0.25, det_thresh=0.25)

if a.tracker == "ocsort":
    from boxmot.trackers.bbox.ocsort import OcSort

    def make():
        return OcSort(**COMMON)
    DESC = "ocsort (boxmot 23.0.0 defaults; min_conf, det_thresh 0.25)"
else:
    from boxmot import ReIDModel
    from boxmot.trackers.bbox.deepocsort import DeepOcSort

    reid = ReIDModel("osnet_x0_25_msmt17.pt", device=a.device, half=False)
    backend = getattr(reid, "model", reid)

    def make():
        return DeepOcSort(reid_model=backend, **COMMON)
    DESC = "deepocsort (boxmot 23.0.0 defaults, osnet_x0_25_msmt17, CMC on; min_conf, det_thresh 0.25)"

for delta in a.deltas:
    out = a.out_dir / f"{d['video']}_s{d['imgsz']}_d{delta}.json"
    if out.exists():
        continue
    trk = make()
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
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"config": {"tracker": DESC, "weights": d["weights"], "imgsz": d["imgsz"],
                                          "frame_step": delta, "frame_offset": 0, "conf": d["conf"]},
                               "overall": None,
                               "videos": [{"video": d["video"], "frames": len(frames), "annotated_frames": len(d["frames"]),
                                           "frame_predicted_ids": fp_ids, "frame_predicted_boxes": fp_boxes,
                                           "frame_gt_ids": fg_ids, "frame_gt_boxes": fg_boxes, "frame_names": names}]}))
    tmp.rename(out)
    print(f"{a.tracker} {d['video']} s{d['imgsz']} d{delta}: {len(frames)} frames, "
          f"{len({i for f in fp_ids for i in f})} identities", flush=True)
