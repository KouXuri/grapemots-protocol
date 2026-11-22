#!/usr/bin/env python3
"""Dump the full-frame detections of one video at one scale, for trackers run elsewhere.

The detections are produced by the SAME functions track_grapemots_mot.py uses
(resize_raw, then merge_detections with IoU 0.5 / NMS, confidence 0.25), so a
tracker fed from this file sees exactly the boxes BoT-SORT saw in the surface.
Every annotated frame is dumped; the tracker subsamples by interval itself.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import cv2
sys.path.insert(0, "tools")
from track_grapemots_mot import load_gt_tracks, merge_detections, resize_raw  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--split", default="all"); ap.add_argument("--video", required=True)
ap.add_argument("--imgsz", type=int, required=True); ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
frames = sorted(p for p in (a.root / "images" / a.split).iterdir()
                if p.name.split("__")[0] == a.video and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
model = YOLO(a.weights)
rows = []
for p in frames:
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    rb, rs = resize_raw(model, img, a.imgsz, a.conf)
    b, s = merge_detections(rb, rs, 0.5, "iou", fusion="nms")
    gi, gb = load_gt_tracks(a.root / "tracks" / a.split / f"{p.stem}.txt", w, h)
    rows.append({"name": p.name, "path": str(p), "boxes": b.tolist(), "scores": s.tolist(),
                 "gt_ids": list(gi), "gt_boxes": gb.tolist()})
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"weights": a.weights, "video": a.video, "imgsz": a.imgsz, "conf": a.conf,
                             "frames": rows}))
print(f"{a.video} s{a.imgsz}: {len(rows)} frames, {sum(len(r['boxes']) for r in rows)} detections")
