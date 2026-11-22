#!/usr/bin/env python3
"""Oracle StrongSORT: reference boxes as detections, appearance association (raptor, CPU).

Question: does an associator that can re-link by appearance, without box overlap, remove
the surplus identities that BoT-SORT issues on multi-view flights even with perfect
detections? Nothing in the tracker is new: tools/strongsort_track_0920.py (BoxMOT
StrongSORT, OSNet-x0.25 MSMT17, min_conf 0.25) runs unchanged. This script only writes,
per video, a detection file in dump_dets_0920.py's schema whose boxes are the reference
boxes (load_gt_tracks, the reference every arm is scored against) at score 0.9, the
oracle_cadence_1119 convention, then calls the tracker at intervals 1, 2, 4, 8 (phase 0).

    nice -n 19 .venv/bin/python tools/oracle_strongsort_1120.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / "tools"))
from track_grapemots_mot import load_gt_tracks  # noqa: E402

DATA = ROOT / "datasets/grapemots_det_721"
OUT = ROOT / "runs/error_sources_1120/oracle_strongsort"
BOX = "/home/kou/my_env/boxmot_venv/bin/python"
VIDEOS = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]


def main():
    (OUT / "dets").mkdir(parents=True, exist_ok=True)
    for v in VIDEOS:
        dets = OUT / "dets" / f"{v}_s0.json"
        if not dets.exists():
            frames = sorted(p for p in (DATA / "images/all").iterdir() if p.name.split("__")[0] == v)
            h, w = cv2.imread(str(frames[0])).shape[:2]
            rows = []
            for p in frames:
                gi, gb = load_gt_tracks(DATA / "tracks/all" / f"{p.stem}.txt", w, h)
                rows.append({"name": p.name, "path": str(p), "boxes": gb.tolist(), "scores": [0.9] * len(gb),
                             "gt_ids": [int(i) for i in gi], "gt_boxes": gb.tolist()})
            dets.write_text(json.dumps({"weights": "oracle (reference boxes, score 0.9)", "video": v,
                                        "imgsz": 0, "conf": 0.9, "frames": rows}))
        subprocess.run(f"OMP_NUM_THREADS=2 {BOX} tools/strongsort_track_0920.py --dets {dets} "
                       f"--out-dir {OUT}/tracks --device cpu", shell=True, cwd=ROOT, check=True)
    (OUT / "status.json").write_text(json.dumps({"state": "completed"}))


if __name__ == "__main__":
    main()
