#!/usr/bin/env python3
"""Association alone: the processing interval with ground-truth boxes as detections.

No detector. Every annotated frame's reference boxes (tracks/all sidecars, the
reference every main-study arm is scored against) are handed to the tracker as
detections at score 0.9, so every bunch is detected in every processed frame and
any surplus or loss is the associator's. The tracker and its settings are the main
study's: BoT-SORT, cfg/trackers/botsort_gmc.yaml, sparse-optical-flow GMC on the
REAL frames (oracle_cmc_check.py showed GMC matters: median oracle error 0.94 with
GMC off, 0.30 with it on). ByteTrack (bytetrack.yaml, no imagery) is the second arm.

Grid: all eleven GrapeMOTS videos (no training, so no split and no leakage), interval
1, 2, 4, 8 annotated frames, every phase 0..interval-1. The tracker receives frames
offset, offset+interval, ...; output layout is track_grapemots_mot.py's, so the same
decomposition, matched-design and TrackEval tools read it.

Runs on raptor, where the frames are. Light by design: one process, nice 19, idle
I/O class (set by the launcher), and it waits whenever MemAvailable < 2 GiB so it can
never press the disease training sharing the machine.

    nice -n 19 ionice -c3 .venv/bin/python tools/oracle_cadence_1119.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / "tools"))
from ultralytics.engine.results import Boxes  # noqa: E402
from oracle_count_surface import read_resolutions, video_frames  # noqa: E402
from track_grapemots_mot import build_tracker, load_gt_tracks  # noqa: E402

DATA = ROOT / "datasets/grapemots_det_721"
IMAGES = DATA / "images/all"
OUT = Path(sys.argv[1])
DELTAS = (1, 2, 4, 8)
ARMS = (("botsort_gmc", "cfg/trackers/botsort_gmc.yaml", True),
        ("bytetrack", "bytetrack.yaml", False))
torch.set_num_threads(1)
cv2.setNumThreads(1)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def wait_for_memory(floor=2.0):
    while mem_available_gib() < floor:
        time.sleep(30)


def run(frames, size, cfg, use_images):
    width, height = size
    tracker = build_tracker(cfg, frame_rate=30)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    rec = {k: [] for k in ("frame_names", "frame_gt_ids", "frame_gt_boxes",
                           "frame_predicted_ids", "frame_predicted_boxes")}
    for path in frames:
        gt_ids, gt_boxes = load_gt_tracks(path, width, height)
        if use_images:
            image = cv2.imread(str(IMAGES / (path.stem + ".PNG")))
            if image is None:
                raise SystemExit(f"missing image for {path.stem}")
        else:
            image = blank
        scores = np.full((len(gt_boxes), 1), 0.9, dtype=np.float32)
        classes = np.zeros((len(gt_boxes), 1), dtype=np.float32)
        data = (np.concatenate([np.asarray(gt_boxes, np.float32), scores, classes], axis=1)
                if len(gt_boxes) else np.empty((0, 6), dtype=np.float32))
        tracks = tracker.update(Boxes(torch.as_tensor(data), (height, width)), image)
        rec["frame_names"].append(path.stem + ".PNG")
        rec["frame_gt_ids"].append([int(i) for i in gt_ids])
        rec["frame_gt_boxes"].append([[float(x) for x in b] for b in np.asarray(gt_boxes).reshape(-1, 4)])
        rec["frame_predicted_ids"].append([int(t) for t in tracks[:, 4]] if len(tracks) else [])
        rec["frame_predicted_boxes"].append([[float(x) for x in t[:4]] for t in tracks] if len(tracks) else [])
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = read_resolutions(DATA)
    videos = sorted(sizes)
    status = OUT / "status.json"
    for name, cfg, use_images in ARMS:
        for video in videos:
            frames_all = video_frames(DATA, video)
            for d in DELTAS:
                for k in range(d):
                    dest = OUT / name / f"{video}_d{d}_o{k}.json"
                    if dest.exists():
                        continue
                    wait_for_memory()
                    t0 = time.time()
                    rec = run(frames_all[k::d], sizes[video], cfg, use_images)
                    rec["video"] = video
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps({"config": {"tracker": cfg, "images": use_images, "interval": d,
                                                          "offset": k, "detections": "reference boxes, score 0.9"},
                                               "videos": [rec]}))
                    tmp.rename(dest)
                    status.write_text(json.dumps({"arm": name, "video": video, "interval": d, "offset": k,
                                                  "seconds": round(time.time() - t0, 1),
                                                  "updated": time.strftime("%Y-%m-%d %H:%M:%S")}))
                    print(f"{name} {video} d{d} o{k}: {len(rec['frame_names'])} frames, "
                          f"{round(time.time() - t0, 1)} s", flush=True)
    status.write_text(json.dumps({"state": "completed", "updated": time.strftime("%Y-%m-%d %H:%M:%S")}))


if __name__ == "__main__":
    main()
