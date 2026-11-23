#!/usr/bin/env python3
"""Association alone on two other agricultural MOTS datasets (external check, ledger 10.28.16).

Question, written before the run: does the surplus of identities that five associators issue on
GrapeMOTS with perfect detection appear on other crops and platforms, and is it larger where
objects leave the view and come back, as the GrapeMOTS duplicate classes imply?

  APPLE MOTS (de Jong et al., RA-L 2022; Zenodo 10.5281/zenodo.5939726, CC BY 4.0): apple trees
      filmed from a UAV or a wearable camera in single passes along the row; sequences 0000-0008
      (0010-0012 repeat 0006-0008 in the release and are left out).
  LettuceMOTS (Hu et al., J. Field Robotics 2024; github.com/NanH5837/LettuceMOTS): a downward
      camera on a field robot; six sequences in which the robot passes the same plants again and
      six single passes. All twelve sequences.

Nothing in the trackers is new and every setting is the GrapeMOTS oracle's: the reference boxes
(load_gt_tracks on the track sidecars written by create_grapemots_detection_dataset.py, min area
20 px) are handed over at score 0.9 in every processed frame;
  BoT-SORT   Ultralytics 8.4.46, cfg/trackers/botsort_gmc.yaml, sparse-optical-flow GMC on the
             real frames (as tools/oracle_cadence_1119.py)
  ByteTrack  Ultralytics bytetrack.yaml, no imagery (as tools/oracle_cadence_1119.py)
  StrongSORT tools/strongsort_track_0920.py (BoxMOT 23.0.0, OSNet-x0.25 MSMT17), CPU
  OC-SORT, Deep OC-SORT   tools/boxmot_track_1120.py, CPU
at intervals 1, 2, 4, 8 annotated frames, phase 0. No detector is involved, so no split.

    nice -n 19 .venv/bin/python tools/oracle_external_1122.py [apple|lettuce ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / "tools"))
from ultralytics.engine.results import Boxes  # noqa: E402
from track_grapemots_mot import build_tracker, load_gt_tracks  # noqa: E402

OUT = ROOT / "runs/external_1122"
BOX = "/home/kou/my_env/boxmot_venv/bin/python"
DELTAS = (1, 2, 4, 8)
DATASETS = {
    "apple": (ROOT / "datasets/applemots_det", [f"{i:04d}" for i in range(9)]),
    "lettuce": (ROOT / "datasets/lettucemots_det", [f"{i:04d}" for i in range(12)]),
}
ULTRA = (("botsort_gmc", "cfg/trackers/botsort_gmc.yaml", True), ("bytetrack", "bytetrack.yaml", False))
torch.set_num_threads(2)
cv2.setNumThreads(2)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def wait_for_memory(floor=3.0):
    while mem_available_gib() < floor:
        time.sleep(30)


def reference_dets(ds, root, video):
    """dump_dets_0920.py schema with the reference boxes at score 0.9 (oracle_strongsort_1120.py)."""
    dest = OUT / ds / "dets" / f"{video}_s0.json"
    if dest.exists():
        return dest
    frames = []
    for split in ("train", "val", "test"):
        d = root / "images" / split
        if d.exists():
            frames += [(split, p) for p in d.iterdir() if p.name.split("__")[0] == video]
    frames.sort(key=lambda sp: sp[1].name)
    assert frames, (ds, video)
    h, w = cv2.imread(str(frames[0][1])).shape[:2]
    rows = []
    for split, p in frames:
        gi, gb = load_gt_tracks(root / "tracks" / split / f"{p.stem}.txt", w, h)
        gb = np.asarray(gb, np.float32).reshape(-1, 4)
        rows.append({"name": p.name, "path": str(p), "boxes": gb.tolist(), "scores": [0.9] * len(gb),
                     "gt_ids": [int(i) for i in gi], "gt_boxes": gb.tolist()})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"weights": "oracle (reference boxes, score 0.9)", "video": video,
                                "imgsz": 0, "conf": 0.9, "size": [w, h], "frames": rows}))
    return dest


def run_ultralytics(frames, size, cfg, use_images):
    """tools/oracle_cadence_1119.py run(), fed from the detection file."""
    width, height = size
    tracker = build_tracker(cfg, frame_rate=30)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    rec = {k: [] for k in ("frame_names", "frame_gt_ids", "frame_gt_boxes",
                           "frame_predicted_ids", "frame_predicted_boxes")}
    for f in frames:
        image = cv2.imread(f["path"]) if use_images else blank
        if image is None:
            raise SystemExit(f"missing image {f['path']}")
        gb = np.asarray(f["gt_boxes"], np.float32).reshape(-1, 4)
        scores = np.full((len(gb), 1), 0.9, dtype=np.float32)
        classes = np.zeros((len(gb), 1), dtype=np.float32)
        data = np.concatenate([gb, scores, classes], axis=1) if len(gb) else np.empty((0, 6), np.float32)
        tracks = tracker.update(Boxes(torch.as_tensor(data), (height, width)), image)
        rec["frame_names"].append(f["name"])
        rec["frame_gt_ids"].append([int(i) for i in f["gt_ids"]])
        rec["frame_gt_boxes"].append([[float(x) for x in b] for b in gb])
        rec["frame_predicted_ids"].append([int(t) for t in tracks[:, 4]] if len(tracks) else [])
        rec["frame_predicted_boxes"].append([[float(x) for x in t[:4]] for t in tracks] if len(tracks) else [])
    return rec


def main(which):
    for ds in which:
        root, videos = DATASETS[ds]
        for video in videos:
            dets = reference_dets(ds, root, video)
            d = json.loads(dets.read_text())
            size = tuple(d["size"])
            for name, cfg, use_images in ULTRA:
                for delta in DELTAS:
                    dest = OUT / ds / name / f"{video}_d{delta}.json"
                    if dest.exists():
                        continue
                    wait_for_memory()
                    t0 = time.time()
                    rec = run_ultralytics(d["frames"][::delta], size, cfg, use_images)
                    rec["video"] = video
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"config": {"tracker": cfg, "images": use_images, "interval": delta,
                                                          "offset": 0, "detections": "reference boxes, score 0.9"},
                                               "videos": [rec]}))
                    tmp.rename(dest)
                    print(f"{ds} {name} {video} d{delta}: {len(rec['frame_names'])} frames, "
                          f"{len({i for f in rec['frame_predicted_ids'] for i in f})} ids, "
                          f"{time.time() - t0:.1f} s", flush=True)
            for trk, script in (("strongsort", "tools/strongsort_track_0920.py"),
                                ("ocsort", "tools/boxmot_track_1120.py --tracker ocsort"),
                                ("deepocsort", "tools/boxmot_track_1120.py --tracker deepocsort")):
                outdir = OUT / ds / trk
                if all((outdir / f"{video}_s0_d{dl}.json").exists() for dl in DELTAS):
                    continue
                wait_for_memory()
                subprocess.run(f"OMP_NUM_THREADS=2 {BOX} {script} --dets {dets} --out-dir {outdir} --device cpu",
                               shell=True, cwd=ROOT, check=True)
    (OUT / f"oracle_status_{'_'.join(which)}.json").write_text(
        json.dumps({"state": "completed", "updated": time.strftime("%Y-%m-%d %H:%M:%S")}))


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
