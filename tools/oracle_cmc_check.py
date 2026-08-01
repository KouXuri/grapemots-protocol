#!/usr/bin/env python3
"""Does the oracle over-count survive camera-motion compensation?

Review objection: the oracle feeds ground-truth boxes to a tracker that never
sees imagery, so global motion compensation is off. On a moving UAV that is
exactly where a pure Kalman associator is weakest, so the measured over-count
could be a property of an under-configured tracker rather than of association.

This runs the p=0 oracle three ways on all eleven sequences with one identical
downstream analysis: the ByteTrack arm the paper reports, BoT-SORT with GMC off
(blank frames, as before), and BoT-SORT with sparseOptFlow GMC on real frames.
Summary statistic is the frozen one: retained cell = tau <= L/2 and
coverage = G(L)/G(full) >= 0.8; per video take the median retained cell; headline
is the median over the eleven per-video values.
"""
from __future__ import annotations
import os
import json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch
import cv2

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools"))
from ultralytics.engine.results import Boxes  # noqa: E402
from oracle_count_surface import (video_frames, read_resolutions,  # noqa: E402
                                  prefix_surface)
from track_grapemots_mot import build_tracker, load_gt_tracks  # noqa: E402

LENGTHS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300,
           350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
TAUS = [1, 2, 3, 5, 8]
DATA = ROOT / "datasets/grapemots_det_721"
IMAGES = DATA / "images/all"


def run(frames, size, cfg, use_images):
    width, height = size
    tracker = build_tracker(cfg, frame_rate=15)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    pred, truth = [], []
    for path in frames:
        gt_ids, gt_boxes = load_gt_tracks(path, width, height)
        if use_images:
            image = cv2.imread(str(IMAGES / (path.stem + ".PNG")))
            if image is None:
                raise SystemExit(f"missing image for {path.stem}")
        else:
            image = blank
        boxes = gt_boxes
        scores = np.full((len(boxes), 1), 0.9, dtype=np.float32)
        classes = np.zeros((len(boxes), 1), dtype=np.float32)
        data = (np.concatenate([boxes.astype(np.float32), scores, classes], axis=1)
                if len(boxes) else np.empty((0, 6), dtype=np.float32))
        tracks = tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32),
                                      (height, width)), image)
        pred.append([int(t) for t in tracks[:, 4]] if len(tracks) else [])
        truth.append(list(gt_ids))
    return pred, truth


def retained(grid, coverage=0.8):
    cells = [c for c in grid if c["min_track_len"] <= c["window_frames"] / 2 and c["gt_tracks"]]
    if not cells:
        return []
    full = max(c["gt_tracks"] for c in cells)
    return [c for c in cells if c["gt_tracks"] / full >= coverage]


def main():
    sizes = read_resolutions(DATA)
    videos = sorted(sizes)
    arms = [("bytetrack.yaml", False, "bytetrack, no imagery (paper baseline)"),
            ("cfg/trackers/botsort_nogmc.yaml", False, "BoT-SORT, GMC off"),
            ("cfg/trackers/botsort_gmc.yaml", True, "BoT-SORT, GMC on, real frames")]
    out = {}
    for cfg, use_images, label in arms:
        per_video, negatives, started = {}, [], time.time()
        for video in videos:
            frames = video_frames(DATA, video)
            pred, truth = run(frames, sizes[video], cfg, use_images)
            grid = prefix_surface(pred, truth, LENGTHS, TAUS)
            cells = retained(grid)
            errs = [(c["predicted_tracks"] - c["gt_tracks"]) / c["gt_tracks"] for c in cells]
            per_video[video] = statistics.median(errs)
            if any(e < 0 for e in errs):
                negatives.append(video)
            print(f"  [{label}] {video}: {len(frames)} frames, "
                  f"{len(cells)} retained, median {per_video[video]:+.4f}", flush=True)
        head = statistics.median(per_video.values())
        out[label] = {"cfg": cfg, "images": use_images,
                      "median_of_video_medians": head,
                      "videos_with_a_negative_cell": negatives,
                      "per_video_median": {k: round(v, 4) for k, v in per_video.items()},
                      "seconds": round(time.time() - started, 1)}
        print(f"== {label}: median-of-medians {head:+.4f}, "
              f"{len(negatives)} videos with a negative cell, "
              f"{out[label]['seconds']}s", flush=True)
    dest = ROOT / "results/oracle_cmc_check.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
