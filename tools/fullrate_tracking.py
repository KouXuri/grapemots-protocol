#!/usr/bin/env python3
"""Does the over-count survive when the tracker sees every source frame?

The released annotation is every second source frame for the two test videos, and
the paper's tracking therefore runs at an effective 15 Hz. A reviewer asked the
obvious question: how much of the measured over-count is the temporal
down-sampling rather than the association problem at the real 30 Hz frame rate?

This decodes PathPlanning_2 and PathPlanning_4 from the released MP4s, runs the
same tiled YOLO26s detector and the same BoT-SORT configuration on EVERY source
frame, and reads the count out only on the annotated frames, against the same
ground truth. Everything except the frame rate the tracker sees is held fixed.
"""
from __future__ import annotations
import os
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path

import cv2, numpy as np, torch

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools"))
from ultralytics import YOLO                                        # noqa: E402
from ultralytics.engine.results import Boxes                        # noqa: E402
from track_grapemots_mot import (tiled_raw, merge_detections,       # noqa: E402
                                 build_tracker, load_gt_tracks)

DATA = ROOT / "datasets/grapemots_det_721"
VIDEOS = {"PathPlanning_2": ROOT.parent / "MOTS2024/PathPlanning_2.mp4",
          "PathPlanning_4": ROOT.parent / "MOTS2024/PathPlanning_4.mp4"}
WEIGHTS = ROOT / "runs/detect/cbdcom2026/gm_ctrl_newsplit_oldcfg/weights/best.pt"
LENGTHS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 374]
TAUS = [1, 2, 3, 5, 8]


def annotated_frames(video: str) -> dict[int, Path]:
    out = {}
    for split in ("train", "val", "test"):
        for p in (DATA / "tracks" / split).glob(f"{video}__frame_*.txt"):
            out[int(re.search(r"__frame_(\d+)\.txt$", p.name).group(1))] = p
    return dict(sorted(out.items()))


def surface(pred_ids, gt_ids, lengths, taus):
    grid, total = [], len(pred_ids)
    seen, truth = Counter(), set()
    wanted = {n for n in lengths if n <= total} | {total}
    snaps = {}
    for i in range(total):
        seen.update(pred_ids[i]); truth.update(gt_ids[i])
        if i + 1 in wanted:
            snaps[i + 1] = (Counter(seen), len(truth))
    for n in sorted(snaps):
        counts, g = snaps[n]
        for tau in taus:
            p = sum(c >= tau for c in counts.values())
            grid.append({"window_frames": n, "min_track_len": tau,
                         "predicted_tracks": p, "gt_tracks": g,
                         "signed_relative_error": (p - g) / g if g else None})
    return grid


def run(video: str, step: int, model, args) -> dict:
    """step=1 -> every source frame; step=2 -> the released annotated subsequence."""
    ann = annotated_frames(video)
    last = max(ann)
    cap = cv2.VideoCapture(str(VIDEOS[video]))
    tracker = build_tracker(args.tracker, frame_rate=int(round(30 / step)))
    pred_ids, gt_ids = [], []
    idx, t0, seen_frames = 0, time.time(), 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            boxes, scores = tiled_raw(model, frame, args.imgsz, args.conf,
                                      args.tile, args.stride)
            boxes, scores = merge_detections(boxes, scores, args.merge_threshold, "iou")
            data = (np.concatenate([boxes, scores[:, None],
                                    np.zeros((len(boxes), 1), np.float32)], axis=1)
                    if len(boxes) else np.empty((0, 6), np.float32))
            tracks = tracker.update(
                Boxes(torch.as_tensor(data, dtype=torch.float32), (h, w)), frame)
            ids = [int(t) for t in tracks[:, 4]] if len(tracks) else []
            seen_frames += 1
            if idx in ann:                      # score only where truth exists
                g, _ = load_gt_tracks(ann[idx], w, h)
                pred_ids.append(ids); gt_ids.append(list(g))
        idx += 1
    cap.release()
    grid = surface(pred_ids, gt_ids, LENGTHS, TAUS)
    whole = grid[-len(TAUS)]
    print(f"  {video} step={step}: tracker saw {seen_frames} frames, scored on "
          f"{len(pred_ids)}, whole-sequence tau=1 error "
          f"{whole['signed_relative_error']:+.4f} "
          f"({whole['predicted_tracks']} vs {whole['gt_tracks']}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return {"video": video, "step": step, "tracker_frames": seen_frames,
            "scored_frames": len(pred_ids), "count_error_surface": grid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=960)
    ap.add_argument("--merge-threshold", type=float, default=0.5)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/fullrate_tracking.json")
    args = ap.parse_args()

    model = YOLO(str(WEIGHTS))
    records = []
    for step in args.steps:
        for video in VIDEOS:
            records.append(run(video, step, model, args))
    # pooled whole-sequence error at tau=1, the number Table IV reports
    pooled = {}
    for step in args.steps:
        p = g = 0
        for r in records:
            if r["step"] != step:
                continue
            cell = [c for c in r["count_error_surface"] if c["min_track_len"] == 1][-1]
            p += cell["predicted_tracks"]; g += cell["gt_tracks"]
        pooled[step] = {"predicted": p, "gt": g, "signed_relative_error": (p - g) / g}
        print(f"POOLED step={step}: {p} vs {g} -> {(p-g)/g:+.4f}")
    args.out.write_text(json.dumps({"config": vars(args) | {"out": str(args.out)},
                                    "runs": records, "pooled": pooled},
                                   indent=1, default=str) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
