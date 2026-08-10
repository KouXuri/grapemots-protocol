#!/usr/bin/env python3
"""Compare annotated-rate and source-rate tracking on identical scoring frames.

Each video may use its own out-of-fold checkpoint. The tracker sees either every
source frame or the released annotated subsequence, while counts are always read
on the same annotated frames.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("GRAPEMOTS_ROOT", REPOSITORY)).resolve()
sys.path.insert(0, str(REPOSITORY / "tools"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402

from track_grapemots_mot import (  # noqa: E402
    build_tracker,
    load_gt_tracks,
    merge_detections,
    tiled_raw,
)

DEFAULT_DATA = WORKSPACE / "datasets/grapemots_det_721"
DEFAULT_VIDEO_ROOT = Path(
    os.environ.get("GRAPEMOTS_VIDEO_ROOT", WORKSPACE.parent / "MOTS2024")
).resolve()
DEFAULT_VIDEOS = ["PathPlanning_2", "PathPlanning_4"]
DEFAULT_WEIGHTS = WORKSPACE / "runs/detect/cbdcom2026/gm_ctrl_newsplit_oldcfg/weights/best.pt"
DEFAULT_LENGTHS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
DEFAULT_TAUS = [1, 2, 3, 5, 8]


def annotated_frames(video: str, data_root: Path) -> dict[int, Path]:
    out = {}
    track_root = data_root / "tracks"
    splits = ("all",) if (track_root / "all").is_dir() else ("train", "val", "test")
    for split in splits:
        for path in (track_root / split).glob(f"{video}__frame_*.txt"):
            match = re.search(r"__frame_(\d+)\.txt$", path.name)
            if match:
                out[int(match.group(1))] = path
    return dict(sorted(out.items()))


def source_video(video: str, video_root: Path) -> Path:
    candidates = {path.stem.casefold(): path for path in video_root.glob("*.mp4")}
    try:
        return candidates[video.casefold()]
    except KeyError as exc:
        raise FileNotFoundError(f"No source MP4 for {video} in {video_root}") from exc


def read_weights_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not all(
        isinstance(video, str) and isinstance(weights, str) for video, weights in raw.items()
    ):
        raise ValueError("--weights-map must map video names to checkpoint paths")
    return {video: Path(weights) for video, weights in raw.items()}


def surface(pred_ids, gt_ids, lengths, taus):
    grid = []
    seen = Counter()
    truth = set()
    wanted = {length for length in lengths if length <= len(pred_ids)} | {len(pred_ids)}
    snapshots = {}
    for index, (predicted, ground_truth) in enumerate(zip(pred_ids, gt_ids), start=1):
        seen.update(predicted)
        truth.update(ground_truth)
        if index in wanted:
            snapshots[index] = (seen.copy(), len(truth))
    for length in sorted(snapshots):
        counts, ground_truth = snapshots[length]
        for tau in taus:
            predicted = sum(observations >= tau for observations in counts.values())
            grid.append(
                {
                    "window_frames": length,
                    "min_track_len": tau,
                    "predicted_tracks": predicted,
                    "gt_tracks": ground_truth,
                    "signed_relative_error": (
                        (predicted - ground_truth) / ground_truth if ground_truth else None
                    ),
                }
            )
    return grid


def whole_sequence_cell(record: dict) -> dict:
    return next(
        cell
        for cell in record["count_error_surface"]
        if cell["window_frames"] == record["scored_frames"] and cell["min_track_len"] == 1
    )


def run(video: str, video_path: Path, weights: Path, step: int, model, args) -> dict:
    annotations = annotated_frames(video, args.root)
    if not annotations:
        raise ValueError(f"No annotations found for {video} under {args.root}")

    # The released frame numbering does not index the released video for every
    # sequence. NoPathPlanning_2 and _3 are numbered as if the three frontal
    # videos formed one continuous recording (offsets 1050 and 1950, verified by
    # pixel comparison: the aligned frame differs by ~2 grey levels while a
    # one-frame shift differs by ~45), and PathPlanning_1 is annotated over 934
    # frames while its MP4 holds 801, so for that one no offset recovers the
    # correspondence at all. Declaring the offset here is what makes the
    # source-rate comparison possible; guessing it would silently score the
    # wrong frames.
    offset = int(getattr(args, "frame_offsets", {}).get(video, 0))
    if offset:
        annotations = {number - offset: path for number, path in annotations.items()}
        if min(annotations) < 0:
            raise ValueError(f"{video}: offset {offset} pushes annotations before frame 0")

    # A constant offset only works where the release labelled every frame. On a
    # corpus that labelled roughly one frame in fifteen, the correspondence is a
    # measured map, not an arithmetic one, so an explicit mapping recovered by
    # tools/align_annotated_to_source.py takes precedence over any offset.
    mapping = getattr(args, "frame_map", {}).get(video)
    if mapping:
        rekeyed = {}
        for number, path in annotations.items():
            source = mapping.get(str(number), mapping.get(number))
            if source is None:
                raise ValueError(f"{video}: annotated frame {number} is not in the frame map")
            rekeyed[int(source)] = path
        if len(rekeyed) != len(annotations):
            raise ValueError(f"{video}: frame map is not injective")
        annotations = rekeyed

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = build_tracker(args.tracker, frame_rate=max(1, round(source_fps / step)))
    last_annotated_frame = max(annotations)
    pred_ids = []
    gt_ids = []
    source_index = 0
    tracker_frames = 0
    started = time.time()

    while source_index <= last_annotated_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % step == 0:
            height, width = frame.shape[:2]
            boxes, scores = tiled_raw(
                model, frame, args.imgsz, args.conf, args.tile, args.stride
            )
            boxes, scores = merge_detections(
                boxes, scores, args.merge_threshold, args.merge_metric
            )
            data = (
                np.concatenate(
                    [boxes, scores[:, None], np.zeros((len(boxes), 1), dtype=np.float32)],
                    axis=1,
                )
                if len(boxes)
                else np.empty((0, 6), dtype=np.float32)
            )
            tracks = tracker.update(
                Boxes(torch.as_tensor(data, dtype=torch.float32), (height, width)), frame
            )
            tracker_frames += 1
            if source_index in annotations:
                ground_truth, _ = load_gt_tracks(
                    annotations[source_index], width, height
                )
                pred_ids.append(
                    [int(track_id) for track_id in tracks[:, 4]] if len(tracks) else []
                )
                gt_ids.append(list(ground_truth))
        source_index += 1
    capture.release()

    if len(pred_ids) != len(annotations):
        raise RuntimeError(
            f"{video} step={step}: scored {len(pred_ids)} of {len(annotations)} annotations"
        )
    grid = surface(pred_ids, gt_ids, args.window_lengths, args.taus)
    record = {
        "video": video,
        "video_path": str(video_path),
        "weights": str(weights),
        "step": step,
        "source_fps": source_fps,
        "tracker_frames": tracker_frames,
        "scored_frames": len(pred_ids),
        "count_error_surface": grid,
    }
    whole = whole_sequence_cell(record)
    elapsed = time.time() - started
    print(
        f"{video} step={step}: tracker={tracker_frames}, scored={len(pred_ids)}, "
        f"tau=1 {whole['predicted_tracks']} vs {whole['gt_tracks']} "
        f"({whole['signed_relative_error']:+.4f}), {elapsed:.0f}s",
        flush=True,
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--weights-map", type=Path, help="JSON mapping each video to an out-of-fold checkpoint"
    )
    parser.add_argument(
        "--frame-map", type=Path,
        help="alignment report from tools/align_annotated_to_source.py; its "
             "sequences.<video>.annotated_to_source entry replaces the offset"
    )
    parser.add_argument(
        "--frame-offsets", type=Path,
        help="JSON mapping a video to the constant to subtract from its annotated "
             "frame numbers to obtain the index into its released MP4"
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--merge-metric", choices=["iou", "ios"], default="iou")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--window-lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--taus", type=int, nargs="+", default=DEFAULT_TAUS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/fullrate/fullrate_tracking.json"),
    )
    args = parser.parse_args()
    args.frame_offsets = (
        json.loads(args.frame_offsets.read_text()) if args.frame_offsets else {}
    )
    if args.frame_map:
        report = json.loads(args.frame_map.read_text())
        args.frame_map = {
            video: entry["annotated_to_source"]
            for video, entry in report.get("sequences", {}).items()
            if entry.get("status") == "aligned" and "annotated_to_source" in entry
        }
    else:
        args.frame_map = {}

    if 1 not in args.taus:
        parser.error("--taus must include 1")
    if not args.videos or any(step < 1 for step in args.steps):
        parser.error("Require at least one video and positive --steps")
    return args


def main() -> None:
    args = parse_args()
    weights_by_video = read_weights_map(args.weights_map)
    unknown = sorted(set(weights_by_video) - set(args.videos))
    if unknown:
        raise SystemExit(f"Weights supplied for unrequested videos: {', '.join(unknown)}")

    model_cache = {}
    records = []
    for step in args.steps:
        for video in args.videos:
            weights = weights_by_video.get(video, args.weights)
            if not weights.is_file():
                raise SystemExit(f"Missing checkpoint for {video}: {weights}")
            key = str(weights.resolve())
            if key not in model_cache:
                model_cache[key] = YOLO(key)
            records.append(
                run(
                    video,
                    source_video(video, args.video_root),
                    weights,
                    step,
                    model_cache[key],
                    args,
                )
            )

    pooled = {}
    for step in args.steps:
        predicted = ground_truth = 0
        for record in records:
            if record["step"] != step:
                continue
            cell = whole_sequence_cell(record)
            predicted += cell["predicted_tracks"]
            ground_truth += cell["gt_tracks"]
        pooled[str(step)] = {
            "predicted": predicted,
            "gt": ground_truth,
            "signed_relative_error": (predicted - ground_truth) / ground_truth,
        }
        print(f"POOLED step={step}: {predicted} vs {ground_truth} -> "
              f"{(predicted - ground_truth) / ground_truth:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args.out.write_text(
        json.dumps({"config": config, "runs": records, "pooled": pooled}, indent=1) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
