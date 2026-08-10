#!/usr/bin/env python3
"""Evaluate tiled detection, merge and tracking on the frozen GrapeMOTS split."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import motmetrics as mm
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# stand-in for "no possible match" in the assignment cost matrix
LARGE_COST = 1e6
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

# motmetrics 1.4 still calls np.asfarray(), which NumPy 2 removed.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda values: np.asarray(values, dtype=float)  # type: ignore[attr-defined]


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tiled_raw(model, image, imgsz: int, conf: float, tile: int, stride: int):
    height, width = image.shape[:2]
    boxes, scores = [], []
    for y0 in tile_starts(height, tile, stride):
        for x0 in tile_starts(width, tile, stride):
            result = model.predict(
                image[y0:y0 + tile, x0:x0 + tile], imgsz=imgsz, conf=conf, verbose=False
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy().copy()
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            boxes.append(xyxy)
            scores.append(result.boxes.conf.cpu().numpy())
    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(boxes).astype(np.float32), np.concatenate(scores).astype(np.float32)


def resize_raw(model, image, imgsz: int, conf: float):
    """One inference over the whole letterboxed frame -- the control detector.

    Needed for the counting-surface arm that swaps the detector: if the drift
    also appears with a detector that never tiles, then tile-boundary duplicates
    cannot be what causes it.
    """
    result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return (result.boxes.xyxy.cpu().numpy().astype(np.float32),
            result.boxes.conf.cpu().numpy().astype(np.float32))


def merge_detections(boxes: np.ndarray, scores: np.ndarray, threshold: float, metric: str):
    """Greedy NMS using intersection-over-union or intersection-over-smaller."""
    if not len(boxes):
        return boxes, scores
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32)
    order = score_tensor.argsort(descending=True)
    areas = (tensor[:, 2] - tensor[:, 0]).clamp(min=0) * (tensor[:, 3] - tensor[:, 1]).clamp(min=0)
    keep: list[int] = []
    while len(order):
        index = int(order[0])
        keep.append(index)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(tensor[index, 0], tensor[rest, 0])
        yy1 = torch.maximum(tensor[index, 1], tensor[rest, 1])
        xx2 = torch.minimum(tensor[index, 2], tensor[rest, 2])
        yy2 = torch.minimum(tensor[index, 3], tensor[rest, 3])
        intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        if metric == "iou":
            denominator = areas[index] + areas[rest] - intersection
        else:
            denominator = torch.minimum(areas[index].expand_as(areas[rest]), areas[rest])
        overlap = intersection / denominator.clamp(min=1e-6)
        order = rest[overlap <= threshold]
    return boxes[keep], scores[keep]


def load_gt_tracks(path: Path, width: int, height: int):
    ids, boxes = [], []
    if not path.is_file():
        return ids, np.empty((0, 4), dtype=np.float32)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        _cls, track_id, xc, yc, box_width, box_height = map(float, parts)
        xc, yc = xc * width, yc * height
        box_width, box_height = box_width * width, box_height * height
        ids.append(int(track_id))
        boxes.append([xc - box_width / 2, yc - box_height / 2,
                      xc + box_width / 2, yc + box_height / 2])
    return ids, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def effective_fps(frames: list[Path], source_fps: float = 30.0) -> int:
    indices = []
    for path in frames:
        match = re.search(r"(\d+)$", path.stem)
        if match:
            indices.append(int(match.group(1)))
    deltas = [right - left for left, right in zip(indices, indices[1:]) if right > left]
    step = median(deltas) if deltas else 1
    return max(1, round(source_fps / step))


def build_tracker(config: str, frame_rate: int):
    settings = IterableSimpleNamespace(**YAML.load(check_yaml(config)))
    if settings.tracker_type == "botsort":
        return BOTSORT(settings, frame_rate=frame_rate)
    if settings.tracker_type == "bytetrack":
        return BYTETracker(settings, frame_rate=frame_rate)
    raise ValueError(f"Unsupported tracker type: {settings.tracker_type}")


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    if len(converted):
        converted[:, 2] -= converted[:, 0]
        converted[:, 3] -= converted[:, 1]
    return converted


def serialise_metric(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def count_over_windows(
    frame_pred_ids: list[list[int]],
    frame_gt_ids: list[list[int]],
    window: int,
    min_len: int,
    max_windows: int = 64,
) -> dict | None:
    """Count unique tracks inside every window of `window` consecutive frames.

    Unique-track counting is only well defined relative to a time span: predicted
    identities accumulate as the sequence grows (each ID break adds one), while
    the ground-truth bunch count saturates once the drone has passed the vines.
    Sweeping the window length therefore shows whether a reported counting error
    reflects the method or merely the length of clip that was evaluated.

    `min_len` is re-applied inside each window, so a track only counts when it is
    seen at least `min_len` times within that window.
    """
    total = len(frame_pred_ids)
    if window > total:
        return None

    starts = list(range(0, total - window + 1))
    if max_windows and len(starts) > max_windows:  # 0 enumerates every start
        stride = len(starts) / max_windows
        starts = [starts[int(i * stride)] for i in range(max_windows)]

    errors, pred_counts, gt_counts = [], [], []
    for start in starts:
        pred_seen: Counter[int] = Counter()
        gt_seen: set[int] = set()
        for index in range(start, start + window):
            pred_seen.update(frame_pred_ids[index])
            gt_seen.update(frame_gt_ids[index])
        predicted = sum(seen >= min_len for seen in pred_seen.values())
        truth = len(gt_seen)
        pred_counts.append(predicted)
        gt_counts.append(truth)
        if truth:
            errors.append((predicted - truth) / truth)

    return {
        "window_frames": window,
        "windows_evaluated": len(starts),
        "mean_predicted_tracks": sum(pred_counts) / len(pred_counts),
        "mean_gt_tracks": sum(gt_counts) / len(gt_counts),
        "mean_signed_relative_error": sum(errors) / len(errors) if errors else None,
        "min_signed_relative_error": min(errors) if errors else None,
        "max_signed_relative_error": max(errors) if errors else None,
        # Prefix window: what you would report after flying this many frames.
        "prefix_predicted_tracks": pred_counts[0],
        "prefix_gt_tracks": gt_counts[0],
        "prefix_signed_relative_error": errors[0] if errors else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--root", type=Path, default=Path("datasets/grapemots_det_721"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--videos", nargs="+")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--merge", choices=["iou", "ios"], default="iou")
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--detector-mode", choices=["tiled", "resize"], default="tiled",
                        help="'resize' runs one inference per whole frame instead of "
                             "tiling; use it with the full-frame control model")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--window-lengths", type=int, nargs="+",
                        default=[10, 20, 50, 100, 200, 300, 400],
                        help="sequence lengths for the counting-vs-span sweep")
    parser.add_argument("--min-track-lens", type=int, nargs="+", default=[1, 2, 3, 5, 8],
                        help="track-length thresholds for the counting-error surface")
    parser.add_argument("--max-windows", type=int, default=64,
                        help="sliding starts sampled per window length; 0 uses every "
                             "legal start, which the protocol section has to declare")
    parser.add_argument("--save-frame-boxes", action="store_true",
                        help="also store per-frame predicted/GT boxes and frame "
                             "names, for qualitative figures; large output")
    parser.add_argument("--save-frame-tracks", action="store_true",
                        help="store per-frame predicted/GT track ids for re-analysis")
    parser.add_argument(
        "--dump-detections", type=Path,
        help="write the RAW per-tile detections to this gzipped JSON. They are "
             "cached at --conf, so build the cache at the lowest confidence any "
             "later arm will ask for")
    parser.add_argument(
        "--load-detections", type=Path,
        help="read raw detections from a cache instead of running the detector. "
             "Confidence and the merge rule are still applied here, so one cache "
             "serves every operating point at or above the confidence it was built "
             "with. Images are still read, because global motion compensation and "
             "appearance re-identification look at them")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1 or not 0 <= args.conf <= 1:
        raise SystemExit("Require --frame-step >= 1 and 0 <= --conf <= 1")
    image_dir = args.root / "images" / args.split
    track_dir = args.root / "tracks" / args.split
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            grouped[path.name.split("__")[0]].append(path)
    if args.videos:
        grouped = {video: grouped[video] for video in args.videos if video in grouped}
    if not grouped:
        raise SystemExit(f"No videos found in {image_dir}")

    # A cached run reproduces the detector exactly, so the cache is only accepted
    # when every field that changes what the detector produced still agrees. The
    # confidence is the one exception: filtering a cache further is sound, while
    # asking it for detections below the threshold it was built with is not.
    cache = None
    if args.load_detections:
        with gzip.open(args.load_detections, "rt") as handle:
            cache = json.load(handle)
        cached_conf = float(cache["config"]["conf"])
        if args.conf < cached_conf - 1e-9:
            raise SystemExit(
                f"cache was built at confidence {cached_conf} and cannot serve the "
                f"requested {args.conf}; rebuild it lower")
        for key in ("tile", "stride", "imgsz", "detector_mode"):
            if str(cache["config"][key]) != str(getattr(args, key)):
                raise SystemExit(
                    f"cache {key}={cache['config'][key]} does not match requested "
                    f"{getattr(args, key)}")
        if str(Path(args.weights).resolve()) != cache["config"]["weights"]:
            raise SystemExit("cache was built with a different checkpoint")
    dump = {} if args.dump_detections else None

    model = YOLO(args.weights) if cache is None else None
    metric_names = [
        "idf1", "mota", "num_switches", "num_fragmentations", "precision", "recall",
        "num_false_positives", "num_misses", "num_objects", "num_predictions",
    ]
    accumulators, names, rows = [], [], []
    for video, all_frames in sorted(grouped.items()):
        frames = all_frames[::args.frame_step]
        if args.limit:
            frames = frames[:args.limit]
        frame_rate = max(1, round(effective_fps(all_frames) / args.frame_step))
        tracker = build_tracker(args.tracker, frame_rate)
        accumulator = mm.MOTAccumulator(auto_id=True)
        predicted_lengths: Counter[int] = Counter()
        frame_pred_boxes: list[list[list[float]]] = []
        frame_gt_boxes: list[list[list[float]]] = []
        frame_names: list[str] = []
        gt_tracks_seen: set[int] = set()
        gt_to_pred: dict[int, set[int]] = defaultdict(set)
        matched_pred_ids: set[int] = set()
        frame_pred_ids: list[list[int]] = []
        frame_gt_ids: list[list[int]] = []
        elapsed = 0.0

        for image_path in frames:
            image = cv2.imread(str(image_path))
            if image is None:
                raise SystemExit(f"Could not read {image_path}")
            height, width = image.shape[:2]
            gt_ids, gt_boxes = load_gt_tracks(track_dir / f"{image_path.stem}.txt", width, height)
            gt_tracks_seen.update(gt_ids)

            start = time.perf_counter()
            if cache is not None:
                entry = cache["videos"][video][image_path.name]
                raw_boxes = np.asarray(entry["boxes"], dtype=np.float32).reshape(-1, 4)
                raw_scores = np.asarray(entry["scores"], dtype=np.float32)
                keep = raw_scores >= args.conf
                raw_boxes, raw_scores = raw_boxes[keep], raw_scores[keep]
            elif args.detector_mode == "resize":
                raw_boxes, raw_scores = resize_raw(model, image, args.imgsz, args.conf)
            else:
                raw_boxes, raw_scores = tiled_raw(
                    model, image, args.imgsz, args.conf, args.tile, args.stride
                )
            if dump is not None:
                dump.setdefault(video, {})[image_path.name] = {
                    "boxes": raw_boxes.tolist(),
                    "scores": raw_scores.tolist(),
                }
            merged_boxes, merged_scores = merge_detections(
                raw_boxes, raw_scores, args.merge_threshold, args.merge
            )
            data = np.concatenate(
                [merged_boxes, merged_scores[:, None], np.zeros((len(merged_boxes), 1), dtype=np.float32)], axis=1
            ) if len(merged_boxes) else np.empty((0, 6), dtype=np.float32)
            tracks = tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32), image.shape[:2]), image)
            elapsed += time.perf_counter() - start

            if len(tracks):
                predicted_boxes = np.asarray(tracks[:, :4], dtype=np.float32)
                predicted_ids = [int(track_id) for track_id in tracks[:, 4]]
            else:
                predicted_boxes = np.empty((0, 4), dtype=np.float32)
                predicted_ids = []
            predicted_lengths.update(predicted_ids)
            frame_pred_ids.append(list(predicted_ids))
            frame_gt_ids.append(list(gt_ids))
            if args.save_frame_boxes:
                frame_pred_boxes.append(predicted_boxes.tolist())
                frame_gt_boxes.append(gt_boxes.tolist())
                frame_names.append(image_path.name)
            distances = mm.distances.iou_matrix(
                xyxy_to_xywh(gt_boxes), xyxy_to_xywh(predicted_boxes), max_iou=1 - args.match_iou
            )
            accumulator.update(gt_ids, predicted_ids, distances)

            # Duplicate detections do not necessarily raise the standard
            # Fragmentations count: a bunch covered by two parallel predicted
            # tracks for its whole life produces no break in either of them.
            # What over-counting actually needs is the number of distinct
            # predicted identities attached to one GT identity, plus the
            # predicted tracks that never matched anything at all.
            if distances.size and np.isfinite(distances).any():
                cost = np.where(np.isfinite(distances), distances, LARGE_COST)
                for row, col in zip(*linear_sum_assignment(cost)):
                    if np.isfinite(distances[row, col]):
                        gt_to_pred[gt_ids[row]].add(predicted_ids[col])
                        matched_pred_ids.add(predicted_ids[col])

        metrics = mm.metrics.create().compute(
            accumulator, metrics=metric_names, name=video
        ).loc[video].to_dict()

        # How many distinct predicted identities ended up on one GT bunch, and
        # how many predicted tracks never matched any GT at all. Both feed the
        # over-count directly and neither is visible in IDSW or Frag.
        counts = sorted(len(v) for v in gt_to_pred.values())
        never_ids = [t for t in predicted_lengths if t not in matched_pred_ids]
        multiplicity = {
            "gt_tracks_matched": len(counts),
            "mean": sum(counts) / len(counts) if counts else None,
            "median": median(counts) if counts else None,
            "max": max(counts) if counts else None,
            "gt_with_multiple_ids": sum(1 for c in counts if c > 1),
        }
        never_matched = {
            "count": len(never_ids),
            "share_of_predicted": len(never_ids) / len(predicted_lengths) if predicted_lengths else None,
            "short_ones": sum(1 for t in never_ids if predicted_lengths[t] <= 3),
        }
        sensitivity = {}
        for minimum in (1, 3, 5, 8):
            predicted_count = sum(length >= minimum for length in predicted_lengths.values())
            gt_count = len(gt_tracks_seen)
            sensitivity[str(minimum)] = {
                "predicted_tracks": predicted_count,
                "gt_tracks": gt_count,
                "signed_relative_error": (predicted_count - gt_count) / gt_count if gt_count else None,
            }
        # Two post-processing knobs govern the reported count, and papers state
        # neither: the length of clip that was evaluated, and how short a track
        # may be before it is discarded. Sweep both to get the error surface --
        # a zero-error contour running across it means any system can be tuned
        # to report "no counting error" at some (length, threshold) pair.
        windows = sorted({*args.window_lengths, len(frames)})
        sequence_sweep = []
        error_surface = []
        for minimum in args.min_track_lens:
            for window in windows:
                entry = count_over_windows(
                    frame_pred_ids,
                    frame_gt_ids,
                    window,
                    min_len=minimum,
                    max_windows=args.max_windows,
                )
                if entry is None:
                    continue
                entry = {"min_track_len": minimum, **entry}
                error_surface.append(entry)
                if minimum == 1:
                    sequence_sweep.append(entry)

        row = {
            "video": video,
            "frames": len(frames),
            "effective_fps": frame_rate,
            "metrics": {key: serialise_metric(value) for key, value in metrics.items()},
            "count_sensitivity": sensitivity,
            "count_vs_sequence_length": sequence_sweep,
            "count_error_surface": error_surface,
            "track_multiplicity": multiplicity,
            "never_matched_tracks": never_matched,
            "predicted_track_lengths": sorted(predicted_lengths.values()),
            "mean_ms_per_frame": 1000 * elapsed / max(1, len(frames)),
        }
        if args.save_frame_tracks:
            row["frame_predicted_ids"] = frame_pred_ids
            row["frame_gt_ids"] = frame_gt_ids
        if args.save_frame_boxes:
            row["frame_predicted_boxes"] = frame_pred_boxes
            row["frame_gt_boxes"] = frame_gt_boxes
            row["frame_names"] = frame_names
        rows.append(row)
        accumulators.append(accumulator)
        names.append(video)
        print(f"{video}: frames={len(frames)} IDF1={row['metrics']['idf1']:.4f} "
              f"MOTA={row['metrics']['mota']:.4f} IDSW={row['metrics']['num_switches']} "
              f"frag={row['metrics']['num_fragmentations']}")
        spans = " ".join(
            f"{entry['window_frames']}f:{entry['mean_signed_relative_error']:+.2f}"
            for entry in sequence_sweep
            if entry["mean_signed_relative_error"] is not None
        )
        if spans:
            print(f"  count-vs-span {spans}")

    overall_frame = mm.metrics.create().compute_many(
        accumulators, names=names, metrics=metric_names, generate_overall=True
    ).loc["OVERALL"].to_dict()
    output = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "root": str(args.root.resolve()),
            "split": args.split,
            "tracker": args.tracker,
            "merge": args.merge,
            "merge_threshold": args.merge_threshold,
            "match_iou": args.match_iou,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "tile": args.tile,
            "stride": args.stride,
            "frame_step": args.frame_step,
            "detector_mode": args.detector_mode,
        },
        "overall": {key: serialise_metric(value) for key, value in overall_frame.items()},
        "videos": rows,
    }
    if dump is not None:
        args.dump_detections.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.dump_detections, "wt") as handle:
            json.dump({
                "config": {
                    "weights": str(Path(args.weights).resolve()),
                    "conf": args.conf,
                    "tile": args.tile,
                    "stride": args.stride,
                    "imgsz": args.imgsz,
                    "detector_mode": args.detector_mode,
                },
                "videos": dump,
            }, handle)
        print(f"Wrote detection cache {args.dump_detections}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"OVERALL: {output['overall']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
