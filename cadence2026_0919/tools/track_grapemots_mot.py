#!/usr/bin/env python3
"""Evaluate tiled detection, merge and tracking on the frozen GrapeMOTS split."""

from __future__ import annotations

import argparse
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

try:  # package imports used by pytest and library callers
    from .caout_tracker import CAOUTTracker
    from .caout_math import covariance_intersection
except ImportError:  # direct ``python tools/track_grapemots_mot.py`` execution
    from caout_tracker import CAOUTTracker
    from caout_math import covariance_intersection

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


def _all_tile_origins(height: int, width: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Return stable ``(x0, y0)`` origins in the existing row-major order."""
    return [
        (x0, y0)
        for y0 in tile_starts(height, tile, stride)
        for x0 in tile_starts(width, tile, stride)
    ]


def select_track_aware_tiles(
    height: int,
    width: int,
    tile: int,
    stride: int,
    predicted_boxes: np.ndarray | None,
    frame_position: int,
    keyframe_interval: int = 10,
    roi_margin: int = 320,
    max_tiles: int | None = None,
) -> list[tuple[int, int]]:
    """Select tiles around the previous tracks under a deterministic budget.

    A full tiled pass is retained for the first frame and each keyframe. Between
    keyframes, tiles intersecting an expanded previous-track box are selected,
    then a round-robin exploration tile is added when the budget allows. This is
    intentionally a causal baseline: only information available before the
    current detector call is used.
    """
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be positive")
    if frame_position < 0:
        raise ValueError("frame_position must be non-negative")
    if roi_margin < 0:
        raise ValueError("roi_margin must be non-negative")
    if max_tiles is not None and max_tiles <= 0:
        raise ValueError("max_tiles must be positive")
    origins = _all_tile_origins(height, width, tile, stride)
    if not origins:
        return []
    budget = len(origins) if max_tiles is None else max(1, min(int(max_tiles), len(origins)))
    boxes = np.asarray(predicted_boxes if predicted_boxes is not None else [], dtype=np.float32)
    boxes = boxes.reshape(-1, 4) if boxes.size else np.empty((0, 4), dtype=np.float32)
    keyframe = frame_position == 0 or (
        keyframe_interval > 0 and frame_position % keyframe_interval == 0
    )
    if keyframe or not len(boxes):
        # Spread a constrained keyframe budget over the full image instead of
        # repeatedly favouring the top-left corner.
        indices = np.linspace(0, len(origins) - 1, budget, dtype=int)
        selected = [origins[int(index)] for index in np.unique(indices)]
    else:
        selected = []
        for origin in origins:
            x0, y0 = origin
            x1, y1 = x0 + tile, y0 + tile
            intersects = np.any(
                (boxes[:, 2] >= x0 - roi_margin)
                & (boxes[:, 0] <= x1 + roi_margin)
                & (boxes[:, 3] >= y0 - roi_margin)
                & (boxes[:, 1] <= y1 + roi_margin)
            )
            if bool(intersects):
                selected.append(origin)

        # Keep tiles closest to the predicted objects when the ROI exceeds the
        # budget. The tuple tie-break makes runs reproducible.
        if len(selected) > budget:
            def distance(origin: tuple[int, int]) -> tuple[float, int, int]:
                x0, y0 = origin
                centre_x, centre_y = x0 + tile / 2.0, y0 + tile / 2.0
                distances = np.square(boxes[:, 0] + boxes[:, 2] - 2 * centre_x) + np.square(
                    boxes[:, 1] + boxes[:, 3] - 2 * centre_y
                )
                return float(np.min(distances)), y0, x0

            selected = sorted(selected, key=distance)[:budget]

        # A round-robin tile gives the scheduler a chance to discover new
        # bunches that are outside the current track set.
        exploration = origins[frame_position % len(origins)]
        if exploration not in selected and len(selected) < budget:
            selected.append(exploration)
        selected = selected[:budget]
    return selected


def tiled_raw(
    model,
    image,
    imgsz: int,
    conf: float,
    tile: int,
    stride: int,
    tile_origins: list[tuple[int, int]] | list[tuple[int, int, int, int]] | None = None,
    return_metadata: bool = False,
):
    """Run tile inference, optionally retaining per-detection tile provenance.

    The default two-array return is intentionally unchanged for existing
    scripts. With ``return_metadata=True`` a third list is returned, aligned
    with the raw boxes and carrying stable tile origin, boundary distance and
    truncation-risk fields.
    """
    height, width = image.shape[:2]
    boxes, scores = [], []
    provenance = []
    origins = tile_origins if tile_origins is not None else _all_tile_origins(height, width, tile, stride)
    for origin in origins:
        if len(origin) == 2:
            x0, y0 = origin
            requested_x1, requested_y1 = x0 + tile, y0 + tile
        elif len(origin) == 4:
            # CAOUT's value scheduler returns clipped rectangles; accepting
            # them here keeps one detector interface for both schedulers.
            x0, y0, requested_x1, requested_y1 = origin
        else:
            raise ValueError(f"tile origin must have 2 or 4 values, got {origin!r}")
        if x0 < 0 or y0 < 0 or x0 >= width or y0 >= height:
            raise ValueError(f"tile origin {(x0, y0)} lies outside image {(width, height)}")
        crop_x1 = min(int(requested_x1), width)
        crop_y1 = min(int(requested_y1), height)
        if crop_x1 <= x0 or crop_y1 <= y0:
            raise ValueError(f"tile rectangle {origin!r} has no pixels in image {(width, height)}")
        crop = image[y0:crop_y1, x0:crop_x1]
        crop_height, crop_width = crop.shape[:2]
        tile_id = f"x{x0}_y{y0}"
        result = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        local_xyxy = result.boxes.xyxy.cpu().numpy().copy()
        xyxy = local_xyxy.copy()
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0
        boxes.append(xyxy)
        scores.append(result.boxes.conf.cpu().numpy())
        if return_metadata:
            for local_box in local_xyxy:
                boundary_distance = float(
                    min(
                        local_box[0], local_box[1],
                        crop_width - local_box[2], crop_height - local_box[3],
                    )
                )
                margin_touch = 2.0
                touches_inner_tile_edge = (
                    (local_box[0] <= margin_touch and x0 > 0)
                    or (local_box[1] <= margin_touch and y0 > 0)
                    or (local_box[2] >= crop_width - margin_touch and x0 + crop_width < width)
                    or (local_box[3] >= crop_height - margin_touch and y0 + crop_height < height)
                )
                provenance.append({
                    "tile_id": tile_id,
                    "tile_x0": int(x0),
                    "tile_y0": int(y0),
                    "tile_x1": int(crop_x1),
                    "tile_y1": int(crop_y1),
                    "tile_width": int(crop_width),
                    "tile_height": int(crop_height),
                    "boundary_distance": boundary_distance,
                    "truncation_risk": bool(touches_inner_tile_edge),
                })
    if not boxes:
        empty = (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32))
        return (*empty, []) if return_metadata else empty
    arrays = (np.concatenate(boxes).astype(np.float32), np.concatenate(scores).astype(np.float32))
    return (*arrays, provenance) if return_metadata else arrays


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


def merge_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metric: str,
    metadata: list[dict] | None = None,
    return_metadata: bool = False,
    fusion: str = "nms",
):
    """Merge duplicate tile observations, optionally with CI fusion.

    ``nms`` preserves the historical control.  ``ci`` fuses boxes suppressed by
    the same overlap rule with covariance intersection, which is conservative
    when overlapping tiles are correlated observations of one bunch.
    """
    if fusion not in {"nms", "ci"}:
        raise ValueError("fusion must be 'nms' or 'ci'")
    if not len(boxes):
        return (boxes, scores, []) if return_metadata else (boxes, scores)
    if metadata is not None and len(metadata) != len(boxes):
        raise ValueError("metadata must align one-to-one with boxes")
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32)
    order = score_tensor.argsort(descending=True)
    areas = (tensor[:, 2] - tensor[:, 0]).clamp(min=0) * (tensor[:, 3] - tensor[:, 1]).clamp(min=0)
    keep: list[int] = []
    fused_boxes: list[np.ndarray] = []
    fused_scores: list[float] = []
    fused_metadata: list[dict] = []

    def covariance_for(index: int) -> np.ndarray:
        item = metadata[index] if metadata is not None else {}
        boundary = max(0.0, float(item.get("boundary_distance", 32.0)))
        truncation = bool(item.get("truncation_risk", False))
        sigma_position = max(2.0, 0.04 * (32.0 - min(boundary, 32.0)))
        if truncation:
            sigma_position *= 2.0
        sigma_size = max(2.0, 0.5 * sigma_position)
        return np.diag([
            sigma_position * sigma_position,
            sigma_position * sigma_position,
            sigma_size * sigma_size,
            sigma_size * sigma_size,
        ])

    def as_measurement(value: np.ndarray) -> np.ndarray:
        return np.asarray([
            (value[0] + value[2]) / 2.0,
            (value[1] + value[3]) / 2.0,
            max(1.0, value[2] - value[0]),
            max(1.0, value[3] - value[1]),
        ], dtype=np.float64)

    def as_box(value: np.ndarray) -> np.ndarray:
        cx, cy, width, height = value
        return np.asarray([
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ], dtype=np.float32)

    while len(order):
        index = int(order[0])
        keep.append(index)
        if len(order) == 1:
            if fusion == "ci":
                fused_boxes.append(boxes[index])
                fused_scores.append(float(scores[index]))
                if metadata is not None:
                    fused_metadata.append(dict(metadata[index]))
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
        suppressed = rest[overlap > threshold]
        if fusion == "ci" and len(suppressed):
            mean = as_measurement(boxes[index])
            covariance = covariance_for(index)
            for duplicate in suppressed.tolist():
                mean, covariance, _ = covariance_intersection(
                    mean,
                    covariance,
                    as_measurement(boxes[duplicate]),
                    covariance_for(int(duplicate)),
                    weight=0.5,
                )
            fused_boxes.append(as_box(mean))
            fused_scores.append(float(np.max(scores[[index, *suppressed.tolist()]])))
            item = dict(metadata[index]) if metadata is not None else {}
            item["fusion"] = "covariance_intersection"
            item["fused_tile_count"] = int(len(suppressed) + 1)
            fused_metadata.append(item)
        else:
            fused_boxes.append(boxes[index])
            fused_scores.append(float(scores[index]))
            if metadata is not None:
                fused_metadata.append(dict(metadata[index]))
        order = rest[overlap <= threshold]
    if fusion == "ci":
        merged = (np.asarray(fused_boxes, dtype=np.float32), np.asarray(fused_scores, dtype=np.float32))
    else:
        merged = (boxes[keep], scores[keep])
    if return_metadata:
        return (*merged, fused_metadata if fusion == "ci" else [metadata[index] for index in keep] if metadata is not None else [])
    return merged


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


def build_tracker(config: str, frame_rate: int, source_fps: float = 30.0):
    # "caout" keeps the historical uncompensated behaviour so old numbers stay
    # reproducible; "caout+gmc" adds the same global motion compensation
    # BoT-SORT defaults to, which is the only part CAOUT was missing.
    if config.lower() == "caout":
        return CAOUTTracker(source_fps=source_fps)
    if config.lower() == "caout+gmc":
        return CAOUTTracker(source_fps=source_fps, gmc_method="sparseOptFlow")
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
    parser.add_argument("--merge-fusion", choices=["nms", "ci"], default="nms",
                        help="duplicate-tile merge: historical NMS or covariance intersection")
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--detector-mode", choices=["tiled", "resize", "hybrid"], default="tiled",
                        help="resize is a full-frame control; hybrid adds a low-resolution "
                             "global pass to selected high-resolution tiles")
    parser.add_argument("--global-imgsz", type=int,
                        help="image size for hybrid global pass; defaults to --imgsz")
    parser.add_argument("--tile-scheduler", choices=["all", "track_roi", "risk"], default="all",
                        help="tile policy; risk uses CAOUT covariance/value ranking")
    parser.add_argument("--tile-keyframe-interval", type=int, default=10,
                        help="processed frames between full tiled passes for track_roi")
    parser.add_argument("--roi-margin", type=int, default=320,
                        help="pixel margin around previous track boxes for track_roi")
    parser.add_argument("--max-tiles", type=int,
                        help="per-frame tile budget for track_roi; default uses all tiles")
    parser.add_argument("--tile-budget-fraction", type=float, default=0.5,
                        help="fraction of complete-cover pixels for the risk scheduler")
    parser.add_argument("--tile-exploration-fraction", type=float, default=0.0,
                        help="fraction of risk tile capacity reserved for low-coverage discovery")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame-offset", type=int, default=0,
                        help="sampling phase: process all_frames[offset::frame_step]; "
                             "0 reproduces every earlier run")
    parser.add_argument("--catchup", action="store_true",
                        help="advance every live and lost track's motion model by "
                             "frame_step - 1 extra predictions before each update, so "
                             "the filter moves on elapsed time rather than one step "
                             "per processed frame (ported from cadence_timescale_0815)")
    parser.add_argument("--source-fps", type=float, default=30.0,
                        help="source video FPS used by cadence-aware tracker")
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
    parser.add_argument("--save-tile-provenance", action="store_true",
                        help="store merged detection-to-tile provenance per frame")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1 or not 0 <= args.conf <= 1:
        raise SystemExit("Require --frame-step >= 1 and 0 <= --conf <= 1")
    if not 0 <= args.frame_offset < args.frame_step:
        raise SystemExit("Require 0 <= --frame-offset < --frame-step")
    if args.catchup and args.tracker.lower().startswith("caout"):
        raise SystemExit("--catchup applies to BoT-SORT/ByteTrack only")
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

    model = YOLO(args.weights)
    metric_names = [
        "idf1", "mota", "num_switches", "num_fragmentations", "precision", "recall",
        "num_false_positives", "num_misses", "num_objects", "num_predictions",
    ]
    accumulators, names, rows = [], [], []
    for video, all_frames in sorted(grouped.items()):
        frames = all_frames[args.frame_offset::args.frame_step]
        if args.limit:
            frames = frames[:args.limit]
        frame_rate = max(1, round(effective_fps(all_frames) / args.frame_step))
        tracker = build_tracker(args.tracker, frame_rate, args.source_fps)
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
        previous_track_boxes = np.empty((0, 4), dtype=np.float32)
        tile_counts: list[int] = []
        tile_detections = 0
        tile_boundary_risk = 0
        frame_tile_provenance: list[list[dict]] = []
        frame_tile_audits: list[list[dict]] = []
        elapsed = 0.0
        frame_ms: list[float] = []

        for position, image_path in enumerate(frames):
            image = cv2.imread(str(image_path))
            if image is None:
                raise SystemExit(f"Could not read {image_path}")
            height, width = image.shape[:2]
            gt_ids, gt_boxes = load_gt_tracks(track_dir / f"{image_path.stem}.txt", width, height)
            gt_tracks_seen.update(gt_ids)

            start = time.perf_counter()
            if args.detector_mode == "resize":
                raw_boxes, raw_scores = resize_raw(model, image, args.imgsz, args.conf)
                raw_metadata = None
            else:
                tile_origins = None
                if args.tile_scheduler == "track_roi":
                    tile_origins = select_track_aware_tiles(
                        height,
                        width,
                        args.tile,
                        args.stride,
                        previous_track_boxes,
                        frame_position=position,
                        keyframe_interval=args.tile_keyframe_interval,
                        roi_margin=args.roi_margin,
                        max_tiles=args.max_tiles,
                    )
                elif args.tile_scheduler == "risk":
                    if not isinstance(tracker, CAOUTTracker):
                        raise SystemExit("--tile-scheduler risk requires --tracker caout")
                    if not 0 < args.tile_budget_fraction <= 1:
                        raise SystemExit("--tile-budget-fraction must lie in (0, 1]")
                    tile_origins = tracker.propose_tiles(
                        width,
                        height,
                        args.tile,
                        args.stride,
                        args.tile_budget_fraction,
                        args.tile_exploration_fraction,
                    )
                    frame_tile_audits.append(list(tracker.last_tile_audit))
                tile_boxes, tile_scores, tile_metadata = tiled_raw(
                    model,
                    image,
                    args.imgsz,
                    args.conf,
                    args.tile,
                    args.stride,
                    tile_origins=tile_origins,
                    return_metadata=True,
                )
                tile_counts.append(len(tile_origins) if tile_origins is not None else len(
                    _all_tile_origins(height, width, args.tile, args.stride)
                ))
                if args.detector_mode == "hybrid":
                    global_boxes, global_scores = resize_raw(
                        model,
                        image,
                        args.global_imgsz or args.imgsz,
                        args.conf,
                    )
                    global_metadata = [
                        {
                            "tile_id": "global_resize",
                            "tile_x0": 0,
                            "tile_y0": 0,
                            "tile_x1": int(width),
                            "tile_y1": int(height),
                            "tile_width": int(width),
                            "tile_height": int(height),
                            "boundary_distance": float(min(width, height)),
                            "truncation_risk": False,
                            "source": "global_resize",
                        }
                        for _ in range(len(global_boxes))
                    ]
                    raw_boxes = np.concatenate([global_boxes, tile_boxes], axis=0)
                    raw_scores = np.concatenate([global_scores, tile_scores], axis=0)
                    raw_metadata = global_metadata + tile_metadata
                else:
                    raw_boxes, raw_scores, raw_metadata = tile_boxes, tile_scores, tile_metadata
            merged_boxes, merged_scores, merged_metadata = merge_detections(
                raw_boxes,
                raw_scores,
                args.merge_threshold,
                args.merge,
                metadata=raw_metadata,
                return_metadata=True,
                fusion=args.merge_fusion,
            )
            tile_detections += len(merged_metadata)
            tile_boundary_risk += sum(bool(item["truncation_risk"]) for item in merged_metadata)
            if args.save_tile_provenance:
                frame_tile_provenance.append(merged_metadata)
            if isinstance(tracker, CAOUTTracker):
                frame_match = re.search(r"(\d+)$", image_path.stem)
                frame_index = int(frame_match.group(1)) if frame_match else position
                tracks = tracker.update(merged_boxes, merged_scores, frame_index, image)
            else:
                if args.catchup and position > 0 and args.frame_step > 1:
                    # update() predicts once; frame_step - 1 more puts the filter
                    # at the elapsed time. Lost tracks are what a re-acquisition
                    # matches against, so they advance too.
                    pool = list(tracker.tracked_stracks) + list(tracker.lost_stracks)
                    if pool:
                        for _ in range(args.frame_step - 1):
                            tracker.multi_predict(pool)
                data = np.concatenate(
                    [merged_boxes, merged_scores[:, None], np.zeros((len(merged_boxes), 1), dtype=np.float32)], axis=1
                ) if len(merged_boxes) else np.empty((0, 6), dtype=np.float32)
                tracks = tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32), image.shape[:2]), image)
            frame_seconds = time.perf_counter() - start
            elapsed += frame_seconds
            frame_ms.append(1000 * frame_seconds)

            if len(tracks):
                predicted_boxes = np.asarray(tracks[:, :4], dtype=np.float32)
                predicted_ids = [int(track_id) for track_id in tracks[:, 4]]
            else:
                predicted_boxes = np.empty((0, 4), dtype=np.float32)
                predicted_ids = []
            previous_track_boxes = predicted_boxes.copy()
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
            # Per processed frame, same timer span as the mean (detection, merge,
            # tracker incl. GMC; excludes image decode and scoring). The first
            # entry carries the model's lazy initialisation.
            "frame_ms": [round(value, 3) for value in frame_ms],
            "annotated_frames": len(all_frames),
            "tile_schedule": {
                "scheduler": args.tile_scheduler if args.detector_mode in {"tiled", "hybrid"} else "resize",
                "global_pass": args.detector_mode == "hybrid",
                "mean_tiles_per_frame": sum(tile_counts) / len(tile_counts) if tile_counts else 1.0,
                "min_tiles_per_frame": min(tile_counts) if tile_counts else 1,
                "max_tiles_per_frame": max(tile_counts) if tile_counts else 1,
                "merged_detections": tile_detections,
                "merged_truncation_risk": tile_boundary_risk,
                "truncation_risk_rate": (
                    tile_boundary_risk / tile_detections if tile_detections else None
                ),
            },
        }
        if args.save_frame_tracks:
            row["frame_predicted_ids"] = frame_pred_ids
            row["frame_gt_ids"] = frame_gt_ids
        if args.save_frame_boxes:
            row["frame_predicted_boxes"] = frame_pred_boxes
            row["frame_gt_boxes"] = frame_gt_boxes
            row["frame_names"] = frame_names
        if args.save_tile_provenance:
            row["frame_tile_provenance"] = frame_tile_provenance
            if args.tile_scheduler == "risk":
                row["frame_tile_value_audit"] = frame_tile_audits
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
            "merge_fusion": args.merge_fusion,
            "merge_threshold": args.merge_threshold,
            "match_iou": args.match_iou,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "tile": args.tile,
            "stride": args.stride,
            "frame_step": args.frame_step,
            "frame_offset": args.frame_offset,
            "catchup": args.catchup,
            "detector_mode": args.detector_mode,
            "global_imgsz": args.global_imgsz or args.imgsz,
            "tile_scheduler": args.tile_scheduler,
            "tile_keyframe_interval": args.tile_keyframe_interval,
            "roi_margin": args.roi_margin,
            "max_tiles": args.max_tiles,
            "tile_budget_fraction": args.tile_budget_fraction,
            "tile_exploration_fraction": args.tile_exploration_fraction,
        },
        "overall": {key: serialise_metric(value) for key, value in overall_frame.items()},
        "videos": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"OVERALL: {output['overall']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
