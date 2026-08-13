#!/usr/bin/env python3
"""How far a target moves between retained frames, and how much overlap survives.

Three things the manuscript asserts about the geometry are checked here rather
than argued, and all three come off the reference annotations alone.

WHAT r IS AT EACH THINNING STEP. The criterion is written in r, the centre
displacement in units of target size, but the tables report it only at the step a
release happens to use. Thinning a densely annotated corpus by k walks r across
its whole range on one corpus, which is what the cadence result needs beside it:
the sign of the count error can then be read against the r that produced it
rather than against k, and r is the quantity the criterion is stated in.

WHICH SCALE. The implementation divides by the square root of the earlier box's
area, so swapping the order of a pair changes r. Both that convention and the
symmetric one, the geometric mean of the two areas, are reported here so the
asymmetry is a measured quantity rather than an unstated choice.

WHETHER THE SQUARE CURVE BOUNDS REAL BOXES. Equation (2) is derived for equal
squares. For equal rectangles of aspect ratio a displaced along the long axis it
becomes (sqrt(a) - r) / (sqrt(a) + r), which is above the square curve, so the
square curve is not a bound on rectangles. The aspect-ratio distribution of the
real boxes is what decides how far above, and it is reported per corpus.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FRAME_RE = re.compile(r"__frame_(\d+)\.txt$")


def video_frames(root: Path, video: str) -> list[Path]:
    found: list[Path] = []
    for split in ("all", "train", "val", "test"):
        directory = root / "tracks" / split
        if directory.is_dir():
            found.extend(directory.glob(f"{video}__frame_*.txt"))
    if not found:
        raise SystemExit(f"no track sidecars for {video} under {root}")
    return sorted(found, key=lambda p: int(FRAME_RE.search(p.name).group(1)))


def read_manifest(root: Path) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    with (root / "manifest.csv").open() as handle:
        for row in csv.DictReader(handle):
            first = row["resolution"].split(";")[0].split(":")[0]
            width, height = (int(value) for value in first.split("x"))
            sizes[row["video"]] = (width, height)
    return sizes


def read_boxes(path: Path, width: int, height: int) -> dict[int, np.ndarray]:
    boxes: dict[int, np.ndarray] = {}
    if not path.is_file():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        _cls, track, xc, yc, w, h = map(float, parts)
        boxes[int(track)] = np.array(
            [xc * width, yc * height, w * width, h * height], dtype=float
        )
    return boxes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"pairs": 0}
    array = np.asarray(values, dtype=float)
    return {
        "pairs": int(array.size),
        "p05": float(np.percentile(array, 5)),
        "q1": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "q3": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 15, 30, 60])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sizes = read_manifest(args.root)
    videos = args.videos or sorted(sizes)
    per_step: dict[int, dict] = {}
    aspect_all: list[float] = []
    aspect_ratio_of_pair: list[float] = []

    cache: dict[str, list[dict[int, np.ndarray]]] = {}
    for video in videos:
        width, height = sizes[video]
        frames = video_frames(args.root, video)
        cache[video] = [read_boxes(path, width, height) for path in frames]
        for boxes in cache[video]:
            for box in boxes.values():
                if box[3] > 0:
                    aspect_all.append(float(box[2] / box[3]))

    for step in args.steps:
        r_earlier: list[float] = []
        r_symmetric: list[float] = []
        overlaps: list[float] = []
        medians_r: list[float] = []
        medians_iou: list[float] = []
        for video in videos:
            frames = cache[video]
            kept = list(range(0, len(frames), step))
            seq_r: list[float] = []
            seq_iou: list[float] = []
            for left, right in zip(kept, kept[1:]):
                a_boxes, b_boxes = frames[left], frames[right]
                for track, a in a_boxes.items():
                    b = b_boxes.get(track)
                    if b is None:
                        continue
                    displacement = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                    area_a = float(a[2] * a[3])
                    area_b = float(b[2] * b[3])
                    if area_a <= 0 or area_b <= 0:
                        continue
                    r_a = displacement / np.sqrt(area_a)
                    # symmetric scale: sqrt of the geometric mean of the two areas
                    r_g = displacement / (area_a * area_b) ** 0.25
                    overlap = iou(a, b)
                    r_earlier.append(r_a)
                    r_symmetric.append(r_g)
                    overlaps.append(overlap)
                    seq_r.append(r_a)
                    seq_iou.append(overlap)
                    if step == 1 and a[3] > 0 and b[3] > 0:
                        aspect_ratio_of_pair.append(
                            float((a[2] / a[3]) / (b[2] / b[3]))
                        )
            if seq_r:
                medians_r.append(float(np.median(seq_r)))
                medians_iou.append(float(np.median(seq_iou)))
        ratio = None
        if r_earlier:
            a_array = np.asarray(r_earlier)
            g_array = np.asarray(r_symmetric)
            live = g_array > 0
            if live.any():
                ratio = float(np.median(a_array[live] / g_array[live]))
        # the square curve as a claim about real boxes: how often is it exceeded
        if r_earlier:
            r_array = np.asarray(r_earlier)
            curve = np.where(r_array < 1.0, (1 - r_array) / (1 + r_array), 0.0)
            above = float(np.mean(np.asarray(overlaps) > curve + 1e-9))
        else:
            above = None
        per_step[step] = {
            "r_earlier_box": quantiles(r_earlier),
            "r_geometric_mean": quantiles(r_symmetric),
            "iou": quantiles(overlaps),
            "sequence_median_r": float(np.median(medians_r)) if medians_r else None,
            "sequence_median_iou": float(np.median(medians_iou)) if medians_iou else None,
            "sequences": len(medians_r),
            "median_ratio_earlier_over_symmetric": ratio,
            "share_iou_below_0.3": (
                float(np.mean(np.asarray(overlaps) < 0.3)) if overlaps else None
            ),
            "share_iou_zero": (
                float(np.mean(np.asarray(overlaps) <= 0.0)) if overlaps else None
            ),
            "share_above_square_curve": above,
        }
        block = per_step[step]
        if not block["r_earlier_box"]["pairs"]:
            print(f"{args.corpus} k={step:<3} no same-identity pairs survive", flush=True)
            continue
        print(
            f"{args.corpus} k={step:<3} pairs={block['r_earlier_box']['pairs']:<7} "
            f"seq-median r={block['sequence_median_r']:.4f} "
            f"pair-median r={block['r_earlier_box']['median']:.4f} "
            f"(sym {block['r_geometric_mean']['median']:.4f}) "
            f"IoU median={block['iou']['median']:.4f} "
            f"share<0.3={block['share_iou_below_0.3']:.3f}",
            flush=True,
        )

    aspect = np.asarray(aspect_all, dtype=float)
    payload = {
        "corpus": args.corpus,
        "root": str(args.root),
        "videos": videos,
        "steps": args.steps,
        "aspect_ratio": {
            "boxes": int(aspect.size),
            "p05": float(np.percentile(aspect, 5)),
            "q1": float(np.percentile(aspect, 25)),
            "median": float(np.median(aspect)),
            "q3": float(np.percentile(aspect, 75)),
            "p95": float(np.percentile(aspect, 95)),
            "share_within_1.25_of_square": float(
                np.mean((aspect >= 0.8) & (aspect <= 1.25))
            ),
        },
        "pair_aspect_ratio_change": quantiles(aspect_ratio_of_pair),
        "by_step": {str(k): v for k, v in per_step.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    a = payload["aspect_ratio"]
    print(
        f"{args.corpus} aspect ratio w/h: median={a['median']:.3f} "
        f"IQR=[{a['q1']:.3f},{a['q3']:.3f}] "
        f"90% range=[{a['p05']:.3f},{a['p95']:.3f}]",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
