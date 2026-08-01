#!/usr/bin/env python3
"""Convert the GrapeMOTS (2024) MOTS masks into a YOLO detection dataset.

GrapeMOTS layout (per video):
    <root>/<Video>/<default|default-2>/images/frame_XXXXXX.PNG
    <root>/<Video>/<default|default-2>/instances/frame_XXXXXX.png   (uint16 MOTS mask)
    <root>/<Video>/<default|default-2>/instances/labels.txt         (class names, 1-based)

MOTS encoding: pixel value 0 = background; non-zero value = class_id * 1000 + track_id,
where class_id is the 1-based index into labels.txt (grape=1, trunk=2, pole=3).

This builder writes an Ultralytics-compatible detection dataset using symlinked images
(no 4K frame copies) plus, for each frame, a track sidecar carrying the MOTS track id so
downstream tracking/counting evaluation can match predicted tracks to ground-truth bunches.

Splits are made at the VIDEO level (never per frame) to avoid adjacent-frame leakage. The
default split is the closest 7:2:1 allocation under this constraint while retaining both
acquisition modes in the training and validation partitions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Videos whose CVAT export lives under "default-2" instead of "default".
BASE_OVERRIDE = {"NoPathPlanning_3": "default-2"}

VIDEOS = [
    "NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3",
    "PathPlanning_1", "PathPlanning_2", "PathPlanning_3", "PathPlanning_4",
    "PathPlanning_5", "PathPlanning_6", "PathPlanning_7", "PathPlanning_8",
]

# Closest video-level 7:2:1 split over the 5,755 valid image-mask pairs.
# Frame totals: train=4,012 (69.71%), val=1,160 (20.16%), test=583 (10.13%).
DEFAULT_SPLIT = {
    "train": ["NoPathPlanning_2", "NoPathPlanning_3", "PathPlanning_1", "PathPlanning_3",
              "PathPlanning_6", "PathPlanning_7", "PathPlanning_8"],
    "val":   ["NoPathPlanning_1", "PathPlanning_5"],
    "test":  ["PathPlanning_2", "PathPlanning_4"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="GrapeMOTS root containing the per-video folders.")
    p.add_argument("--dst", type=Path, default=Path("datasets/grapemots_det"),
                   help="Output dataset root.")
    p.add_argument("--classes", nargs="+", default=["grape"],
                   help="Class names (1-based, MOTS order) to KEEP. Default: grape only.")
    p.add_argument("--min-area", type=int, default=20, help="Drop instances smaller than this (px).")
    p.add_argument("--split-json", type=Path,
                   help="Optional JSON overriding the default {split: [videos]} assignment.")
    p.add_argument("--limit", type=int, help="Process only the first N frames per video (smoke test).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def base_dir(src: Path, video: str) -> Path:
    return src / video / BASE_OVERRIDE.get(video, "default")


def read_labels(inst_dir: Path) -> list[str]:
    """Read labels.txt (1-based class names) if present; fall back to ['grape']."""
    path = inst_dir / "labels.txt"
    if not path.exists():
        return ["grape"]
    names = [ln.strip().lower() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names or ["grape"]


def mask_to_boxes(mask: np.ndarray, keep_class_ids: set[int], min_area: int):
    """Yield (class_id, track_id, x0, y0, x1, y1) for each kept instance in a MOTS mask."""
    h, w = mask.shape[:2]
    for value in np.unique(mask):
        if value == 0:
            continue
        value = int(value)
        class_id = value // 1000
        track_id = value % 1000
        if class_id not in keep_class_ids:
            continue
        ys, xs = np.where(mask == value)
        if xs.size == 0:
            continue
        area = int(xs.size)
        if area < min_area:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if x1 <= x0 or y1 <= y0:
            continue
        yield class_id, track_id, x0, y0, x1, y1, area


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".PNG", ".png", ".jpg", ".JPG", ".jpeg"):
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main() -> None:
    args = parse_args()
    split_map = json.loads(args.split_json.read_text()) if args.split_json else DEFAULT_SPLIT
    video_to_split = {v: s for s, vids in split_map.items() for v in vids}

    dst = args.dst
    for split in split_map:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        (dst / "tracks" / split).mkdir(parents=True, exist_ok=True)

    # Class-id remap: kept MOTS class (1-based) -> contiguous YOLO class (0-based).
    manifest_rows: list[dict] = []
    per_video_stats: dict[str, dict] = {}
    global_keep_names = [c.lower() for c in args.classes]

    for video in VIDEOS:
        if video not in video_to_split:
            print(f"[skip] {video}: not in split map")
            continue
        split = video_to_split[video]
        bdir = base_dir(args.src, video)
        images_dir, inst_dir = bdir / "images", bdir / "instances"
        if not inst_dir.exists():
            print(f"[warn] {video}: missing {inst_dir}")
            continue

        names = read_labels(inst_dir)  # 1-based class names for THIS video
        keep_class_ids = {i + 1 for i, n in enumerate(names) if n in global_keep_names}
        if not keep_class_ids:
            print(f"[warn] {video}: none of {global_keep_names} in {names}; skipping")
            continue
        # Remap kept video class-id -> global class index (by name position in --classes).
        remap = {i + 1: global_keep_names.index(names[i]) for i in range(len(names))
                 if (i + 1) in keep_class_ids}

        inst_files = sorted(f for f in inst_dir.glob("*.png"))
        if args.limit:
            inst_files = inst_files[: args.limit]

        n_frames = n_inst = 0
        track_ids: set[int] = set()
        res_counter: Counter = Counter()
        for inst_path in inst_files:
            stem = inst_path.stem
            img_path = find_image(images_dir, stem)
            if img_path is None:
                continue
            mask = np.array(Image.open(inst_path))
            h, w = mask.shape[:2]
            label_lines, track_lines = [], []
            for class_id, track_id, x0, y0, x1, y1, _area in mask_to_boxes(mask, keep_class_ids, args.min_area):
                cls = remap[class_id]
                xc = ((x0 + x1) / 2) / w
                yc = ((y0 + y1) / 2) / h
                bw = (x1 - x0) / w
                bh = (y1 - y0) / h
                label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                # Track sidecar keeps the global MOTS track key (class*1000+track_id).
                track_lines.append(f"{cls} {class_id * 1000 + track_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                track_ids.add(class_id * 1000 + track_id)

            key = f"{video}__{stem}"
            # Symlink image (no 4K copy); labels + track sidecar written as text.
            link = dst / "images" / split / f"{key}{img_path.suffix}"
            if link.exists() or link.is_symlink():
                if args.overwrite:
                    link.unlink()
                    link.symlink_to(img_path.resolve())
            else:
                link.symlink_to(img_path.resolve())
            (dst / "labels" / split / f"{key}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
            (dst / "tracks" / split / f"{key}.txt").write_text("\n".join(track_lines) + ("\n" if track_lines else ""))

            n_frames += 1
            n_inst += len(label_lines)
            res_counter[f"{w}x{h}"] += 1

        mode = "multi-view" if video.startswith("PathPlanning") else "frontal"
        per_video_stats[video] = {
            "split": split, "mode": mode, "frames": n_frames, "instances": n_inst,
            "gt_tracks": len(track_ids), "resolution": dict(res_counter),
        }
        manifest_rows.append({
            "video": video, "split": split, "mode": mode,
            "frames": n_frames, "instances": n_inst, "gt_tracks": len(track_ids),
            "resolution": ";".join(f"{k}:{v}" for k, v in res_counter.items()),
        })
        print(f"[ok] {video:18s} split={split:5s} mode={mode:10s} "
              f"frames={n_frames:4d} inst={n_inst:6d} tracks={len(track_ids):3d} res={dict(res_counter)}")

    # Dataset YAML for Ultralytics.
    yaml_path = dst / "grapemots_det.yaml"
    yaml_path.write_text(
        f"path: {dst.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(global_keep_names)}\n"
        f"names: {global_keep_names}\n"
    )
    # Manifest + stats.
    with (dst / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "split", "mode", "frames", "instances", "gt_tracks", "resolution"])
        w.writeheader(); w.writerows(manifest_rows)
    (dst / "dataset_stats.json").write_text(json.dumps(per_video_stats, indent=2))

    totals = defaultdict(int)
    for r in manifest_rows:
        totals[r["split"]] += r["frames"]
    print("\n=== split frame totals ===")
    for s, n in totals.items():
        print(f"  {s}: {n} frames")
    print(f"\nWrote dataset YAML: {yaml_path}")
    print(f"Wrote manifest:     {dst / 'manifest.csv'}")


if __name__ == "__main__":
    main()
