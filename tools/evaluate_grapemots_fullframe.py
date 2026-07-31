#!/usr/bin/env python3
"""Fair full-frame detection evaluation: resize vs tiled, on the SAME full frames.

The earlier comparison was unfair: the resize model was scored on full frames while
the tiled model was scored on 1280 tiles (where bunches are large and easy). Here
both are scored on the SAME full-frame test set with COCO box AP:
  --mode resize : whole frame resized to imgsz (one forward pass)
  --mode tiled  : SAHI-style overlapping tiles, boxes merged back with NMS

Ground truth comes from the full-frame YOLO labels already produced for the test
split, so GT is identical across modes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def resize_pred(model, img, imgsz, conf):
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def overlap_nms(boxes, scores, threshold, metric):
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32)
    if metric == "iou":
        return nms(tensor, score_tensor, threshold).numpy()
    order = score_tensor.argsort(descending=True)
    areas = (tensor[:, 2] - tensor[:, 0]).clamp(min=0) * (tensor[:, 3] - tensor[:, 1]).clamp(min=0)
    keep = []
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
        overlap = intersection / torch.minimum(areas[index].expand_as(areas[rest]), areas[rest]).clamp(min=1e-6)
        order = rest[overlap <= threshold]
    return np.asarray(keep, dtype=int)


def tiled_pred(model, img, imgsz, conf, tile=1280, stride=960, iou=0.6, merge_metric="iou"):
    H, W = img.shape[:2]
    boxes, scores = [], []
    for y0 in tile_starts(H, tile, stride):
        for x0 in tile_starts(W, tile, stride):
            crop = img[y0:y0 + tile, x0:x0 + tile]
            r = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xy = r.boxes.xyxy.cpu().numpy().copy()
            sc = r.boxes.conf.cpu().numpy()
            xy[:, [0, 2]] += x0
            xy[:, [1, 3]] += y0
            boxes.append(xy); scores.append(sc)
    if not boxes:
        return np.empty((0, 4)), np.empty((0,))
    boxes = np.concatenate(boxes); scores = np.concatenate(scores)
    keep = overlap_nms(boxes, scores, iou, merge_metric)
    return boxes[keep], scores[keep]


def coco_metrics(dataset, detections, image_ids):
    selected = set(image_ids)
    subset_detections = [row for row in detections if row["image_id"] in selected]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        if subset_detections:
            coco_dt = coco_gt.loadRes(subset_detections)
        else:
            # pycocotools indexes anns[0] in loadRes(), so an empty list raises
            # before COCOeval can report the valid zero-detection case.
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": dataset["images"],
                "annotations": [],
                "categories": dataset["categories"],
            }
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(selected)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "ap50_95": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
        "ar100": float(stats[8]),
        "frames": len(selected),
        "gt": sum(1 for row in dataset["annotations"] if row["image_id"] in selected),
        "detections": len(subset_detections),
    }


def print_metrics(label, metrics):
    print(f"{label} frames={metrics['frames']} GT={metrics['gt']} det={metrics['detections']}")
    print(f"  AP50={metrics['ap50']:.4f}  AP75={metrics['ap75']:.4f}  "
          f"AP50-95={metrics['ap50_95']:.4f}  AR100={metrics['ar100']:.4f}")
    print(f"  AP_small={metrics['ap_small']:.4f}  AP_medium={metrics['ap_medium']:.4f}  "
          f"AP_large={metrics['ap_large']:.4f}")


def load_gt_boxes(label_path: Path, W: int, H: int):
    """Accept both YOLO detection labels (cls xc yc w h) and segmentation
    polygons (cls x1 y1 x2 y2 ...); polygons are reduced to their bounding box."""
    boxes = []
    if not label_path.exists():
        return boxes
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        coords = [float(v) for v in p[1:]]
        if len(coords) == 4:
            xc, yc, w, h = coords
            x1 = (xc - w / 2) * W; y1 = (yc - h / 2) * H; bw = w * W; bh = h * H
        elif len(coords) >= 6:
            xs = coords[0::2]; ys = coords[1::2]
            x1 = min(xs) * W; y1 = min(ys) * H
            bw = (max(xs) - min(xs)) * W; bh = (max(ys) - min(ys)) * H
        else:
            continue
        boxes.append([x1, y1, bw, bh])  # COCO xywh
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", choices=["resize", "tiled"], required=True)
    ap.add_argument("--root", default="datasets/grapemots_det_721")
    ap.add_argument("--split", default="test",
                    help="a split directory, or 'all' to scan train+val+test")
    ap.add_argument("--videos", nargs="+",
                    help="restrict to these videos; use with --split all to score "
                         "a split-sensitivity model on its own held-out videos")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=960)
    ap.add_argument("--merge-iou", type=float, default=0.6)
    ap.add_argument("--merge-metric", choices=["iou", "ios"], default="iou")
    ap.add_argument("--per-video", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--coco-dir", type=Path)
    args = ap.parse_args()

    root = Path(args.root)
    # `--split all` scans every split directory.  The split-sensitivity study
    # rotates which videos are held out, so a video that is validation for one
    # split sits in images/train of the frozen 721 layout; selecting by video
    # name is the only way to score each model on its own held-out videos
    # without duplicating 5,755 frames per split.
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    imgs, lbl_dirs = [], {}
    for split in splits:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            continue
        for path in img_dir.iterdir():
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                imgs.append(path)
                lbl_dirs[path.name] = root / "labels" / split
    if args.videos:
        wanted = set(args.videos)
        imgs = [p for p in imgs if p.name.split("__")[0] in wanted]
        missing = wanted - {p.name.split("__")[0] for p in imgs}
        if missing:
            raise SystemExit(f"no frames found for {sorted(missing)}")
    # Evaluate on ALL annotated frames (not the sub-sampled training list),
    # so the reported AP is not biased by frame subsampling.
    imgs = sorted(imgs, key=lambda p: p.name)
    if not imgs:
        raise SystemExit(f"no images selected under {root}")

    model = YOLO(args.weights)
    coco_images, coco_anns, coco_dets = [], [], []
    video_image_ids = {}
    ann_id = 1
    for img_id, ip in enumerate(imgs, 1):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        H, W = img.shape[:2]
        video = ip.name.split("__")[0]
        coco_images.append({"id": img_id, "width": W, "height": H, "file_name": ip.name, "video": video})
        video_image_ids.setdefault(video, []).append(img_id)
        for b in load_gt_boxes(lbl_dirs[ip.name] / (ip.stem + ".txt"), W, H):
            coco_anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                              "bbox": b, "area": b[2] * b[3], "iscrowd": 0})
            ann_id += 1
        if args.mode == "resize":
            xyxy, scores = resize_pred(model, img, args.imgsz, args.conf)
        else:
            xyxy, scores = tiled_pred(
                model, img, args.imgsz, args.conf, args.tile, args.stride, args.merge_iou, args.merge_metric
            )
        for (x1, y1, x2, y2), s in zip(xyxy, scores):
            coco_dets.append({"image_id": img_id, "category_id": 1,
                              "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                              "score": float(s)})

    gt = {"images": coco_images, "annotations": coco_anns,
          "categories": [{"id": 1, "name": "grape"}]}
    overall = coco_metrics(gt, coco_dets, [row["id"] for row in coco_images])
    per_video = {}
    print_metrics(f"MODE={args.mode} SPLIT={args.split}", overall)
    if args.per_video:
        for video, image_ids in sorted(video_image_ids.items()):
            per_video[video] = coco_metrics(gt, coco_dets, image_ids)
            print_metrics(f"VIDEO={video}", per_video[video])

    summary = {
        "weights": str(Path(args.weights).resolve()),
        "root": str(root.resolve()),
        "split": args.split,
        "mode": args.mode,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "tile": args.tile if args.mode == "tiled" else None,
        "stride": args.stride if args.mode == "tiled" else None,
        "merge_iou": args.merge_iou if args.mode == "tiled" else None,
        "merge_metric": args.merge_metric if args.mode == "tiled" else None,
        "overall": overall,
        "per_video": per_video,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out}")
    if args.coco_dir:
        args.coco_dir.mkdir(parents=True, exist_ok=True)
        (args.coco_dir / "ground_truth.json").write_text(json.dumps(gt))
        (args.coco_dir / "predictions.json").write_text(json.dumps(coco_dets))
        print(f"Wrote COCO payloads under {args.coco_dir}")


if __name__ == "__main__":
    main()
