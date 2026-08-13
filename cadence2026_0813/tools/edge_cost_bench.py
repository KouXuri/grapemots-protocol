#!/usr/bin/env python3
"""What each place to cut the stream costs on the processor and on the link.

Section IV-B of the manuscript argues from one measured number, 3.3 fps, and
otherwise leaves the deployment reading as an inference. This measures the rest
of it: decode, detection and association timed separately on the same frames, at
two detector sizes and two input representations, with GPU power sampled through
the run so the cost can be quoted per frame rather than per second and carried to
a device with a different budget.

The link side is measured on the same footage rather than assumed. Three
architectures move different things over it: uplink every frame, uplink only the
frames a sparse annotation cadence would keep and associate off-board, or
associate on board and uplink the tracks. The first two are JPEG bytes actually
encoded here; the third is the size of the track records the same run produced.

The device is stated in the output and is a desktop GPU, not an airborne one. The
per-frame energy is what carries across.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402

from track_grapemots_mot import build_tracker, merge_detections, tiled_raw  # noqa: E402


class PowerSampler(threading.Thread):
    """Poll nvidia-smi for board power while the benchmark runs."""

    def __init__(self, index: int, interval: float = 0.2):
        super().__init__(daemon=True)
        self.index = index
        self.interval = interval
        self.samples: list[float] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.index}",
                     "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                value = out.stdout.strip().splitlines()[0]
                self.samples.append(float(value))
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=3)
        if not self.samples:
            return {"samples": 0}
        array = np.asarray(self.samples, dtype=float)
        return {
            "samples": int(array.size),
            "mean_w": float(array.mean()),
            "median_w": float(np.median(array)),
            "max_w": float(array.max()),
        }


def full_frame(model, frame, imgsz, conf):
    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty((0,))
    return boxes.astype(np.float32), scores.astype(np.float32)


def benchmark(args) -> dict:
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(str(args.weights))
    model.to(args.device)
    tracker = build_tracker(args.tracker, frame_rate=max(1, round(fps)))

    decode_ms: list[float] = []
    detect_ms: list[float] = []
    track_ms: list[float] = []
    track_records = 0
    detections = 0

    # warm-up outside the timed loop: the first inference pays for CUDA context,
    # autotuning and weight upload, none of which recurs in flight
    for _ in range(max(0, args.skip)):
        capture.read()
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("empty video")
    for _ in range(args.warmup):
        if args.tiled:
            tiled_raw(model, frame, args.imgsz, args.conf, args.tile, args.stride)
        else:
            full_frame(model, frame, args.imgsz, args.conf)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)

    sampler = None
    if args.device.startswith("cuda"):
        sampler = PowerSampler(int(args.device.split(":")[-1]) if ":" in args.device else 0)
        sampler.start()

    started = time.perf_counter()
    processed = 0
    while processed < args.frames:
        t0 = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        t1 = time.perf_counter()
        if args.tiled:
            boxes, scores = tiled_raw(model, frame, args.imgsz, args.conf, args.tile, args.stride)
            boxes, scores = merge_detections(boxes, scores, 0.5, "iou")
        else:
            boxes, scores = full_frame(model, frame, args.imgsz, args.conf)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        t2 = time.perf_counter()
        data = (
            np.concatenate(
                [boxes, scores[:, None], np.zeros((len(boxes), 1), dtype=np.float32)], axis=1
            )
            if len(boxes)
            else np.empty((0, 6), dtype=np.float32)
        )
        tracks = tracker.update(
            Boxes(torch.as_tensor(data, dtype=torch.float32), (height, width)), frame
        )
        t3 = time.perf_counter()

        decode_ms.append((t1 - t0) * 1e3)
        detect_ms.append((t2 - t1) * 1e3)
        track_ms.append((t3 - t2) * 1e3)
        detections += len(boxes)
        track_records += len(tracks)
        processed += 1
    elapsed = time.perf_counter() - started
    capture.release()

    power = sampler.stop() if sampler else {"samples": 0}
    peak_memory = (
        torch.cuda.max_memory_allocated(args.device) / 2 ** 20
        if args.device.startswith("cuda") else None
    )
    stage = {
        "decode_ms_median": float(np.median(decode_ms)),
        "detect_ms_median": float(np.median(detect_ms)),
        "track_ms_median": float(np.median(track_ms)),
        "total_ms_median": float(np.median(np.array(decode_ms) + np.array(detect_ms)
                                           + np.array(track_ms))),
    }
    result = {
        "video": str(args.video),
        "weights": str(args.weights),
        "device": args.device,
        "tiled": bool(args.tiled),
        "imgsz": args.imgsz,
        "tile": args.tile if args.tiled else None,
        "stride": args.stride if args.tiled else None,
        "conf": args.conf,
        "tracker": args.tracker,
        "resolution": [width, height],
        "source_fps": fps,
        "frames": processed,
        "wall_seconds": elapsed,
        "fps": processed / elapsed if elapsed else None,
        "stage_ms": stage,
        "detections_total": detections,
        "track_records_total": track_records,
        "peak_gpu_mib": peak_memory,
        "power": power,
    }
    if power.get("samples") and result["fps"]:
        result["joules_per_frame"] = power["mean_w"] / result["fps"]
        result["watts_at_source_rate"] = power["mean_w"] * fps / result["fps"]
    return result


def link_cost(args) -> dict:
    """Bytes each architecture puts on the uplink, measured on this footage."""
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sizes: list[int] = []
    index = 0
    while len(sizes) < args.jpeg_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % max(1, args.jpeg_step) == 0:
            ok_enc, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            )
            if ok_enc:
                sizes.append(int(buffer.nbytes))
        index += 1
    capture.release()
    mean_jpeg = float(np.mean(sizes)) if sizes else None
    file_bytes = args.video.stat().st_size
    duration = total / fps if fps else None
    return {
        "video": str(args.video),
        "file_bytes": file_bytes,
        "frames": total,
        "duration_s": duration,
        "source_mbit_s": file_bytes * 8 / duration / 1e6 if duration else None,
        "jpeg_quality": args.jpeg_quality,
        "jpeg_frames_encoded": len(sizes),
        "jpeg_mean_bytes": mean_jpeg,
        "jpeg_mbit_s_at_source_rate": mean_jpeg * 8 * fps / 1e6 if mean_jpeg else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tracker", default="cfg/trackers/botsort_gmc.yaml")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--skip", type=int, default=0,
                        help="frames to drop before timing, past an empty opening")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--label", default=None)
    parser.add_argument("--link-only", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--jpeg-frames", type=int, default=40)
    parser.add_argument("--jpeg-step", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"label": args.label or args.out.stem, "link": link_cost(args)}
    if not args.link_only:
        payload["compute"] = benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
