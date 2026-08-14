#!/usr/bin/env python3
"""Price a sparse uplink with the same codec as the full-rate one.

The link row of the cost table compares the released MP4's own bitrate against
JPEG q90 frames. Two reviewers noted that this is a comparison across codecs, so
part of the saving could be the encoder rather than the sampling. Here the sparse
frame set is re-encoded with the same H.264 encoder as the source, all-intra
(keyint 1), because frames 0.6 s apart over a moving canopy have nothing to
predict from one another. Reported alongside is the same set at the source
cadence, so the two rows differ only in which frames are sent.

    python3 tools/link_allintra.py --video "<path>.mp4"

Writes runs/link_allintra_0814/results/link_allintra.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--step", type=int, default=36,
                   help="source frames between kept frames (the released median)")
    p.add_argument("--crf", type=int, default=23, help="x264 quality, lower is better")
    p.add_argument("--frames", type=int, default=40, help="kept frames to encode")
    p.add_argument("--full-frames", type=int, default=120,
                   help="consecutive frames to encode as the full-rate control")
    p.add_argument("--out", type=Path,
                   default=ROOT / "runs/link_allintra_0814/results/link_allintra.json")
    return p.parse_args()


def encode(paths: list[Path], fps: float, crf: int, all_intra: bool) -> int:
    """Encode a frame sequence with x264 and return the byte size."""
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in paths))
        out = Path(tmp) / "out.mp4"
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-r", str(fps),
               "-f", "concat", "-safe", "0", "-i", str(listing),
               "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if all_intra:
            cmd += ["-g", "1", "-keyint_min", "1", "-x264-params", "scenecut=0"]
        cmd.append(str(out))
        subprocess.run(cmd, check=True)
        return out.stat().st_size


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_bytes = args.video.stat().st_size
    duration = total / fps

    with tempfile.TemporaryDirectory() as tmp:
        kept, index, saved = [], 0, 0
        while saved < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.step == 0:
                path = Path(tmp) / f"f{saved:04d}.png"
                cv2.imwrite(str(path), frame)
                kept.append(path)
                saved += 1
            index += 1
        cap.release()
        if not kept:
            raise SystemExit("no frames kept")
        # the sparse arm, coded all-intra with the same encoder
        intra_bytes = encode(kept, fps / args.step, args.crf, all_intra=True)
        # the same frames coded with inter prediction allowed, as a control on how
        # little there is to predict at this spacing
        inter_bytes = encode(kept, fps / args.step, args.crf, all_intra=False)
        # the full-rate arm through the same encoder at the same quality, so the
        # two rows differ only in which frames are sent
        cap2 = cv2.VideoCapture(str(args.video))
        dense = []
        for i in range(args.full_frames):
            ok, frame = cap2.read()
            if not ok:
                break
            path = Path(tmp) / f"d{i:04d}.png"
            cv2.imwrite(str(path), frame)
            dense.append(path)
        cap2.release()
        dense_bytes = encode(dense, fps, args.crf, all_intra=False)

    sparse_rate = fps / args.step
    per_frame_intra = intra_bytes / len(kept)
    payload = {
        "video": args.video.name,
        "resolution": [width, height], "fps": round(fps, 2),
        "frames_total": total, "duration_s": round(duration, 2),
        "step": args.step, "kept_frames": len(kept),
        "sparse_rate_hz": round(sparse_rate, 3),
        "crf": args.crf,
        "source_mbit_s": round(source_bytes * 8 / duration / 1e6, 2),
        "fullrate_same_codec_mbit_s": round(dense_bytes / len(dense) * 8 * fps / 1e6, 2),
        "fullrate_frames_encoded": len(dense),
        "allintra_mbit_s_at_sparse_rate": round(per_frame_intra * 8 * sparse_rate / 1e6, 2),
        "interframe_mbit_s_at_sparse_rate": round(
            inter_bytes / len(kept) * 8 * sparse_rate / 1e6, 2),
        "allintra_bytes_per_frame": round(per_frame_intra),
    }
    payload["byte_saving_vs_source"] = round(
        1 - payload["allintra_mbit_s_at_sparse_rate"] / payload["source_mbit_s"], 4)
    payload["byte_saving_same_codec"] = round(
        1 - payload["allintra_mbit_s_at_sparse_rate"] / payload["fullrate_same_codec_mbit_s"], 4)
    payload["frame_saving"] = round(1 - 1 / args.step, 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    for k, v in payload.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
