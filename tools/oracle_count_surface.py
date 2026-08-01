#!/usr/bin/env python3
"""Counting drift under a perfect detector, on all 11 GrapeMOTS videos.

The main experiment measures how a reported bunch count varies with the length
of the evaluated clip and with the minimum-track-length filter, using our own
detector on the two test videos.  A reviewer can answer that with "your detector
is weak" or "two sequences prove nothing".  This script removes both objections.

Detections here are the ground-truth boxes themselves, thinned by an explicit
per-frame miss probability p.  There is no detector and no appearance model, so
whatever drift survives is a property of counting distinct track identities over
a temporal window -- not of our model.  Sweeping p sweeps the identity-break rate
phi, which is the free parameter of the drift model, and every one of the 11
videos can be used because no video was ever trained on.

Model (paper Sec. III): with window length L and minimum track length tau,
    P(L,tau) ~= G(L) + phi(tau) * L - delta(tau)
so the signed relative error E = (P - G) / G crosses zero at L* = delta / phi.
We fit phi and delta by regressing P - G on L, then check the predicted L*
against the measured crossing.

Output: one JSON holding the (L x tau) grid and the fit for every
(video, tracker, miss rate, seed).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Boxes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_grapemots_mot import build_tracker, load_gt_tracks  # noqa: E402

FRAME_RE = re.compile(r"__frame_(\d+)\.txt$")


def video_frames(root: Path, video: str) -> list[Path]:
    """Every annotated frame of a video, in temporal order, across all splits."""
    found: list[Path] = []
    for split in ("train", "val", "test"):
        found.extend((root / "tracks" / split).glob(f"{video}__frame_*.txt"))
    if not found:
        raise SystemExit(f"no track sidecars for {video} under {root}")
    return sorted(found, key=lambda p: int(FRAME_RE.search(p.name).group(1)))


def read_resolutions(root: Path) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    with (root / "manifest.csv").open() as handle:
        for row in csv.DictReader(handle):
            # "3840x2160:890" or several "WxH:n" segments joined by ';'
            first = row["resolution"].split(";")[0].split(":")[0]
            width, height = (int(value) for value in first.split("x"))
            sizes[row["video"]] = (width, height)
    return sizes


def run_sequence(frames, size, tracker_cfg, miss, jitter, rng, frame_rate=15,
                 miss_mode="bernoulli", block_len=15):
    """Track thinned ground-truth boxes; return per-frame predicted and true ids."""
    width, height = size
    tracker = build_tracker(tracker_cfg, frame_rate=frame_rate)
    blank = np.zeros((height, width, 3), dtype=np.uint8)  # unused when gmc is off
    frame_pred_ids: list[list[int]] = []
    frame_gt_ids: list[list[int]] = []

    # Occlusion hides a bunch for a run of consecutive frames, not independently
    # each frame. In block mode an identity, once dropped, stays dropped for
    # block_len frames, which is what a leaf passing in front actually does.
    blocked_until: dict[int, int] = {}

    for index, path in enumerate(frames):
        gt_ids, gt_boxes = load_gt_tracks(path, width, height)
        if not gt_ids:
            keep = np.zeros(0, dtype=bool)
        elif miss_mode == "block":
            keep_list = []
            # per-frame hazard chosen so the expected fraction of hidden frames
            # matches `miss` for a mean visibility run of block_len
            hazard = miss / max(block_len * (1.0 - miss), 1e-9)
            for track in gt_ids:
                if blocked_until.get(track, -1) > index:
                    keep_list.append(False)
                elif rng.random() < min(hazard, 1.0):
                    blocked_until[track] = index + block_len
                    keep_list.append(False)
                else:
                    keep_list.append(True)
            keep = np.array(keep_list, dtype=bool)
        else:
            keep = rng.random(len(gt_ids)) >= miss
        boxes = gt_boxes[keep] if len(gt_boxes) else gt_boxes
        if jitter and len(boxes):
            spans = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
            offset = rng.normal(0.0, jitter, size=(len(boxes), 2)) * spans
            boxes = boxes + np.concatenate([offset, offset], axis=1)
        scores = np.full((len(boxes), 1), 0.9, dtype=np.float32)
        classes = np.zeros((len(boxes), 1), dtype=np.float32)
        data = (np.concatenate([boxes.astype(np.float32), scores, classes], axis=1)
                if len(boxes) else np.empty((0, 6), dtype=np.float32))
        tracks = tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32),
                                      (height, width)), blank)
        frame_pred_ids.append([int(t) for t in tracks[:, 4]] if len(tracks) else [])
        frame_gt_ids.append(list(gt_ids))
    return frame_pred_ids, frame_gt_ids


def prefix_surface(frame_pred_ids, frame_gt_ids, lengths, min_lens):
    """Counts over the first L frames -- what you would report after flying L frames."""
    grid = []
    total = len(frame_pred_ids)
    pred_seen: Counter[int] = Counter()
    gt_seen: set[int] = set()
    wanted = {length for length in lengths if length <= total} | {total}
    snapshots: dict[int, tuple[Counter, int]] = {}
    for index in range(total):
        pred_seen.update(frame_pred_ids[index])
        gt_seen.update(frame_gt_ids[index])
        if index + 1 in wanted:
            snapshots[index + 1] = (Counter(pred_seen), len(gt_seen))
    for length in sorted(snapshots):
        counts, truth = snapshots[length]
        for tau in min_lens:
            predicted = sum(seen >= tau for seen in counts.values())
            grid.append({
                "window_frames": length,
                "min_track_len": tau,
                "predicted_tracks": predicted,
                "gt_tracks": truth,
                "signed_relative_error": (predicted - truth) / truth if truth else None,
            })
    return grid


def fit_drift(grid, min_lens):
    """Regress P - G on L for each tau; L* = delta / phi is the zero-error length."""
    fits = {}
    for tau in min_lens:
        points = [(row["window_frames"], row["predicted_tracks"] - row["gt_tracks"])
                  for row in grid if row["min_track_len"] == tau and row["gt_tracks"]]
        if len(points) < 3:
            continue
        lengths = np.array([p[0] for p in points], dtype=float)
        excess = np.array([p[1] for p in points], dtype=float)
        phi, negative_delta = np.polyfit(lengths, excess, 1)
        predicted = phi * lengths + negative_delta
        ss_res = float(((excess - predicted) ** 2).sum())
        ss_tot = float(((excess - excess.mean()) ** 2).sum())

        errors = [(row["window_frames"], row["signed_relative_error"])
                  for row in grid if row["min_track_len"] == tau
                  and row["signed_relative_error"] is not None]
        observed = None
        for (left_len, left), (right_len, right) in zip(errors, errors[1:]):
            if left == 0:
                observed = left_len
                break
            if left * right < 0:  # sign change between two measured lengths
                observed = left_len + (right_len - left_len) * abs(left) / (abs(left) + abs(right))
                break
        fits[str(tau)] = {
            "phi_per_frame": float(phi),
            "delta_tracks": float(-negative_delta),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
            "predicted_zero_error_length": float(-negative_delta / phi) if phi > 0 else None,
            "observed_zero_error_length": observed,
        }
    return fits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets/grapemots_det_721"))
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack.yaml"],
                    help="tracker yamls; BoT-SORT must have gmc_method: none here, "
                         "because no real imagery is passed to the tracker")
    ap.add_argument("--miss-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="box centre noise as a fraction of box size")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # identity/size need the whole-sequence identity draw, which lives in
    # oracle_master.py; use that script for those two modes.
    ap.add_argument("--miss-mode", choices=["bernoulli", "block"], default="bernoulli",
                    help="'block' hides an identity for a run of frames, which is "
                         "what occlusion does; i.i.d. dropout is the control")
    ap.add_argument("--block-len", type=int, default=15,
                    help="frames an identity stays hidden once occluded")
    # The 28 lengths the paper reports. The earlier 11-length default did not
    # match the manuscript and is kept only in the git history.
    ap.add_argument("--window-lengths", type=int, nargs="+",
                    default=[5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150,
                             175, 200, 250, 300, 350, 400, 450, 500, 550, 600,
                             650, 700, 750, 800, 850, 900])
    ap.add_argument("--min-track-lens", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sizes = read_resolutions(args.root)
    videos = args.videos or sorted(sizes)
    records = []

    for video in videos:
        frames = video_frames(args.root, video)
        print(f"{video}: {len(frames)} annotated frames", flush=True)
        for tracker_cfg in args.trackers:
            for miss in args.miss_rates:
                seeds = args.seeds if miss > 0 else args.seeds[:1]  # p=0 is deterministic
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    pred_ids, gt_ids = run_sequence(
                        frames, sizes[video], tracker_cfg, miss, args.jitter, rng,
                        miss_mode=args.miss_mode, block_len=args.block_len
                    )
                    grid = prefix_surface(pred_ids, gt_ids,
                                          args.window_lengths, args.min_track_lens)
                    records.append({
                        "video": video,
                        "frames": len(frames),
                        "tracker": tracker_cfg,
                        "miss_rate": miss,
                        "miss_mode": args.miss_mode,
                        "jitter": args.jitter,
                        "seed": seed,
                        "count_error_surface": grid,
                        "drift_fit": fit_drift(grid, args.min_track_lens),
                    })
                    tail = grid[-1]
                    print(f"  {tracker_cfg} p={miss:.2f} seed={seed}: "
                          f"L={tail['window_frames']} tau={tail['min_track_len']} "
                          f"err={tail['signed_relative_error']:+.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "root": str(args.root), "trackers": args.trackers,
            "miss_rates": args.miss_rates, "jitter": args.jitter, "seeds": args.seeds,
            "window_lengths": args.window_lengths, "min_track_lens": args.min_track_lens,
        },
        "runs": records,
    }, indent=1) + "\n")
    print(f"wrote {args.out}  ({len(records)} runs)")


if __name__ == "__main__":
    main()
