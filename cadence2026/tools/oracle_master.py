#!/usr/bin/env python3
"""Oracle counting surfaces, master rerun. No detector, no GPU.

This supersedes oracle_count_surface.py for the paper's oracle evidence. Three
things changed, each because a reviewer objection could not be answered with the
old artefact.

DENSE WINDOW GRID. The old grid stopped at 700 frames while five sequences hold
890-900. Coverage >= 0.8 is only reached near the end of a sequence, so for those
five videos exactly one cell survived the filter and the drift fit had nothing to
fit. The grid is a set of snapshot points taken during a single tracker pass, so
making it dense costs nothing.

WINDOW ENUMERATION IS DECLARED. The old surface was prefix-only ("count over the
first L frames") while the text also spoke of sliding windows. Here all three
definitions are computed from the same tracker pass and reported side by side:
prefix, every legal sliding start, and non-overlapping blocked windows. The last
is what a held-out validation of the drift model needs, because prefix windows
are nested and share almost all their data.

FOUR WAYS TO MISS A BUNCH, not one. i.i.d. per-frame dropout interrupts an object
that the tracker keeps seeing; occlusion hides it for a run of frames; a weak
detector never registers some bunches at all. Those are different perturbations
and there is no reason they move a distinct-track count the same way:

  bernoulli  each observation dropped independently          (interruption)
  block      an identity hidden for block_len frames          (occlusion)
  identity   a random subset of identities dropped entirely   (never detected)
  size       the smallest identities dropped entirely         (weak detector)

Only the last two can remove an identity from the count rather than fragment it,
so only they can push the count down. The paper's cancellation account needs that
distinction; the old Bernoulli-only control could not make it.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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

DENSE_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200,
              250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]


def video_frames(root: Path, video: str) -> list[Path]:
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
            first = row["resolution"].split(";")[0].split(":")[0]
            width, height = (int(value) for value in first.split("x"))
            sizes[row["video"]] = (width, height)
    return sizes


def identity_areas(frames, width, height) -> dict[int, float]:
    """Median box area per ground-truth identity, in pixels."""
    areas: dict[int, list[float]] = defaultdict(list)
    for path in frames:
        ids, boxes = load_gt_tracks(path, width, height)
        for track, box in zip(ids, boxes):
            areas[int(track)].append(float((box[2] - box[0]) * (box[3] - box[1])))
    return {track: float(np.median(values)) for track, values in areas.items()}


def choose_dropped_identities(mode, areas, miss, rng) -> set[int]:
    """Identities removed for the whole sequence, under the two entity-level modes."""
    if miss <= 0 or not areas:
        return set()
    order = sorted(areas)
    keep_out = int(round(miss * len(order)))
    if keep_out <= 0:
        return set()
    if mode == "identity":
        return set(rng.choice(np.array(order), size=keep_out, replace=False).tolist())
    # 'size': a weak detector loses the smallest targets first, deterministically
    by_area = sorted(order, key=lambda track: areas[track])
    return set(by_area[:keep_out])


def run_sequence(frames, size, tracker_cfg, miss, rng, frame_rate=15,
                 miss_mode="bernoulli", block_len=15, dropped_ids=frozenset()):
    """Track thinned ground-truth boxes; return per-frame predicted and true ids.

    The ground-truth side is never thinned: G is what the sequence actually
    contains, so an identity removed from the observations is still counted in
    the reference. That is the point of the entity-level modes -- they can drive
    the reported count below the truth, which frame-level dropout cannot.
    """
    width, height = size
    tracker = build_tracker(tracker_cfg, frame_rate=frame_rate)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    frame_pred_ids: list[list[int]] = []
    frame_gt_ids: list[list[int]] = []
    blocked_until: dict[int, int] = {}

    for index, path in enumerate(frames):
        gt_ids, gt_boxes = load_gt_tracks(path, width, height)
        if not gt_ids:
            keep = np.zeros(0, dtype=bool)
        elif miss_mode == "block":
            keep_list = []
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
        elif miss_mode in ("identity", "size"):
            keep = np.array([int(t) not in dropped_ids for t in gt_ids], dtype=bool)
        else:
            keep = rng.random(len(gt_ids)) >= miss
        boxes = gt_boxes[keep] if len(gt_boxes) else gt_boxes
        scores = np.full((len(boxes), 1), 0.9, dtype=np.float32)
        classes = np.zeros((len(boxes), 1), dtype=np.float32)
        data = (np.concatenate([boxes.astype(np.float32), scores, classes], axis=1)
                if len(boxes) else np.empty((0, 6), dtype=np.float32))
        tracks = tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32),
                                      (height, width)), blank)
        frame_pred_ids.append([int(t) for t in tracks[:, 4]] if len(tracks) else [])
        frame_gt_ids.append([int(t) for t in gt_ids])
    return frame_pred_ids, frame_gt_ids


def prefix_surface(frame_pred_ids, frame_gt_ids, lengths, min_lens):
    """Counts over the first L frames: what you report after flying L frames."""
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
                "window_frames": length, "min_track_len": tau,
                "predicted_tracks": predicted, "gt_tracks": truth,
                "signed_relative_error": (predicted - truth) / truth if truth else None,
            })
    return grid


def _window_counts(frame_ids, length, taus):
    """Number of ids reaching each tau, for every sliding start, in one pass.

    A window is entered by adding its right-hand frame and leaving its left-hand
    frame, so the whole sweep costs one increment per observation rather than one
    recount per window. Returns {tau: [count for each start]}.
    """
    total = len(frame_ids)
    if length > total:
        return None
    counts: Counter[int] = Counter()
    reaching = {tau: 0 for tau in taus}

    def add(track):
        counts[track] += 1
        value = counts[track]
        for tau in taus:
            if value == tau:
                reaching[tau] += 1

    def drop(track):
        value = counts[track]
        for tau in taus:
            if value == tau:
                reaching[tau] -= 1
        if value == 1:
            del counts[track]
        else:
            counts[track] = value - 1

    for index in range(length):
        for track in frame_ids[index]:
            add(track)
    series = {tau: [reaching[tau]] for tau in taus}
    for start in range(1, total - length + 1):
        for track in frame_ids[start - 1]:
            drop(track)
        for track in frame_ids[start + length - 1]:
            add(track)
        for tau in taus:
            series[tau].append(reaching[tau])
    return series


def sliding_surface(frame_pred_ids, frame_gt_ids, lengths, min_lens):
    """Every legal sliding start, no subsampling, with the spread across starts."""
    grid = []
    total = len(frame_pred_ids)
    for length in sorted({length for length in lengths if length <= total}):
        pred = _window_counts(frame_pred_ids, length, min_lens)
        truth = _window_counts(frame_gt_ids, length, [1])[1]
        for tau in min_lens:
            errors = [(p - g) / g for p, g in zip(pred[tau], truth) if g]
            if not errors:
                continue
            values = np.array(errors)
            grid.append({
                "window_frames": length, "min_track_len": tau,
                "n_windows": len(errors),
                "mean_error": float(values.mean()),
                "median_error": float(np.median(values)),
                "iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
                "min_error": float(values.min()), "max_error": float(values.max()),
                "mean_predicted": float(np.mean(pred[tau])),
                "mean_gt": float(np.mean(truth)),
            })
    return grid


def blocked_surface(frame_pred_ids, frame_gt_ids, lengths, min_lens):
    """Non-overlapping windows tiling the sequence: the only independent samples.

    Prefix windows are nested and sliding windows overlap by L-1 frames, so
    neither can validate a model fitted on the same sequence. These can.
    """
    grid = []
    total = len(frame_pred_ids)
    for length in sorted({length for length in lengths if length * 2 <= total}):
        blocks = total // length
        cells = defaultdict(list)
        for block in range(blocks):
            lo, hi = block * length, (block + 1) * length
            counts: Counter[int] = Counter()
            for index in range(lo, hi):
                counts.update(frame_pred_ids[index])
            truth = len({t for index in range(lo, hi) for t in frame_gt_ids[index]})
            for tau in min_lens:
                predicted = sum(seen >= tau for seen in counts.values())
                cells[tau].append((predicted, truth))
        for tau in min_lens:
            pairs = [(p, g) for p, g in cells[tau] if g]
            if not pairs:
                continue
            grid.append({
                "window_frames": length, "min_track_len": tau,
                "n_windows": len(pairs),
                "mean_excess": float(np.mean([p - g for p, g in pairs])),
                "mean_error": float(np.mean([(p - g) / g for p, g in pairs])),
                "windows": [{"predicted": int(p), "gt": int(g)} for p, g in pairs],
            })
    return grid


def fit_drift(grid, min_lens):
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
        fits[str(tau)] = {
            "phi_per_frame": float(phi),
            "delta_tracks": float(-negative_delta),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        }
    return fits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets/grapemots_det_721"))
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--miss-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--miss-mode", choices=["bernoulli", "block", "identity", "size"],
                    default="bernoulli")
    ap.add_argument("--block-len", type=int, default=15)
    ap.add_argument("--window-lengths", type=int, nargs="+", default=DENSE_GRID)
    ap.add_argument("--min-track-lens", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--sliding-at", type=float, nargs="+", default=[0.0, 0.2],
                    help="miss rates for which every sliding start is enumerated; "
                         "the sweep is exact but costs a pass per window length")
    ap.add_argument("--dump-frame-ids", type=Path,
                    help="gzipped per-frame ids for seed 0, for offline reanalysis")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sizes = read_resolutions(args.root)
    videos = args.videos or sorted(sizes)
    records, dumps = [], {}

    for video in videos:
        frames = video_frames(args.root, video)
        areas = (identity_areas(frames, *sizes[video])
                 if args.miss_mode in ("identity", "size") else {})
        print(f"{video}: {len(frames)} annotated frames, "
              f"{len(areas) if areas else '-'} identities", flush=True)
        for miss in args.miss_rates:
            # p=0 is deterministic, and 'size' drops a fixed set, so one seed each
            seeds = args.seeds if (miss > 0 and args.miss_mode != "size") else args.seeds[:1]
            for seed in seeds:
                rng = np.random.default_rng(seed)
                dropped = choose_dropped_identities(args.miss_mode, areas, miss, rng)
                pred_ids, gt_ids = run_sequence(
                    frames, sizes[video], args.tracker, miss, rng,
                    miss_mode=args.miss_mode, block_len=args.block_len,
                    dropped_ids=frozenset(dropped))
                prefix = prefix_surface(pred_ids, gt_ids,
                                        args.window_lengths, args.min_track_lens)
                record = {
                    "video": video, "frames": len(frames), "tracker": args.tracker,
                    "miss_rate": miss, "miss_mode": args.miss_mode, "seed": seed,
                    "dropped_identities": len(dropped),
                    "total_identities": len(areas) if areas else None,
                    "count_error_surface": prefix,
                    "blocked_surface": blocked_surface(
                        pred_ids, gt_ids, args.window_lengths, args.min_track_lens),
                    "drift_fit": fit_drift(prefix, args.min_track_lens),
                }
                if any(abs(miss - rate) < 1e-9 for rate in args.sliding_at) and seed == args.seeds[0]:
                    record["sliding_surface"] = sliding_surface(
                        pred_ids, gt_ids, args.window_lengths, args.min_track_lens)
                records.append(record)
                if args.dump_frame_ids is not None and seed == args.seeds[0]:
                    dumps[f"{video}|p{miss}"] = {"pred": pred_ids, "gt": gt_ids}
                tail = prefix[-1]
                print(f"  p={miss:.2f} seed={seed} dropped={len(dropped)}: "
                      f"L={tail['window_frames']} tau={tail['min_track_len']} "
                      f"err={tail['signed_relative_error']:+.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "root": str(args.root), "tracker": args.tracker,
            "miss_mode": args.miss_mode, "block_len": args.block_len,
            "miss_rates": args.miss_rates, "seeds": args.seeds,
            "window_lengths": args.window_lengths,
            "min_track_lens": args.min_track_lens,
            "sliding_at": args.sliding_at,
        },
        "runs": records,
    }, indent=1) + "\n")
    print(f"wrote {args.out}  ({len(records)} runs)")

    if args.dump_frame_ids is not None:
        args.dump_frame_ids.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.dump_frame_ids, "wt") as handle:
            json.dump(dumps, handle)
        print(f"wrote {args.dump_frame_ids}  ({len(dumps)} sequences)")


if __name__ == "__main__":
    main()
