#!/usr/bin/env python3
"""Can the interval constant be read off a tracker's own output?

The criterion needs c = r / dt. Reading it from a tracker's tracks costs no
annotation, which would make the criterion free to apply on any footage. This
checks whether that estimate is usable, by splitting the reference trajectories
of a sequence into the ones the tracker covered and the ones it lost, and
measuring c on each side.

    python3 tools/c_estimator_bias.py

Writes runs/c_bias_0814/results/c_estimator_bias.json. No GPU: it reads the
released annotation and a stored tracks.csv.
"""
from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
FPS = 59.94
PAIRS = (("row_8_3", "p3_tiled_row8_3_botsort_stride1"),
         ("row_4.3_2", "p3_tiled_row4_3_2_botsort_stride1"))


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def displacement_ratio(b0, b1) -> float:
    """r: centre displacement in units of the earlier box's size."""
    c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    return math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / math.sqrt((b0[2] - b0[0]) * (b0[3] - b0[1]))


def load_reference(sequence: str):
    frames: dict[int, dict[int, tuple]] = collections.defaultdict(dict)
    path = ROOT / "datasets/bodegas_grape_bunch_seg/gt_tracks_from_mots.csv"
    for row in csv.DictReader(path.open()):
        if row["sequence"] != sequence:
            continue
        frames[int(row["gt_track_id"])][int(row["frame_index"])] = tuple(
            float(row[k]) for k in ("x1", "y1", "x2", "y2"))
    return frames


def load_predictions(run: str):
    by_frame: dict[int, list[tuple]] = collections.defaultdict(list)
    by_track: dict[int, dict[int, tuple]] = collections.defaultdict(dict)
    path = ROOT / "runs/segment/runs_bodegas_video" / run / "tracks.csv"
    for row in csv.DictReader(path.open()):
        frame = int(row["source_frame_zero_based"])
        box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        by_frame[frame].append(box)
        by_track[int(row["track_id"])][frame] = box
    return by_frame, by_track


def c_over(trajectories, ids, annotated_to_source) -> tuple[float, int]:
    values = []
    for tid in ids:
        frames = sorted(trajectories[tid])
        for f0, f1 in zip(frames, frames[1:]):
            dt = (annotated_to_source[f1] - annotated_to_source[f0]) / FPS
            if dt > 0:
                values.append(displacement_ratio(trajectories[tid][f0], trajectories[tid][f1]) / dt)
    return (float(np.median(values)), len(values)) if values else (float("nan"), 0)


def analyse(sequence: str, run: str) -> dict:
    alignment = json.loads(
        (ROOT / "grapemots-protocol/cadence2026/results/bodegas_alignment_all28.json").read_text()
    )["sequences"][sequence]
    annotated_to_source = {int(k): v for k, v in alignment["annotated_to_source"].items()}
    reference = load_reference(sequence)
    by_frame, by_track = load_predictions(run)

    covered: set[int] = set()
    for annotated, source in annotated_to_source.items():
        truth = [(tid, *boxes[annotated]) for tid, boxes in reference.items() if annotated in boxes]
        predicted = by_frame.get(source, [])
        if not truth or not predicted:
            continue
        scores = np.array([[iou(t[1:], p) for p in predicted] for t in truth])
        for i, j in zip(*linear_sum_assignment(-scores)):
            if scores[i, j] >= 0.5:
                covered.add(truth[i][0])
    everything = set(reference)
    lost = everything - covered

    c_all, n_all = c_over(reference, everything, annotated_to_source)
    c_kept, n_kept = c_over(reference, covered, annotated_to_source)
    c_lost, n_lost = c_over(reference, lost, annotated_to_source)

    # what the tracker's own surviving tracks would report, over several lags
    from_tracker = {}
    for lag in (5, 15, 30, 60):
        values = [displacement_ratio(boxes[f], boxes[f + lag]) / (lag / FPS)
                  for boxes in by_track.values() for f in boxes if f + lag in boxes]
        if len(values) >= 30:
            from_tracker[lag] = float(np.median(values))
    c_tracker = float(np.median(list(from_tracker.values()))) if from_tracker else float("nan")

    def interval(c: float, theta: float = 0.20) -> int:
        return math.floor(theta * FPS / c) if c > 0 else 0

    return {
        "sequence": sequence, "run": run,
        "reference_trajectories": len(everything),
        "covered_by_tracker": len(covered), "lost_by_tracker": len(lost),
        "c_all_reference": round(c_all, 3), "pairs_all": n_all,
        "c_kept_by_tracker": round(c_kept, 3), "pairs_kept": n_kept,
        "c_lost_by_tracker": round(c_lost, 3), "pairs_lost": n_lost,
        "lost_over_kept": round(c_lost / c_kept, 3) if c_kept else None,
        "c_from_tracker_output": round(c_tracker, 3),
        "c_from_tracker_by_lag": {k: round(v, 3) for k, v in from_tracker.items()},
        "N_from_reference_theta020": interval(c_all),
        "N_from_tracker_theta020": interval(c_tracker),
    }


def main() -> None:
    rows = [analyse(seq, run) for seq, run in PAIRS]
    out = ROOT / "runs/c_bias_0814/results"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": FPS, "theta": 0.20,
        "question": "is c measurable from a tracker's own output, without annotation?",
        "sequences": rows,
        "ratio_range": [min(r["lost_over_kept"] for r in rows),
                        max(r["lost_over_kept"] for r in rows)],
    }
    (out / "c_estimator_bias.json").write_text(json.dumps(payload, indent=1))
    for r in rows:
        print(f"{r['sequence']}: {r['covered_by_tracker']}/{r['reference_trajectories']} covered; "
              f"c kept {r['c_kept_by_tracker']}, lost {r['c_lost_by_tracker']} "
              f"(x{r['lost_over_kept']}); N ref {r['N_from_reference_theta020']} vs "
              f"tracker {r['N_from_tracker_theta020']}")
    print(f"wrote {out / 'c_estimator_bias.json'}")


if __name__ == "__main__":
    main()
