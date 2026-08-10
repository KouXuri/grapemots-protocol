#!/usr/bin/env python3
"""Choose an annotation frame step from a short densely-labelled pilot clip.

Anyone about to annotate a new UAV video dataset has to pick a frame step, and
the literature offers no basis for the choice. The two public vineyard releases
show what is at stake: GrapeMOTS chose step 2 and its sequences support
association, while the earlier release samples one frame in 12 to 75 and
consecutive boxes of one bunch do not overlap at all, which no amount of extra
training data repairs.

The quantity that decides it is how far a target moves between two labelled
frames relative to its own size. Thinning GrapeMOTS' annotation by factors of 1
to 64 shows that this ratio is linear in the interval to R^2 = 0.999, so it can
be written

    displacement / size  =  c * dt          c = relative speed / target size [1/s]

with c a property of the flight, not of the camera or the resolution: doubling
the pixel scale changes displacement and size together. Measure c once on a
pilot clip and the frame step follows for the whole campaign.

The thresholds are read off the same thinning experiment, where the ratio is the
variable that tracks when association starts to fail:

    <= 0.20   median consecutive IoU about 0.64, comfortable
    <= 0.40   about 0.40, workable
    >= 0.75   about 0.10, marginal
    >= 1.5    0.00, an IoU-gated associator has nothing to work with

Input is a handful of consecutive annotated frames -- thirty is plenty -- in the
same sidecar layout the rest of these tools use.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_grapemots_mot import load_gt_tracks  # noqa: E402

FRAME_RE = re.compile(r"__frame_(\d+)\.txt$")
BANDS = [(0.20, "comfortable"), (0.40, "workable"), (0.75, "marginal")]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def measure(frames: list[Path], size: tuple[int, int], lag: int):
    """Displacement / target size, and IoU, between frames `lag` apart."""
    width, height = size
    per_frame = []
    for path in frames:
        ids, boxes = load_gt_tracks(path, width, height)
        per_frame.append({int(t): box for t, box in zip(ids, boxes)})
    steps, overlaps = [], []
    for index in range(len(per_frame) - lag):
        a, b = per_frame[index], per_frame[index + lag]
        for track, box_a in a.items():
            box_b = b.get(track)
            if box_b is None:
                continue
            scale = np.sqrt(max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
            centre_a = np.array([(box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2])
            centre_b = np.array([(box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2])
            steps.append(float(np.linalg.norm(centre_b - centre_a) / scale))
            overlaps.append(iou(box_a, box_b))
    return np.asarray(steps), np.asarray(overlaps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="track root holding tracks/<split>/<video>__frame_<N>.txt")
    ap.add_argument("--video", required=True)
    ap.add_argument("--size", type=int, nargs=2, required=True, metavar=("W", "H"))
    ap.add_argument("--source-fps", type=float, required=True,
                    help="frame rate of the SOURCE video the pilot clip was labelled from")
    ap.add_argument("--pilot-step", type=int, default=1,
                    help="frame step already used in the pilot clip; 1 means every frame")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8],
                    help="intervals, in pilot frames, at which to measure")
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="quantile of the per-target ratio used for c; 0.5 reproduces the\n                         reported calibration, 0.9 protects the fastest and smallest targets")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    frames: dict[int, Path] = {}
    for split in ("train", "val", "test"):
        directory = args.root / "tracks" / split
        if directory.is_dir():
            for path in directory.glob(f"{args.video}__frame_*.txt"):
                frames.setdefault(int(FRAME_RE.search(path.name).group(1)), path)
    if len(frames) < max(args.lags) + 2:
        raise SystemExit(f"need at least {max(args.lags) + 2} annotated frames, found {len(frames)}")
    ordered = [frames[key] for key in sorted(frames)]

    print(f"{args.video}: {len(ordered)} pilot frames, pilot step {args.pilot_step}, "
          f"source {args.source_fps:g} fps\n")
    print(f"{'lag':>4} {'dt (s)':>8} {'src frames':>11} {'step/size':>10} {'IoU med':>9} "
          f"{'IoU<0.3':>8} {'c (1/s)':>9}")
    records, constants, conservative = [], [], []
    for lag in args.lags:
        steps, overlaps = measure(ordered, tuple(args.size), lag)
        if steps.size == 0:
            continue
        dt = lag * args.pilot_step / args.source_fps
        ratio = float(np.quantile(steps, args.quantile))
        constant = ratio / dt
        constants.append(constant)
        conservative.append(float(np.quantile(steps, 0.9)) / dt)
        records.append({"lag": lag, "dt_seconds": dt, "source_frames": lag * args.pilot_step,
                        "step_over_size_median": ratio,
                        "iou_median": float(np.median(overlaps)),
                        "iou_below_0p3": float(np.mean(overlaps < 0.3)),
                        "c_per_second": constant,
                        "c_per_second_p90": conservative[-1],
                        "pairs": int(steps.size)})
        print(f"{lag:>4} {dt:>8.4f} {lag * args.pilot_step:>11} {ratio:>10.3f} "
              f"{np.median(overlaps):>9.3f} {np.mean(overlaps < 0.3):>8.3f} {constant:>9.2f}")

    # c should be constant across lags; if it is not, the clip is not a straight
    # constant-speed pass and the recommendation below does not transfer.
    c = float(np.median(constants))
    spread = float(np.max(constants) / np.min(constants)) if constants else float("nan")
    c90 = float(np.median(conservative)) if conservative else float("nan")
    print(f"\nc = {c:.2f} /s   (spread across lags {spread:.2f}x; "
          f"{'consistent' if spread < 1.5 else 'NOT consistent -- the clip is not a steady pass'})")
    print(f"c at the 90th percentile of targets = {c90:.2f} /s. A design that must not fail\n"
          f"on the fastest and smallest targets should use this one: it gives\n"
          f"dt_max = {0.20 / c90:.4f} s in the comfortable band, against {0.20 / c:.4f} s at the median.")

    print(f"\n{'band':>13} {'target':>7} {'dt max (s)':>11} {'source frames':>14}")
    recommendation = {}
    for target, label in BANDS:
        dt = target / c
        recommendation[label] = {"target_step_over_size": target, "dt_max_seconds": dt,
                                 "source_frame_step": dt * args.source_fps}
        print(f"{label:>13} {target:>7.2f} {dt:>11.4f} {dt * args.source_fps:>14.1f}")

    workable = recommendation["workable"]["source_frame_step"]
    comfortable = recommendation["comfortable"]["source_frame_step"]
    print(f"\nRECOMMENDATION: annotate every {max(1, int(comfortable)):d} source frames "
          f"(never sparser than every {max(1, int(workable)):d}).")
    print(f"At {args.source_fps:g} fps that is {args.source_fps / max(1, comfortable):.1f} Hz, "
          f"against the 1 Hz floor below which tracking annotation is generally held infeasible.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "video": args.video, "source_fps": args.source_fps,
            "pilot_step": args.pilot_step, "pilot_frames": len(ordered),
            "c_per_second": c, "c_spread": spread,
            "quantile": args.quantile, "c_per_second_p90": c90,
            "measurements": records, "recommendation": recommendation,
        }, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
