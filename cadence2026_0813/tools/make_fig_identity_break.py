#!/usr/bin/env python3
"""Draw one reference trajectory that the tracker issued more than one identity.

Nothing here is placed by hand. The duplicate is found by the ownership rule the
paper defines in the decomposition section --- per-frame one-to-one matching at
IoU 0.5, a predicted track owned by the trajectory it covers in the most frames
--- so the instance drawn is one of the D duplicates the tables count, and the
boxes are the released annotation and the tracker's own output at those frames.

    python3 tools/make_fig_identity_break.py

Writes figures/fig_identity_break_data.{pdf,png} and, beside it, a JSON record of
the instance chosen so the figure can be audited without re-running the search.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperstyle import apply as apply_paper_style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
C_GT = "#2166ac"      # reference, the same blue the plotted figures use
C_PRED = "#b2182b"    # tracker output, the same red
INK = "#17232D"
PAPER = "#FFFFFF"
LINE = "#C7D0D5"
CROP = "#E8A33D"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="row_8_3")
    p.add_argument("--run", default="p3_tiled_row8_3_botsort_stride1")
    p.add_argument("--out", type=Path, default=ROOT / "figures/fig_identity_break_data")
    return p.parse_args()


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_reference(sequence: str) -> dict[int, list[tuple]]:
    frames: dict[int, list[tuple]] = collections.defaultdict(list)
    path = ROOT / "datasets/bodegas_grape_bunch_seg/gt_tracks_from_mots.csv"
    for row in csv.DictReader(path.open()):
        if row["sequence"] != sequence:
            continue
        frames[int(row["frame_index"])].append((
            int(row["gt_track_id"]), float(row["x1"]), float(row["y1"]),
            float(row["x2"]), float(row["y2"])))
    return frames


def load_predictions(run: str) -> dict[int, list[tuple]]:
    frames: dict[int, list[tuple]] = collections.defaultdict(list)
    path = ROOT / "runs/segment/runs_bodegas_video" / run / "tracks.csv"
    for row in csv.DictReader(path.open()):
        frames[int(row["source_frame_zero_based"])].append((
            int(row["track_id"]), float(row["x1"]), float(row["y1"]),
            float(row["x2"]), float(row["y2"])))
    return frames


def find_duplicate(sequence: str, run: str) -> dict:
    """Apply the paper's ownership rule and return the widest-spanning duplicate."""
    alignment = json.loads(
        (ROOT / "grapemots-protocol/cadence2026/results/bodegas_alignment_all28.json").read_text()
    )["sequences"][sequence]
    annotated_to_source = {int(k): v for k, v in alignment["annotated_to_source"].items()}
    reference = load_reference(sequence)
    predictions = load_predictions(run)

    covered: collections.Counter = collections.Counter()
    frames_of: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    boxes_of: dict[tuple[int, int, int], tuple] = {}
    for annotated, source in sorted(annotated_to_source.items()):
        truth, predicted = reference.get(annotated, []), predictions.get(source, [])
        if not truth or not predicted:
            continue
        scores = np.array([[iou(t[1:], p[1:]) for p in predicted] for t in truth])
        for i, j in zip(*linear_sum_assignment(-scores)):
            if scores[i, j] < 0.5:
                continue
            key = (predicted[j][0], truth[i][0])
            covered[key] += 1
            frames_of[key].append(annotated)
            boxes_of[(predicted[j][0], truth[i][0], annotated)] = (truth[i][1:], predicted[j][1:])

    # owner: the trajectory a track covers in the most frames, ties to the smaller id
    owner: dict[int, int] = {}
    for (track, trajectory), count in covered.items():
        if track not in owner or (count, -trajectory) > (covered[(track, owner[track])], -owner[track]):
            owner[track] = trajectory
    by_trajectory: dict[int, list[int]] = collections.defaultdict(list)
    for track, trajectory in owner.items():
        by_trajectory[trajectory].append(track)

    duplicates = {g: sorted(ts) for g, ts in by_trajectory.items() if len(ts) > 1}
    if not duplicates:
        raise SystemExit(f"no duplicate identity found in {sequence}")
    D = sum(len(ts) - 1 for ts in by_trajectory.values())

    # Which duplicate to draw is a presentational choice, so the rule is stated
    # rather than left to the eye. Among all of them, take a clean succession
    # --- one identity ending before the next begins, so the bunch was lost and
    # re-acquired rather than briefly double-claimed --- and require both boxes
    # to sit clear of the frame border, since a bunch at the edge of a 4K frame
    # cannot be shown in a neighbourhood. Of those, take the widest gap.
    width, height = 4096, 2160
    candidates = []
    for trajectory, tracks in duplicates.items():
        for early in tracks:
            for late in tracks:
                if early == late:
                    continue
                early_end = max(frames_of[(early, trajectory)])
                late_start = min(frames_of[(late, trajectory)])
                if early_end >= late_start:
                    continue
                early_frame = min(frames_of[(early, trajectory)])
                late_frame = max(frames_of[(late, trajectory)])
                boxes = [boxes_of[(early, trajectory, early_frame)][0],
                         boxes_of[(late, trajectory, late_frame)][0]]
                clearance = min(min(b[0], b[1], width - b[2], height - b[3]) for b in boxes)
                margin = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
                if clearance < margin:
                    continue
                candidates.append((late_frame - early_frame, trajectory, early, late,
                                   early_frame, late_frame))
    if not candidates:
        raise SystemExit(f"no duplicate in {sequence} is both a succession and clear of the border")
    _, trajectory, first, last, first_frame, last_frame = max(candidates)
    tracks = duplicates[trajectory]
    return {
        "sequence": sequence,
        "run": run,
        "annotated_frames": len(annotated_to_source),
        "D_in_sequence": D,
        "trajectories_with_a_duplicate": len(duplicates),
        "trajectory": trajectory,
        "identities": tracks,
        "first": {"track": first, "annotated_frame": first_frame,
                  "source_frame": annotated_to_source[first_frame],
                  "reference_box": boxes_of[(first, trajectory, first_frame)][0],
                  "predicted_box": boxes_of[(first, trajectory, first_frame)][1]},
        "last": {"track": last, "annotated_frame": last_frame,
                 "source_frame": annotated_to_source[last_frame],
                 "reference_box": boxes_of[(last, trajectory, last_frame)][0],
                 "predicted_box": boxes_of[(last, trajectory, last_frame)][1]},
        "gap_annotated_frames": last_frame - first_frame,
        "gap_source_frames": annotated_to_source[last_frame] - annotated_to_source[first_frame],
        "gap_seconds": (annotated_to_source[last_frame] - annotated_to_source[first_frame]) / 59.94,
        "reference_boxes_in_first_frame": len(reference[first_frame]),
    }


def frame_path(sequence: str, annotated: int) -> Path:
    images = ROOT / "datasets/bodegas_grape_bunch_seg/images"
    for split in ("test", "val", "train"):
        candidate = images / split / f"{sequence}_{annotated:06d}.png"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no image for {sequence} frame {annotated}")


def crop_window(box, image_shape, side_px):
    """A fixed-size window centred on the bunch.

    The camera travels along the row, so the same bunch is somewhere else in the
    frame seconds later --- which is the displacement this paper is about. Each
    panel therefore gets its own window, and both windows are the same size so
    the two views are at one scale. Hitting a border shifts the window rather
    than shrinking it, which would silently change that scale.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    height, width = image_shape[:2]
    # wide, not square: the camera travels along the row, so horizontal context is
    # what makes the bunch findable, and a shorter panel costs less page.
    half_w = min(side_px, width) / 2
    half_h = min(side_px / 1.75, height) / 2
    left = int(min(max(cx - half_w, 0), width - 2 * half_w))
    top = int(min(max(cy - half_h, 0), height - 2 * half_h))
    return left, top, int(left + 2 * half_w), int(top + 2 * half_h)


def draw(record: dict, out: Path) -> None:
    apply_paper_style()
    plt.rcParams.update({"axes.grid": False})
    sequence = record["sequence"]
    first, last = record["first"], record["last"]

    full = mpimg.imread(frame_path(sequence, first["annotated_frame"]))
    reference = load_reference(sequence)
    # one window size for both close-ups, from the larger of the two boxes
    side = max(max(b[2] - b[0], b[3] - b[1])
               for b in (first["reference_box"], last["reference_box"])) * 5.5
    windows = {letter: crop_window(side_data["reference_box"], full.shape, side)
               for letter, side_data in (("b", first), ("c", last))}

    # (a) keeps the full width of the released frame but only the vertical band
    # the annotation occupies: above it is sky and the next row over, below it is
    # bare ground, and neither carries a bunch.
    boxes_here = reference[first["annotated_frame"]]
    band_top = min([b[2] for b in boxes_here] + [windows["b"][1]])
    band_bottom = max([b[4] for b in boxes_here] + [windows["b"][3]])
    pad = 0.05 * (band_bottom - band_top)
    band_top = max(0, int(band_top - pad))
    band_bottom = min(full.shape[0], int(band_bottom + pad))

    fig = plt.figure(figsize=(7.16, 1.30), facecolor=PAPER)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.9, 1.0, 1.0), wspace=0.06)
    axes = [fig.add_subplot(grid[i]) for i in range(3)]

    # (a) the whole released frame, every reference box, and the crop window
    axes[0].imshow(full[band_top:band_bottom])
    for _, x0, y0, x1, y1 in reference[first["annotated_frame"]]:
        axes[0].add_patch(Rectangle((x0, y0 - band_top), x1 - x0, y1 - y0, fill=False,
                                    edgecolor=C_GT, lw=0.7))
    left, top, right, bottom = windows["b"]
    axes[0].add_patch(Rectangle((left, top - band_top), right - left, bottom - top,
                                fill=False, edgecolor=CROP, lw=1.1))
    # anchored past the right edge of the window: the top edge is where another
    # reference box sits, and a label there hides annotation the panel is showing
    axes[0].annotate("(b)", xy=(right, top - band_top), xytext=(3, 2),
                     textcoords="offset points", fontsize=5.6, color=PAPER,
                     va="bottom", ha="left",
                     bbox={"facecolor": CROP, "edgecolor": "none", "pad": 1.2})
    axes[0].set_title(
        f"(a) Released frame, full width, {record['reference_boxes_in_first_frame']} "
        "reference bunches",
        loc="left", fontsize=7.2, fontweight="bold", pad=3, color=INK)

    # (b), (c) the same bunch when each identity owned it
    for ax, side, letter in ((axes[1], first, "b"), (axes[2], last, "c")):
        image = mpimg.imread(frame_path(sequence, side["annotated_frame"]))
        wl, wt, wr, wb = windows[letter]
        ax.imshow(image[wt:wb, wl:wr])
        for box, colour, style in ((side["reference_box"], C_GT, "solid"),
                                   (side["predicted_box"], C_PRED, "dashed")):
            x0, y0, x1, y1 = box
            ax.add_patch(Rectangle((x0 - wl, y0 - wt), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=colour, lw=1.2, linestyle=style))
        for y, text, colour in ((0.955, f"reference {record['trajectory']}", C_GT),
                                (0.80, f"tracker: track {side['track']}", C_PRED)):
            ax.text(0.03, y, text, transform=ax.transAxes, fontsize=5.8, color=PAPER,
                    va="top", bbox={"facecolor": colour, "edgecolor": "none", "pad": 1.4})
        gap = "" if letter == "b" else f", {record['gap_seconds']:.1f}\u2009s later"
        ax.set_title(f"({letter}) Frame {side['annotated_frame']}{gap}",
                     loc="left", fontsize=7.2, fontweight="bold", pad=3, color=INK)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE)
            spine.set_linewidth(0.7)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.015)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025,
                    facecolor=PAPER, dpi=400)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    record = find_duplicate(args.sequence, args.run)
    draw(record, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1, default=list))
    print(f"sequence {record['sequence']}: D={record['D_in_sequence']} over "
          f"{record['annotated_frames']} annotated frames, "
          f"{record['trajectories_with_a_duplicate']} trajectories with a duplicate")
    print(f"drawn: trajectory {record['trajectory']} held by identities "
          f"{record['identities']}, frames {record['first']['annotated_frame']} "
          f"and {record['last']['annotated_frame']} "
          f"({record['gap_annotated_frames']} annotated, "
          f"{record['gap_source_frames']} source frames apart)")
    print(f"wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
