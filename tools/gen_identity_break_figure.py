#!/usr/bin/env python3
"""One bunch, two identities: the over-count in a single picture.

Finds a ground-truth bunch that the tracker covered with one identity early in
the sequence and a different identity later, then renders the two frames side by
side. That single vine is counted twice by a unique-track count, and this is the
event the whole paper is about; a figure showing "the detector found the bunches"
does not show it.

Selection is automatic and reported, so the frame pair is not hand-picked to
flatter the argument: we take the GT track with the longest gap between its two
covering identities, which is simply the clearest instance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperstyle import apply, C_GT, C_PRED, C_NEUTRAL  # noqa: E402

apply()
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counting", type=Path,
                    default=Path("results/qual_pp2_boxes.json"))
    ap.add_argument("--images", type=Path,
                    default=Path("datasets/grapemots_det_721/images/test"))
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--pad", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("figures/fig_identity_break.pdf"))
    args = ap.parse_args()

    entry = json.loads(args.counting.read_text())["videos"][0]
    gt_ids = entry["frame_gt_ids"]; gt_boxes = entry["frame_gt_boxes"]
    pr_ids = entry["frame_predicted_ids"]; pr_boxes = entry["frame_predicted_boxes"]
    names = entry["frame_names"]

    # for each GT identity, which predicted identity covered it in which frame
    cover: dict[int, list[tuple[int, int, list[float], list[float]]]] = defaultdict(list)
    for f in range(len(names)):
        for gi, gbox in zip(gt_ids[f], gt_boxes[f]):
            best, best_iou = None, args.match_iou
            for pi, pbox in zip(pr_ids[f], pr_boxes[f]):
                score = iou(gbox, pbox)
                if score > best_iou:
                    best, best_iou = (pi, pbox), score
            if best:
                cover[gi].append((f, best[0], gbox, best[1]))

    # the clearest identity break: same bunch, two predicted ids, largest gap
    best_case = None
    for gi, rows in cover.items():
        first_id = rows[0][1]
        for row in rows:
            if row[1] != first_id:
                gap = row[0] - rows[0][0]
                if best_case is None or gap > best_case[0]:
                    best_case = (gap, gi, rows[0], row)
                break
    if best_case is None:
        raise SystemExit("no identity break found at this IoU threshold")
    gap, gt_id, early, late = best_case
    print(f"  GT bunch {gt_id}: identity {early[1]} at frame {early[0]} "
          f"-> identity {late[1]} at frame {late[0]} (gap {gap} frames)")

    fig = plt.figure(figsize=(7.0, 2.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.75, 1, 1], wspace=0.05)

    # (a) the whole released frame, so the reader sees the vineyard and the scale
    context = cv2.cvtColor(cv2.imread(str(args.images / names[early[0]])),
                           cv2.COLOR_BGR2RGB)
    ax0 = fig.add_subplot(grid[0])
    ax0.imshow(context)
    gcx, gcy = (early[2][0] + early[2][2]) / 2, (early[2][1] + early[2][3]) / 2
    ax0.add_patch(patches.Rectangle((gcx - args.pad, gcy - args.pad),
                                    2 * args.pad, 2 * args.pad, fill=False,
                                    edgecolor="#f0c000", linewidth=0.9))
    # the bunch itself, at true scale inside that region: without it the yellow
    # rectangle reads as the target rather than as the crop window.
    gx0, gy0, gx1, gy1 = early[2]
    ax0.add_patch(patches.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                                    edgecolor=C_GT, linewidth=0.4))
    ax0.set_title(f"(a) released frame, {context.shape[1]}$\\times${context.shape[0]}, "
                  f"{len(gt_ids[early[0]])} annotated bunches", fontsize=8.5)
    ax0.set_xticks([]); ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_edgecolor(C_NEUTRAL); spine.set_linewidth(0.6)

    axes = [fig.add_subplot(grid[1]), fig.add_subplot(grid[2])]
    for ax, (frame_index, pred_id, gbox, pbox), tag in zip(
            axes, (early, late),
            ("(b) first appearance", f"(c) {gap} frames later")):
        image = cv2.cvtColor(cv2.imread(str(args.images / names[frame_index])),
                             cv2.COLOR_BGR2RGB)
        cx, cy = (gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2
        x0 = int(max(0, cx - args.pad)); y0 = int(max(0, cy - args.pad))
        x1 = int(min(image.shape[1], cx + args.pad)); y1 = int(min(image.shape[0], cy + args.pad))
        ax.imshow(image[y0:y1, x0:x1])
        ax.add_patch(patches.Rectangle((gbox[0] - x0, gbox[1] - y0),
                                       gbox[2] - gbox[0], gbox[3] - gbox[1],
                                       fill=False, edgecolor=C_GT, linewidth=1.3))
        ax.add_patch(patches.Rectangle((pbox[0] - x0, pbox[1] - y0),
                                       pbox[2] - pbox[0], pbox[3] - pbox[1],
                                       fill=False, edgecolor=C_PRED, linewidth=1.3,
                                       linestyle=(0, (3, 2))))
        ax.text(0.5, 0.965, f"track {pred_id}", transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="white",
                bbox=dict(facecolor=C_PRED, edgecolor="none", pad=1.4))
        ax.set_title(tag, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(C_NEUTRAL); spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.3)
    fig.savefig(args.out); fig.savefig(args.out.with_suffix(".png"))
    breaks = sum(1 for rows in cover.values() if len({r[1] for r in rows}) > 1)
    print(f"  {breaks} of {len(cover)} covered GT bunches received more than one "
          f"predicted identity")


if __name__ == "__main__":
    main()
