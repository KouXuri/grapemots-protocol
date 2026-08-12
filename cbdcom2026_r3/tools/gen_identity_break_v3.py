#!/usr/bin/env python3
"""One bunch, two identities, in one IEEE column.

Layout: the released frame across the top with the crop window marked, and the
two crops below it. The earlier two-row version wasted vertical space on axis
titles and left the panels visually unrelated; here the labels sit inside the
panels, the panels touch, and a leader line ties the crop window to the crops
beneath it.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperstyle import apply, C_GT, C_PRED, C_NEUTRAL  # noqa: E402

apply()
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402

C_WINDOW = "#3a3a3a"      # the crop window on the context frame


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def frame_no(name):
    """The annotated frame number a panel shows, from its file name."""
    import re
    m = re.search(r"(\d+)(?!.*\d)", pathlib.Path(name).stem)
    return m.group(1).lstrip("0") or "0"


def panel_label(ax, text, loc="upper left"):
    x, ha = (0.025, "left") if "left" in loc else (0.975, "right")
    ax.text(x, 0.965, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=7.5, color="white",
            bbox=dict(facecolor=(0, 0, 0, 0.62), edgecolor="none",
                      boxstyle="round,pad=0.22"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counting", type=Path,
                    default=Path("runs/cbdcom2026_queue/results/qual_pp2_boxes.json"))
    ap.add_argument("--images", type=Path,
                    default=Path("datasets/grapemots_det_721/images/all"))
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--pad", type=int, default=300)
    ap.add_argument("--band", type=float, default=0.62,
                    help="fraction of the frame height kept in panel (a)")
    ap.add_argument("--source-fps", type=float, default=29.97)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("figures/fig_identity_break_v3.pdf"))
    args = ap.parse_args()

    e = json.loads(args.counting.read_text())["videos"][0]
    gt_ids, gt_boxes = e["frame_gt_ids"], e["frame_gt_boxes"]
    pr_ids, pr_boxes = e["frame_predicted_ids"], e["frame_predicted_boxes"]
    names = e["frame_names"]

    cover = defaultdict(list)
    for f in range(len(names)):
        for gi, gbox in zip(gt_ids[f], gt_boxes[f]):
            best, best_iou = None, args.match_iou
            for pi, pbox in zip(pr_ids[f], pr_boxes[f]):
                s = iou(gbox, pbox)
                if s > best_iou:
                    best, best_iou = (pi, pbox), s
            if best:
                cover[gi].append((f, best[0], gbox, best[1]))

    best_case = None
    for gi, rows in cover.items():
        first = rows[0][1]
        for row in rows:
            if row[1] != first:
                gap = row[0] - rows[0][0]
                if best_case is None or gap > best_case[0]:
                    best_case = (gap, gi, rows[0], row)
                break
    gap, gt_id, early, late = best_case
    seconds = gap * args.frame_step / args.source_fps
    breaks = sum(1 for r in cover.values() if len({x[1] for x in r}) > 1)
    print(f"  bunch {gt_id}: track {early[1]} -> {late[1]}, {gap} annotated frames "
          f"({gap*args.frame_step} source, {seconds:.1f} s); "
          f"{breaks}/{len(cover)} covered bunches split")

    ctx = cv2.cvtColor(cv2.imread(str(args.images / names[early[0]])), cv2.COLOR_BGR2RGB)
    H, W = ctx.shape[:2]
    # keep the full width, which is what sets the scale, and only the canopy band
    band = int(H * args.band)
    ya = int(min(max(0, (early[2][1] + early[2][3]) / 2 - band / 2), H - band))
    ctx = ctx[ya:ya + band]
    gcx, gcy = (early[2][0] + early[2][2]) / 2, (early[2][1] + early[2][3]) / 2
    x0 = int(max(0, gcx - args.pad)); y0 = int(max(0, gcy - args.pad))
    x1 = int(min(W, gcx + args.pad)); y1 = int(min(H, gcy + args.pad))
    wy0, wy1 = y0 - ya, y1 - ya            # the window, in band coordinates

    col = 3.45
    h_ctx = col * band / W
    h_crop = (col - 0.03) / 2
    fig = plt.figure(figsize=(col, h_ctx + h_crop + 0.02))
    gs = fig.add_gridspec(2, 2, height_ratios=[h_ctx, h_crop],
                          hspace=0.012, wspace=0.012,
                          left=0, right=1, top=1, bottom=0)

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(ctx)
    ax.add_patch(patches.Rectangle((x0, wy0), x1 - x0, wy1 - wy0, fill=False,
                                   edgecolor=C_WINDOW, linewidth=1.1))
    ax.add_patch(patches.Rectangle((x0, wy0), x1 - x0, wy1 - wy0, fill=False,
                                   edgecolor="white", linewidth=0.4,
                                   linestyle=(0, (2, 2))))
    panel_label(ax, f"(a) frame {frame_no(names[early[0]])} of {W}$\\times${H}, "
                    f"{len(gt_ids[early[0]])} annotated bunches; box marks (b), (c)")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    for k, (rec, tag) in enumerate(zip((early, late),
                                       (f"(b) frame {frame_no(names[early[0]])}",
                                        f"(c) frame {frame_no(names[late[0]])}, "
                                        f"{seconds:.1f} s later"))):
        f_i, pid, gbox, pbox = rec
        img = cv2.cvtColor(cv2.imread(str(args.images / names[f_i])), cv2.COLOR_BGR2RGB)
        a = fig.add_subplot(gs[1, k])
        a.imshow(img[y0:y1, x0:x1])
        a.add_patch(patches.Rectangle((gbox[0]-x0, gbox[1]-y0),
                                      gbox[2]-gbox[0], gbox[3]-gbox[1],
                                      fill=False, edgecolor=C_GT, linewidth=1.3))
        a.add_patch(patches.Rectangle((pbox[0]-x0, pbox[1]-y0),
                                      pbox[2]-pbox[0], pbox[3]-pbox[1],
                                      fill=False, edgecolor=C_PRED, linewidth=1.3,
                                      linestyle=(0, (2.6, 1.6))))
        panel_label(a, tag)
        a.text(0.975, 0.965, f"track {pid}", transform=a.transAxes,
               ha="right", va="top", fontsize=7.5, color="white",
               bbox=dict(facecolor=C_PRED, edgecolor="none",
                         boxstyle="round,pad=0.22"))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    fig.savefig(args.out, pad_inches=0.005)
    fig.savefig(args.out.with_suffix(".png"), dpi=400, pad_inches=0.005)
    print(f"  wrote {args.out}  ({col:.2f} x {h_ctx + h_crop + 0.02:.2f} in)")


if __name__ == "__main__":
    main()
