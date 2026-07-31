#!/usr/bin/env python3
"""The two figures carrying the paper's main results.

fig_miss_modes
    Four ways to miss a bunch, and the fact that they do not point the same way.
    Dropping observations frame by frame -- independently, or in occlusion-like
    runs -- leaves the object visible elsewhere in the window, so the tracker
    restarts it under a new identity and the count goes UP. Dropping identities
    outright removes them from the count, so it goes DOWN. A weak detector does
    the second, which is why a reported error near zero is not accuracy.

fig_metric_alignment
    Detection recall at the operating point that produced the count, against the
    count error. Eight configurations of one pipeline, every one of them measured
    after the same tile merge, so nothing here compares two pipelines. If better
    components gave better counts this would slope down. It slopes up.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("runs/cbdcom2026_queue/results")
OUT = Path("figures")
COVERAGE = 0.8

# Two families, coloured so the split is visible before the caption is read.
MODES = [
    ("bernoulli", "i.i.d. per-frame miss", "#B2182B", "o", "-"),
    ("block", "occlusion-like block miss", "#EF8A62", "s", "-"),
    ("identity", "identity removed", "#2166AC", "^", "--"),
    ("size", "smallest identities removed", "#67A9CF", "D", "--"),
]

ARMS = [
    ("arm_resize", "Resize", "#4D4D4D", "s"),
    ("count_conf0.55", "conf 0.55", "#2166AC", "v"),
    ("arm_bytetrack_tiled", "ByteTrack", "#67A9CF", "^"),
    ("count_conf0.40", "conf 0.40", "#2166AC", "v"),
    ("arm_yolo11s_tiled", "YOLO11s", "#7B3294", "D"),
    ("arm_ios_tiled", "IoS merge", "#1B7837", "P"),
    ("arm_botsort_tiled", "BoT-SORT", "#B2182B", "o"),
    ("arm_reid", "+ ReID", "#B2182B", "*"),
]


LONG = {"Resize": "Resize + BoT-SORT", "conf 0.55": "Tiles, conf 0.55",
        "ByteTrack": "Tiles + ByteTrack", "conf 0.40": "Tiles, conf 0.40",
        "YOLO11s": "Tiles + YOLO11s", "IoS merge": "Tiles + IoS merge",
        "BoT-SORT": "Tiles + BoT-SORT", "+ ReID": "Tiles + ReID"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.linewidth": 0.6, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "legend.frameon": False,
    })


def retained(run, coverage=COVERAGE):
    """Cells outside the degenerate and trivial regions; see Sec. II-D."""
    cells = [c for c in run["count_error_surface"]
             if c["min_track_len"] <= c["window_frames"] / 2 and c["gt_tracks"]]
    if not cells:
        return []
    full = max(c["gt_tracks"] for c in cells)
    return [c for c in cells if c["gt_tracks"] / full >= coverage]


def miss_mode_series(mode: str):
    """{p: (median over videos, 25th, 75th)} under the frozen definition."""
    data = json.loads((RES / f"oracle_master_{mode}.json").read_text())
    per_video = defaultdict(lambda: defaultdict(list))
    for run in data["runs"]:
        for cell in retained(run):
            e = (cell["predicted_tracks"] - cell["gt_tracks"]) / cell["gt_tracks"]
            per_video[run["miss_rate"]][run["video"]].append(e)
    out = {}
    for miss, videos in sorted(per_video.items()):
        medians = [statistics.median(v) for v in videos.values()]
        out[miss] = (statistics.median(medians),
                     float(np.percentile(medians, 25)),
                     float(np.percentile(medians, 75)))
    return out


def figure_miss_modes() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for mode, label, colour, marker, dash in MODES:
        series = miss_mode_series(mode)
        xs = sorted(series)
        med = [series[x][0] for x in xs]
        lo = [series[x][1] for x in xs]
        hi = [series[x][2] for x in xs]
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, linewidth=0)
        ax.plot(xs, med, dash, color=colour, marker=marker, markersize=3.4,
                linewidth=1.2, label=label)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("miss rate $p$")
    ax.set_ylabel("counting error $e=(P-G)/G$")
    ax.set_xlim(-0.01, 0.44)
    ax.set_ylim(-0.62, 2.55)
    ax.annotate("interrupting an identity\nadds identities",
                xy=(0.435, 2.00), fontsize=6.5, color="#B2182B", ha="right")
    ax.annotate("removing an identity\ntakes it out of the count",
                xy=(0.135, -0.52), fontsize=6.5, color="#2166AC", ha="center")
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_miss_modes.{suffix}", dpi=300, bbox_inches="tight")
    print("saved fig_miss_modes")


HOTA = json.loads((RES / "hota_arms.json").read_text())


def arm_point(stem: str, label: str = None):
    """(recall, AssA, pooled whole-sequence error) for one configuration."""
    d = json.loads((RES / f"{stem}.json").read_text())
    pred = truth = 0
    for video in d["videos"]:
        cells = [c for c in video["count_error_surface"] if c["min_track_len"] == 1]
        last = max(cells, key=lambda c: c["window_frames"])
        pred += last["prefix_predicted_tracks"]
        truth += last["prefix_gt_tracks"]
    return (d["overall"]["recall"], HOTA[label]["AssA"], (pred - truth) / truth)


def spearman(xs, ys) -> float:
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        r = [0.0] * len(values)
        for position, index in enumerate(order):
            r[index] = position + 1.0
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def figure_metric_alignment() -> None:
    points = [(label, colour, marker) + arm_point(stem, LONG[label])
              for stem, label, colour, marker in ARMS]
    recalls = [p[3] for p in points]
    assas = [p[4] for p in points]
    errors = [p[5] for p in points]
    rho_recall = spearman(recalls, errors)
    rho_assa = spearman(assas, errors)
    print(f"  Spearman rho: recall vs count error {rho_recall:+.3f}, "
          f"AssA vs count error {rho_assa:+.3f}")

    fig, axes = plt.subplots(1, 1, figsize=(3.42, 2.55), squeeze=False)
    for ax, values, name, rho in ((axes[0][0], assas, "AssA", rho_assa),):
        # labels are placed away from the axis edge so the rightmost point,
        # which is the one the argument turns on, is not clipped
        span_x = max(values) - min(values)
        placed = []
        for (label, colour, marker, _, _, err), value in zip(points, values):
            right_edge = value > min(values) + 0.72 * span_x
            offset = (-5, 4) if right_edge else (4, 4)
            # two arms can sit almost on top of each other on one axis; drop the
            # second label below its marker instead of printing them overlapped
            if any(abs(value - px) < 0.04 * span_x and abs(err - pe) < 0.18
                   for px, pe in placed):
                offset = (offset[0], -9)
            placed.append((value, err))
            ax.scatter(value, err, s=34, color=colour, marker=marker,
                       edgecolor="white", linewidth=0.4, zorder=3)
            ax.annotate(label, (value, err), textcoords="offset points",
                        xytext=offset, fontsize=6.2, color="#333333",
                        ha="right" if right_edge else "left")
        # No trend line. With eight non-independent points a least-squares slope
        # can run opposite to the rank correlation in the title, which would say
        # more than the data supports.
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        pad = 0.10 * (max(values) - min(values))
        ax.set_xlim(min(values) - pad, max(values) + pad)
        ax.set_ylim(-0.25, 2.95)
        ax.set_xlabel("HOTA association score AssA")
        ax.set_title(rf"Spearman $\rho={rho:+.2f}$, exact $p=0.30$", fontsize=7)
    axes[0][0].set_ylabel("whole-sequence count error")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_metric_alignment.{suffix}", dpi=300, bbox_inches="tight")
    print("saved fig_metric_alignment")
    return rho_recall, rho_assa


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    style()
    figure_miss_modes()
    figure_metric_alignment()
