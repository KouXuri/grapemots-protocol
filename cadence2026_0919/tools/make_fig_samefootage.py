#!/usr/bin/env python3
"""Figure 1: one sequence, five scales, four intervals -- the count changes sign.

Data: figures/data/lovo_surface_terms.json, exported from
runs/lovo_surface_0918/results (each sequence read by a leave-one-video-out
checkpoint that never trained on it) and decomposed by
tools/decompose_count_error.py at tau = 1. The denominator is the number of
reference trajectories over all annotated frames, identical in every cell.

Only measured points are drawn; nothing is interpolated. sigma is ordered, so it
is one blue hue light to dark (ramp validated with the dataviz ordinal check),
and every line is labelled directly at its right end.
"""
from __future__ import annotations
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
SEQ = "PathPlanning_4"
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")


def main() -> None:
    data = json.load(open(HERE.parent / "results" / "lovo_surface_terms.json"))
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 8.0, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ax.axhline(0, color="#222222", lw=0.7, zorder=1)
    ends = []
    for s, c in zip(SIGMAS, RAMP):
        e = [data["cells"][f"{SEQ}|{s}|{d}"]["e"] for d in DELTAS]
        ax.plot(DELTAS, e, "-", color=c, lw=1.6, zorder=3)
        ax.plot(DELTAS, e, "o", color=c, ms=4.2, mec="white", mew=0.7, zorder=4)
        ends.append((e[-1], s, c))
    # direct labels, nudged apart where the end values crowd
    ends.sort()
    placed = []
    for y, s, c in ends:
        y_lab = y if not placed else max(y, placed[-1] + 0.095)
        placed.append(y_lab)
        ax.text(8 * 1.09, y_lab, f"{s}", va="center", ha="left", fontsize=8.0, color="#222222")
    ax.text(8 * 1.09, max(placed) + 0.15, "scale", va="center", ha="left", fontsize=8.0,
            color="#555555", style="italic")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DELTAS, [str(d) for d in DELTAS])
    ax.minorticks_off()
    ax.set_xlim(0.85, 8 * 1.55)
    ax.set_xlabel("Processing interval (annotated frames per processed frame)")
    ax.set_ylabel("Count error, (P $-$ G) / G")
    ax.text(0.95, -0.42, "under-count", fontsize=8.0, color="#555555", va="center")
    ax.text(2.9, 0.66, "over-count", fontsize=8.0, color="#555555", va="center")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", lw=0.4, zorder=0)
    fig.savefig(HERE / "fig_samefootage.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / "fig_samefootage.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    print("wrote fig_samefootage.pdf")


if __name__ == "__main__":
    main()
