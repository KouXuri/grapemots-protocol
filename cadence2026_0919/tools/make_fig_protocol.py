#!/usr/bin/env python3
"""Figure 2: what an arm changes, and what it does not.

Revised 2026-09-19 after the external review (R-02, R-11) and the code audit
(RESEARCH_AUDIT 10.26). Three facts the old version drew wrongly or not at all:
  * the interval is counted in ANNOTATED frames, and the circling sequences are
    annotated at every second source frame;
  * an interval of k can start at any of k phases;
  * the count's reference G is every trajectory in the annotated row, fixed for
    all arms, while the tracker -- and every identity metric -- sees only the
    processed row.
No data is plotted; the interval drawn is illustrative.
"""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = 33
ANNOT = list(range(0, N, 2))           # annotated: every second source frame
INK, MUTED, ACCENT = "#222222", "#9A9A9A", "#1c5cab"


def main() -> None:
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 7.0, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    rows = [
        ("source frames", list(range(N)), MUTED, "|"),
        ("annotated (reference)", ANNOT, INK, "o"),
        ("processed, interval 1", ANNOT, ACCENT, "o"),
        ("processed, interval 4, phase 0", ANNOT[0::4], ACCENT, "o"),
        ("processed, interval 4, phase 1", ANNOT[1::4], ACCENT, "o"),
    ]
    for k, (label, idx, colour, marker) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([0, N - 1], [y, y], "-", color="#E2E2E2", lw=0.6, zorder=1)
        ax.plot(idx, [y] * len(idx), marker, color=colour,
                ms=2.0 if marker == "|" else 3.1, mew=0.8 if marker == "|" else 0.5,
                mec=colour if marker == "|" else "white", linestyle="none", zorder=3)
        ax.text(-1.2, y, label, ha="right", va="center", fontsize=6.5, color=INK)
    ax.text(N + 0.6, 3.0, "G: every\ntrajectory\nin this row", ha="left", va="center",
            fontsize=5.9, color=INK, linespacing=1.1)
    ax.annotate("", xy=(N + 0.3, 2.35), xytext=(N + 0.3, -0.35),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0))
    ax.text(N + 0.8, 1.0, "tracker and\nidentity metrics\nsee only these", ha="left",
            va="center", fontsize=5.9, color=ACCENT, linespacing=1.1)
    ax.set_xlim(-14.5, N + 7.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    fig.savefig(pathlib.Path(__file__).resolve().parent / "fig_protocol.pdf",
                bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pathlib.Path(__file__).resolve().parent / "fig_protocol.png", dpi=200,
                bbox_inches="tight", pad_inches=0.01)
    print("wrote fig_protocol.pdf")


if __name__ == "__main__":
    main()
