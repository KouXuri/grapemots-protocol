#!/usr/bin/env python3
"""Figure 3: where each of ten out-of-fold sequences changes sign, at each scale.

Each cell is the pair of measured intervals between which the count error goes
from positive to negative ('<1': already negative at Delta = 1; '>8': still
positive at Delta = 8). Brackets only -- no interpolated crossing is drawn.
Shading is ordinal, one blue hue from early to late (validated ordinal ramp);
the bracket is printed in every cell, so the figure does not rely on colour.
Data: figures/data/lovo_surface_terms.json.
"""
from __future__ import annotations
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).resolve().parent
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
ROWS = [(f"PathPlanning_{i}", f"C{i}") for i in range(2, 9)] + \
       [(f"NoPathPlanning_{i}", f"F{i}") for i in range(1, 4)]
CATS = ["<1", "1–2", "2–4", "4–8", ">8"]
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
INK = ("#111111", "#111111", "#ffffff", "#ffffff", "#ffffff")


def bracket(es):
    if es[0] < 0:
        return 0
    for i in range(3):
        if es[i] >= 0 > es[i + 1]:
            return i + 1
    return 4


def main() -> None:
    data = json.load(open(HERE.parent / "results" / "lovo_surface_terms.json"))
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 7.0, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    gap = 0.35  # visual break between circling and frontal passes
    for r, (seq, lab) in enumerate(ROWS):
        y = -(r + (gap if r >= 7 else 0))
        for c, s in enumerate(SIGMAS):
            es = [data["cells"][f"{seq}|{s}|{d}"]["e"] for d in DELTAS]
            k = bracket(es)
            ax.add_patch(Rectangle((c + 0.04, y - 0.46), 0.92, 0.92, color=RAMP[k], lw=0))
            ax.text(c + 0.5, y, CATS[k], ha="center", va="center", fontsize=6.4, color=INK[k])
        ax.text(-0.12, y, lab, ha="right", va="center", fontsize=6.6)
    for c, s in enumerate(SIGMAS):
        ax.text(c + 0.5, 0.72, str(s), ha="center", va="bottom", fontsize=6.6)
    ax.text(2.5, 1.28, "Inference scale (pixels, long side)", ha="center", va="bottom", fontsize=6.8)
    ax.text(-1.05, -3.0, "circling", rotation=90, ha="center", va="center", fontsize=6.6, color="#555555")
    ax.text(-1.05, -8.35 - gap + 0.35, "frontal", rotation=90, ha="center", va="center", fontsize=6.6, color="#555555")
    ax.set_xlim(-1.3, 5.05)
    ax.set_ylim(-(len(ROWS) - 1 + gap) - 0.6, 1.75)
    ax.axis("off")
    fig.savefig(HERE / "fig_crossmap.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / "fig_crossmap.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    print("wrote fig_crossmap.pdf")


if __name__ == "__main__":
    main()
