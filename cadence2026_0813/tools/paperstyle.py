"""Shared figure style for the CBDCom paper.

The important line is `pdf.fonttype: 42`. Matplotlib defaults to Type 3 fonts,
which IEEE PDF eXpress rejects; 42 embeds TrueType instead. Everything else is
sizing chosen for a 3.45 in IEEE column so that no figure text ends up smaller
than the 8 pt body text once placed.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ColorBrewer RdBu, colour-blind safe and legible in greyscale print
C_GT = "#2166ac"       # ground truth / reference
C_PRED = "#d95f02"     # prediction / error (orange; no red, and safe in greyscale)
C_ALT = "#ef8a62"      # secondary series
C_COOL = "#67a9cf"     # tertiary series
C_ERR = "#b2182b"      # signed count error; the other end of the same RdBu ramp
C_NEUTRAL = "#4d4d4d"  # axes, zero lines, annotation


def apply() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,          # IEEE: no Type 3 fonts
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": C_NEUTRAL,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
