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

# Okabe-Ito, the colour-blind-safe set most venues' figures use. Checked with the
# dataviz skill's validator maths (Machado 2009 CVD, OKLab dE): every pair a figure
# puts side by side separates at dE >= 7.6 under protan/deutan, and the one pair
# below 8 (bluish green / reddish purple) always carries different markers too.
# The 2021 flights (earlier release) are vermillion and GrapeMOTS blue in every figure.
C_GT = "#0072B2"       # blue: the 2024 release, reference-side quantities
C_PRED = "#D55E00"     # vermillion: the 2023 release, the processing-cadence path
C_ERR = C_PRED         # the cadence path runs on the 2023 release
C_ALT = "#009E73"      # bluish green: third series
C_MAGENTA = "#CC79A7"  # reddish purple: fourth series; D in the decomposition
C_ORANGE = "#E69F00"   # orange: U in the decomposition (needs a direct label)
C_COOL = "#56B4E9"     # sky blue: M in the decomposition (needs a direct label)
C_NEUTRAL = "#4d4d4d"  # axes, zero lines, annotation text


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
