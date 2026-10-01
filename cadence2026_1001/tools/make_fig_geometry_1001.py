#!/usr/bin/env python3
"""Fig. 3 of the camera-ready (1 October 2026, after the paragraph audit).

One point per sequence: the median IoU of consecutive labelled boxes of the same
trajectory against their median r, at each release's own labelling step, beside
curve (2), the IoU of two equal squares shifted along one axis. It replaces the
two-panel figure whose lower panel interpolated a zero crossing between the two
2021 arms, 35 times apart in r; that number carried no information and is gone.
Data: runs/grapemots_journal_0805/results/sequence_structure.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from paperstyle import apply, C_GT, C_PRED, C_NEUTRAL  # noqa: E402

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "cadence_1001" / "figures"
STYLE = {"grapemots": (C_GT, "s", "GrapeMOTS"), "bodegas2023": (C_PRED, "o", "2021 flights")}
structure = [row for row in json.loads(
    (ROOT / "runs/grapemots_journal_0805/results/sequence_structure.json").read_text())["sequences"]
    if row.get("corpus") in STYLE]

fig, ax = plt.subplots(figsize=(3.45, 1.8))
grid = np.linspace(1e-3, 1.0, 400)
ax.plot(grid, (1 - grid) / (1 + grid), color="black", linewidth=1.0, zorder=2)
ax.plot([1.0, 30.0], [0.0, 0.0], color="black", linewidth=1.0, zorder=2)
ax.text(0.47, 0.40, "$(1-r)/(1+r)$", fontsize=8, color="black", ha="left")
ax.axvline(2 ** 0.5, color=C_NEUTRAL, linewidth=0.5, linestyle=":", zorder=1)
ax.text(1.36, 0.93, "$r=\\sqrt{2}$", fontsize=8, color=C_NEUTRAL, ha="right")
for key, (colour, marker, label) in STYLE.items():
    rows = [s for s in structure if s.get("corpus") == key]
    ax.scatter([s["step_over_size_median"] for s in rows], [s["consecutive_iou_median"] for s in rows],
               s=13, c=colour, marker=marker, edgecolors="white", linewidths=0.35,
               label=f"{label} ({len(rows)} sequences)", zorder=3)
    print(label, len(rows), "r range", round(min(s["step_over_size_median"] for s in rows), 3),
          round(max(s["step_over_size_median"] for s in rows), 3), "IoU range",
          round(min(s["consecutive_iou_median"] for s in rows), 3), round(max(s["consecutive_iou_median"] for s in rows), 3))
ax.set_xscale("log")
ax.set_xlim(0.07, 22)
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
ax.tick_params(axis="x", which="minor", bottom=False)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("displacement in units of target size, $r$", fontsize=8)
ax.set_ylabel("overlap of consecutive\nlabeled boxes, IoU", fontsize=8)
ax.tick_params(labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.6), handletextpad=0.3,
          borderpad=0.15, labelspacing=0.2)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.savefig(OUT / "fig_geometry.pdf")
print("wrote", OUT / "fig_geometry.pdf")
