#!/usr/bin/env python3
"""Annotation cadence, not the vineyard, sets the fitted drift rate.

Left: the released sequences split cleanly by how they were labelled, not by how
they were flown. Right: thinning the four every-source-frame sequences on the same
footage walks them up into the cadence-2 band, so the gap is the sampling.
"""
import os
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
RES = ROOT / "results"
PHI = json.loads((ROOT / "tools/paper_numbers.json").read_text())["phi"]["per_video"]
CAD = json.loads((RES / "cadence_control.json").read_text())
CADENCE = {"NoPathPlanning_1": 1, "NoPathPlanning_2": 1, "NoPathPlanning_3": 1,
           "PathPlanning_1": 1, "PathPlanning_2": 2, "PathPlanning_3": 2,
           "PathPlanning_4": 2, "PathPlanning_5": 2, "PathPlanning_6": 2,
           "PathPlanning_7": 2, "PathPlanning_8": 2}
C1, C2, ACC = "#2166AC", "#B2182B", "#4D4D4D"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.5,
                     "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6})
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.35), sharey=True,
                         gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.08})

# ---- left: the released sequences, as published
ax = axes[0]
for i, (v, c) in enumerate(sorted(CADENCE.items(), key=lambda kv: (kv[1], PHI[kv[0]]["source"]))):
    phi = PHI[v]["source"]
    ax.scatter(c + (np.random.RandomState(i).uniform(-0.13, 0.13)), phi, s=18,
               color=C1 if c == 1 else C2, marker="o" if c == 1 else "s",
               edgecolor="white", linewidth=0.4, zorder=3)
lo1 = max(PHI[v]["source"] for v in CADENCE if CADENCE[v] == 1)
hi2 = min(PHI[v]["source"] for v in CADENCE if CADENCE[v] == 2)
ax.axhspan(lo1, hi2, color="#cccccc", alpha=0.35, zorder=0)
ax.annotate(f"{hi2/lo1:.1f}$\\times$ gap,\nnothing between",
            xy=(1.5, (lo1 * hi2) ** 0.5), fontsize=5.6, ha="center", va="center",
            color="#333333")
ax.set_xticks([1, 2]); ax.set_xticklabels(["cad. 1\n4 seq.", "cad. 2\n7 seq."])
ax.set_xlim(0.55, 2.45)
ax.set_ylabel(r"$\varphi$ per source frame")
ax.set_title("as released", fontsize=7)

# ---- right: the same four sequences, thinned
ax = axes[1]
for i, v in enumerate(["NoPathPlanning_1", "NoPathPlanning_2",
                       "NoPathPlanning_3", "PathPlanning_1"]):
    steps = [1, 2, 3]
    ys = [CAD[v][str(s)]["phi_per_source_frame"] for s in steps]
    ax.plot(steps, ys, marker="o", ms=2.8, linewidth=0.9, color=C1, alpha=0.85,
            zorder=3)
    ax.annotate(v.replace("NoPathPlanning_", "NPP").replace("PathPlanning_", "PP"),
                (3, ys[-1]), textcoords="offset points", xytext=(3, -1),
                fontsize=5.6, color="#333333", va="center")
ax.axhspan(hi2, max(PHI[v]["source"] for v in CADENCE if CADENCE[v] == 2),
           color=C2, alpha=0.12, zorder=0)
ax.annotate("released cadence-2", xy=(1.05, hi2 * 1.5), fontsize=5.6,
            color=C2, va="bottom")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["1", "2", "3"])
ax.set_xlim(0.85, 3.85)
ax.set_xlabel("keep every $n$-th annotated frame", fontsize=6.2)
ax.set_title("same footage, thinned", fontsize=7)

for ax in axes:
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", linewidth=0.4, alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
out = ROOT / "figures/fig_cadence.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
print("saved", out)
