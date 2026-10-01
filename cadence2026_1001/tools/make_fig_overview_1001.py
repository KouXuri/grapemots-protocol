#!/usr/bin/env python3
"""Fig. 1 of the 2026-10-01 revision: the protocol and its result in one column.

Top: what each arm of the cadence intervention sees. Source frames run at 59.94 fps;
the release labels one in 36; the sparse arm tracks only the labelled frames, the
source-rate arm tracks every frame, and both are scored at the labelled frames
against the same reference. The top is drawn to the median interval: 36 source
frames between labels, every one ticked.
Bottom: the pooled decomposition P - G = U + D - M of the two arms on the 17
model-unseen sequences, read from runs/decomp_0812/results/cadence_decomposition.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from paperstyle import apply, C_COOL, C_MAGENTA, C_NEUTRAL, C_ORANGE  # noqa: E402

apply()
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["hatch.linewidth"] = 0.7
from matplotlib.patches import Patch  # noqa: E402

DEC = json.loads((ROOT / "runs/decomp_0812/results/cadence_decomposition.json").read_text())["decomposition"]
ARMS = [("sparse arm", DEC["rel_buf30"]), ("source-rate arm", DEC["src_buf30"])]
OUT = ROOT / "cadence_1001" / "figures"

fig, (top, bot) = plt.subplots(2, 1, figsize=(3.45, 2.95),
                               gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.62})
FS = 8                                   # IEEE template: 8 pt figure labels

# ---- top: the three cadences, drawn to the median interval ---------------------
TICKS, SPAN = 36, 3                       # every source frame of three median intervals
xs = range(TICKS * SPAN + 1)
labelled = [i * TICKS for i in range(SPAN + 1)]
rows = {"source frames": 3, "released labels": 2, "sparse arm": 1, "source-rate arm": 0}
for x in labelled:
    top.axvline(x, color=C_NEUTRAL, lw=0.6, ls=(0, (2, 2)), zorder=1)
top.vlines(list(xs), rows["source frames"] - 0.22, rows["source frames"] + 0.22,
           color="#8a8a8a", lw=0.35)
top.scatter(labelled, [rows["released labels"]] * len(labelled), marker="D", s=18,
            facecolors="white", edgecolors="black", linewidths=0.8, zorder=3)
top.scatter(labelled, [rows["sparse arm"]] * len(labelled), marker="s", s=14,
            color=C_NEUTRAL, zorder=3)
top.vlines(list(xs), rows["source-rate arm"] - 0.17, rows["source-rate arm"] + 0.17,
           color=C_NEUTRAL, lw=0.55, zorder=3)
top.set_yticks(list(rows.values()))
top.set_yticklabels(list(rows.keys()), fontsize=FS)
top.set_ylim(-0.45, 3.6)
top.set_xlim(-2.5, TICKS * SPAN + 2.5)
top.set_xticks([])
top.grid(False)
for side in ("top", "right", "bottom", "left"):
    top.spines[side].set_visible(False)
top.tick_params(axis="y", length=0, pad=2)
top.annotate("", xy=(TICKS, 3.5), xytext=(0, 3.5),
             arrowprops=dict(arrowstyle="<->", lw=0.6, color=C_NEUTRAL, shrinkA=0, shrinkB=0))
top.text(TICKS / 2, 3.62, "median 36 frames, 0.6 s", fontsize=FS, color=C_NEUTRAL,
         ha="center", va="bottom", transform=top.transData)

# ---- bottom: the decomposition of each arm -----------------------------------
for y, (name, t) in zip((1, 0), ARMS):
    # colour plus texture: U solid, D hatched, M dotted, so the terms stay apart for
    # colour-vision deficiency and in greyscale print (U and M have equal luminance)
    bot.barh(y, t["U"], left=0, height=0.52, color=C_ORANGE, edgecolor="white", linewidth=0.6)
    bot.barh(y, t["D"], left=t["U"], height=0.52, color=C_MAGENTA, edgecolor="white", linewidth=0.6,
             hatch="////")
    bot.barh(y, -t["M"], left=0, height=0.52, color=C_COOL, edgecolor="white", linewidth=0.6,
             hatch="....")
    net = t["P"] - t["G"]
    bot.plot([net], [y], marker="D", color="black", markersize=4.6, zorder=4)
    bot.text(t["U"] + t["D"] + 15, y,
             f"$e={t['signed_error']:+.3f}$\ncoverage {t['assigned_fraction']:.3f}",
             fontsize=FS, va="center", ha="left", color="black", linespacing=1.05)
bot.axvline(0, color="black", lw=0.7, zorder=3)
bot.set_yticks([1, 0])
bot.set_yticklabels([a for a, _ in ARMS], fontsize=FS)
bot.set_ylim(-0.55, 1.55)
bot.set_xlim(-260, 760)
bot.set_xticks([-200, 0, 200, 400])
bot.set_xlabel("count relative to the reference, $P-G=U+D-M$", fontsize=FS, labelpad=1.5)
bot.tick_params(axis="x", labelsize=FS)
bot.tick_params(axis="y", length=0, pad=2)
bot.grid(axis="y", visible=False)
for side in ("top", "right", "left"):
    bot.spines[side].set_visible(False)
bot.legend(handles=[Patch(facecolor=C_ORANGE, edgecolor="white", label="$U$ ownerless"),
                    Patch(facecolor=C_MAGENTA, edgecolor="white", hatch="////", label="$D$ duplicate"),
                    Patch(facecolor=C_COOL, edgecolor="white", hatch="....", label="$M$ unassigned"),
                    plt.Line2D([], [], marker="D", color="black", lw=0, markersize=4.6, label="$P-G$")],
           frameon=False, fontsize=FS, ncol=2, loc="lower center", bbox_to_anchor=(0.42, 1.0),
           handlelength=1.0, handletextpad=0.35, columnspacing=1.0, borderaxespad=0.1,
           labelspacing=0.2)

fig.align_ylabels([top, bot])
fig.savefig(OUT / "fig_overview.pdf")
fig.savefig(OUT / "fig_overview.png", dpi=400)
print("wrote", OUT / "fig_overview.pdf")
for name, t in ARMS:
    print(name, {k: t[k] for k in ("P", "G", "U", "D", "M")}, round(t["signed_error"], 3), round(t["assigned_fraction"], 3))
