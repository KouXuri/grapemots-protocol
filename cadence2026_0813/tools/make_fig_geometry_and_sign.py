#!/usr/bin/env python3
"""Geometry and consequence on one axis.

Fig. 3 and the sign figure were separate and shared an abscissa, which cost a
caption and an inch of column for no gain in argument. Stacked, they read as one
statement: above, what happens to the overlap between consecutive reference boxes
as r grows; below, what happens to the count that overlap has to support. The
theta bands are drawn once, through both.

Top panel, one point per sequence for all 51, each a per-sequence median, against
the equal-square single-axis reference curve. Bottom panel, four pipelines
observed over a range of processing cadences; two are fed annotated boxes, so
their ownerless-track term is empty, and two are fed a detector's output, which
adds a standing surplus of tracks and lifts the whole curve.
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

OUT = ROOT / "figures" / "journal_0809"
EXT = ROOT / "runs/ext_cadence_0813/results"

STYLE = {"mot17": ("#7f7f7f", "^", "MOT17"),
         "mot20": ("#bcbd22", "v", "MOT20"),
         "grapemots": (C_GT, "s", "vineyard 2024"),
         "bodegas2023": (C_PRED, "o", "vineyard 2023")}

ladder = json.loads((ROOT / "runs/final_analyses_0809/results/density_realpipeline.json")
                    .read_text())["pooled_tau1"]
r_unit = json.loads((EXT / "geometry_grapemots.json").read_text())["by_step"]["2"][
    "sequence_median_r"]
ladder_k = [1, 2, 4, 8, 16, 32]
grape_r = [r_unit * k for k in ladder_k]
grape_e = [ladder[f"k{k}"]["signed_error"] for k in ladder_k]
grape_base = (ladder["k1"]["U"] + ladder["k1"]["D"]) / ladder["k1"]["G"]


def mot(corpus: str, tracker: str = "bytetrack"):
    cadence = json.loads((EXT / f"cadence_{corpus}.json").read_text())["pooled"]
    geometry = json.loads((EXT / f"geometry_{corpus}.json").read_text())["by_step"]
    steps = sorted(int(key) for key in geometry if geometry[key]["r_earlier_box"]["pairs"])
    r = [geometry[str(k)]["sequence_median_r"] for k in steps]
    released = [cadence[f"k={k}|released|{tracker}"]["1"]["signed_error"] for k in steps]
    one = cadence[f"k=1|released|{tracker}"]["1"]
    return r, released, (one["U"] + one["D"]) / one["G"]


def crossing(r_values, e_values):
    for (r0, e0), (r1, e1) in zip(zip(r_values, e_values), zip(r_values[1:], e_values[1:])):
        if e0 > 0 >= e1:
            t = e0 / (e0 - e1)
            return float(np.exp(np.log(r0) + t * (np.log(r1) - np.log(r0))))
    return None


mot17_r, mot17_e, mot17_base = mot("mot17")
mot20_r, mot20_e, mot20_base = mot("mot20")

decomposition = json.loads(
    (ROOT / "runs/decomp_0812/results/cadence_decomposition.json").read_text()
)["decomposition"]
bodegas_r_released = json.loads((EXT / "geometry_bodegas2023.json").read_text())[
    "by_step"]["1"]["sequence_median_r"]
bodegas_r = [bodegas_r_released / 36.0, bodegas_r_released]
bodegas_e = [decomposition["src_buf30"]["signed_error"],
             decomposition["rel_buf30"]["signed_error"]]
bodegas_base = ((decomposition["src_buf30"]["U"] + decomposition["src_buf30"]["D"])
                / decomposition["src_buf30"]["G"])

structure = json.loads((ROOT / "runs/grapemots_journal_0805/results/sequence_structure.json")
                       .read_text())["sequences"]

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.45, 3.25), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.10})

for panel in (ax, bx):
    panel.axvspan(0.20, 0.40, color="#dddddd", alpha=0.6, linewidth=0, zorder=0)

grid = np.linspace(1e-3, 1.0, 400)
ax.plot(grid, (1 - grid) / (1 + grid), color="black", linewidth=1.0, zorder=2)
ax.plot([1.0, 30.0], [0.0, 0.0], color="black", linewidth=1.0, zorder=2)
ax.text(0.55, 0.30, "$(1-r)/(1+r)$", fontsize=6.3, color="black", ha="left")
ax.axvline(2 ** 0.5, color=C_NEUTRAL, linewidth=0.5, linestyle=":", zorder=1)
ax.text(1.52, 0.86, "$\\sqrt{2}$", fontsize=6.3, color=C_NEUTRAL, ha="left")
for key, (colour, marker, label) in STYLE.items():
    rows = [s for s in structure if s.get("corpus") == key]
    ax.scatter([s["step_over_size_median"] for s in rows],
               [s["consecutive_iou_median"] for s in rows],
               s=13, c=colour, marker=marker, edgecolors="white",
               linewidths=0.35, label=f"{label} ({len(rows)})", zorder=3)
ax.set_ylabel("consecutive\nreference IoU")
ax.set_ylim(-0.05, 1.08)
ax.legend(frameon=False, fontsize=6.0, loc="center left", handletextpad=0.3,
          borderpad=0.15, labelspacing=0.2)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

bx.axhline(0, color=C_NEUTRAL, linewidth=0.6, zorder=1)
series = [
    (grape_r, grape_e, C_GT, "s", f"vineyard 2024, {grape_base:.1f}"),
    (bodegas_r, bodegas_e, C_PRED, "o", f"vineyard 2023, {bodegas_base:.1f}"),
    (mot17_r, mot17_e, "#7f7f7f", "^", f"MOT17, {mot17_base:.2f}"),
    (mot20_r, mot20_e, "#bcbd22", "v", f"MOT20, {mot20_base:.2f}"),
]
for x, y, colour, marker, label in series:
    bx.plot(x, y, "--" if len(x) == 2 else "-", marker=marker, color=colour,
            markersize=3.2, markeredgecolor="white", markeredgewidth=0.35,
            label=label, zorder=3)
    crossed = crossing(x, y)
    if crossed:
        bx.plot([crossed], [0.0], marker="|", color=colour, markersize=6.5,
                markeredgewidth=1.3, zorder=4)
bx.set_ylabel("signed count error $e$")
bx.set_yscale("symlog", linthresh=0.5, linscale=0.9)
bx.set_ylim(-0.9, 4.0)
bx.set_yticks([-0.5, 0, 0.5, 1, 2, 3])
bx.set_yticklabels(["$-0.5$", "0", "0.5", "1", "2", "3"])
bx.set_xscale("log")
bx.set_xlim(0.012, 14)
bx.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
bx.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
bx.tick_params(axis="x", which="minor", bottom=False)
bx.set_xlabel("displacement in units of target size,  $r$")
bx.legend(frameon=False, fontsize=5.7, loc="upper left", handletextpad=0.3,
          borderpad=0.1, labelspacing=0.18, handlelength=1.6,
          title="corpus, $(U{+}D)/G$ densest", title_fontsize=5.7)
bx.text(0.283, -0.72, "$\\theta$ bands", fontsize=6.2, color=C_NEUTRAL, ha="center")
ax.text(0.0145, 0.92, "(a)", fontsize=6.6, color=C_NEUTRAL, ha="left", va="top")
bx.text(0.0145, -0.62, "(b)", fontsize=6.6, color=C_NEUTRAL, ha="left", va="top")
for side in ("top", "right"):
    bx.spines[side].set_visible(False)

fig.savefig(OUT / "fig_geometry_and_sign.pdf")
fig.savefig(OUT / "fig_geometry_and_sign.png", dpi=400)
print("wrote", OUT / "fig_geometry_and_sign.pdf")
for label, x, y in (("vineyard2024", grape_r, grape_e),
                    ("vineyard2023", bodegas_r, bodegas_e),
                    ("mot17", mot17_r, mot17_e), ("mot20", mot20_r, mot20_e)):
    print(label, "crossing r =", crossing(x, y))
