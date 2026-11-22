#!/usr/bin/env python3
"""Figures 1, 2 and 6 of the 2026-11-21 restructure, from results/ only (style as make_figures.py).

Fig. 1  fig1_oracle_associators.pdf  association alone: count error per associator, multi-view vs frontal
        at 15 Hz, and the multi-view duplicates by class   (results/oracle_trackers_summary_1120.json)
Fig. 2  fig2_miss_alone.pdf          count error of the six detector sets at sigma 3072: real detections ->
        missed bunches supplied alone; the reference-box value marked   (results/error_sources_summary_s3072.json,
        results/oracle_trackers_summary_1120.json)
Fig. 6  fig5_vines.pdf               per-vine error of the 14 configurations whose total is within 10%, and per-vine
        error after leave-one-vine-out calibration   (calibration_transfer_1120.py records, run here unchanged)
Springer artwork rules: Arial 8 pt, 174 mm wide, TrueType embedding, greyscale-readable (every colour paired with a
marker, a fill or a hatch). Okabe-Ito blue/vermilion/green as categories (validate_palette.py, all pairs, light: all checks pass);
grey only for reference lines and distribution boxes.
    .venv/bin/python cadence_journal_1119/tools/make_figures_1121.py
"""
from __future__ import annotations

import json
import runpy
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
RES = HERE / "results"
OUT = HERE / "figures"
MM = 1 / 25.4
FULL = 174 * MM
BLUE, VERM, GREEN, GREY, LIGHT, INK = "#0072B2", "#D55E00", "#009E73", "#8C8C8C", "#D9D9D9", "#333333"
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "axes.edgecolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 4, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
    "hatch.linewidth": 0.5,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 1
O = json.loads((RES / "oracle_trackers_summary_1120.json").read_text())
TRK = [("botsort", "BoT-SORT"), ("deepocsort", "Deep OC-SORT"), ("strongsort", "StrongSORT"),
       ("ocsort", "OC-SORT"), ("bytetrack", "ByteTrack")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 52 * MM), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.08),
                             sharey=True)
y = np.arange(len(TRK))[::-1]
for yi, (k, name) in zip(y, TRK):
    mv = 100 * O[k]["d1|tau1"]["groups"]["PP2-8"]["e"]
    fr = 100 * O[k]["d2|tau1"]["groups"]["NP1-3"]["e"]
    a1.plot([fr, mv], [yi, yi], color=LIGHT, lw=1.6, zorder=1, solid_capstyle="round")
    a1.plot(mv, yi, "o", color=BLUE, ms=5.5, zorder=3, label="Multi-view (PP2–PP8)" if yi == y[0] else None)
    a1.plot(fr, yi, "s", mfc="white", mec=VERM, mew=1.1, ms=5, zorder=3, label="Frontal (NP1–NP3)" if yi == y[0] else None)
    a1.text(mv + 7, yi, f"{mv:.0f}%", va="center", ha="left", fontsize=7, color=INK)
a1.set_yticks(y, [n for _, n in TRK])
a1.axvline(0, color=INK, lw=0.6)
a1.set_xlim(-5, 330)
a1.set_xlabel("Count error with reference boxes at 15 Hz (%)")
a1.legend(loc="upper right", frameon=False, handletextpad=0.3)
a1.set_title("a", loc="left", fontweight="bold")
cls = [("switch_in_view", "Switch in view", BLUE, None), ("reentry_beyond_buffer", "Re-entry beyond retention", VERM, "////"),
       ("short_occlusion", "Short gap", GREEN, "....")]
G = O["botsort"]["d1|tau1"]["groups"]["PP2-8"]["G"]
for yi, (k, name) in zip(y, TRK):
    g = O[k]["d1|tau1"]["groups"]["PP2-8"]
    left = 0.0
    for key, lab, col, hatch in cls:
        w = g.get(key, 0) / G
        a2.barh(yi, w, left=left, height=0.55, color=col if hatch is None else "white", edgecolor=col if hatch else "white",
                hatch=hatch, linewidth=0.8 if hatch else 0.6, label=lab if yi == y[0] else None)
        left += w
    a2.text(left + 0.04, yi, f"{g['D_per_G']:.2f}", va="center", ha="left", fontsize=7, color=INK)
a2.set_xlim(0, 3.1)
a2.set_xlabel("Duplicates per reference trajectory")
a2.legend(loc="upper right", frameon=False, handlelength=1.2, handletextpad=0.4)
a2.set_title("b", loc="left", fontweight="bold")
a2.tick_params(axis="y", length=0)
save(fig, "fig1_oracle_associators")

# ----------------------------------------------------------------------------- Fig. 2
S = json.loads((RES / "error_sources_summary_s3072.json").read_text())
SETS = [("crop_s0", "Crop, seed 0"), ("crop_s1", "Crop, seed 1"), ("crop_s2", "Crop, seed 2"),
        ("whole_s0", "Whole frame, seed 0"), ("whole_s1", "Whole frame, seed 1"), ("whole_s2", "Whole frame, seed 2")]
RATE = {"1": "15 Hz", "2": "7.5 Hz", "4": "3.75 Hz", "8": "1.9 Hz"}
fig, axs = plt.subplots(1, 4, figsize=(FULL, 50 * MM), sharey=True, gridspec_kw=dict(wspace=0.12))
y = np.arange(len(SETS))[::-1]
for ax, d in zip(axs, ("1", "2", "4", "8")):
    ref = 100 * O["botsort"][f"d{d}|tau1"]["groups"]["PP2-8"]["e"]
    ax.axvline(0, color=INK, lw=0.6)
    ax.axvline(ref, color=GREY, lw=0.8, ls=(0, (3, 2)))
    for yi, (k, name) in zip(y, SETS):
        w = S[k]["by_interval"][d]
        e0 = 100 * w["none"]["e"]
        e1 = 100 * (w["none"]["e"] + w["alone"]["miss"]["e"])
        ax.annotate("", xy=(e1, yi), xytext=(e0, yi),
                    arrowprops=dict(arrowstyle="-|>", color=VERM, lw=0.9, shrinkA=3, shrinkB=2, mutation_scale=7))
        ax.plot(e0, yi, "o", color=BLUE, ms=4.5, zorder=3)
        ax.plot(e1, yi, "^", mfc="white", mec=VERM, mew=1.0, ms=4.5, zorder=3)
    ax.set_title(RATE[d], fontsize=8)
    ax.set_xlim(-60, 340)
    ax.set_xticks([0, 100, 200, 300])
axs[0].set_yticks(y, [n for _, n in SETS])
for ax in axs[1:]:
    ax.tick_params(axis="y", length=0)
fig.supxlabel("Count error (%)", y=-0.07, fontsize=8)
from matplotlib.lines import Line2D  # noqa: E402
fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", ms=4.5, label="Real detections"),
                    Line2D([], [], marker="^", mfc="white", mec=VERM, ls="", ms=4.5, label="Missed bunches supplied, all else unchanged"),
                    Line2D([], [], color=GREY, ls=(0, (3, 2)), lw=0.8, label="Reference boxes")],
           loc="upper center", bbox_to_anchor=(0.55, 1.10), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2)
save(fig, "fig2_miss_alone")

# ----------------------------------------------------------------------------- Fig. 6 (vines)
NS = runpy.run_path(str(HERE / "tools/calibration_transfer_1120.py"), run_name="calib")
P, GF, CONFIGS, VIDEOS = NS["P"], NS["GF"], NS["CONFIGS"], NS["VIDEOS"]
pc = lambda c, v: P[(c[0], c[1], v, c[2], c[3])]
near = [c for c in CONFIGS if abs(sum(pc(c, v) for v in VIDEOS) / sum(GF.values()) - 1) <= 0.10]
assert len(near) == 14, len(near)
R = json.loads((RES / "calibration_robustness_1121.json").read_text())
chosen = {v: tuple(int(x) if x.isdigit() else x for x in s.split("|")) for v, s in R["nested_choice"]["chosen"].items()}
fig, (b1, b2) = plt.subplots(1, 2, figsize=(FULL, 58 * MM), gridspec_kw=dict(wspace=0.22))
rng = np.random.default_rng(0)
xs = np.arange(len(VIDEOS))
for xi, v in zip(xs, VIDEOS):
    ev = np.array([100 * (pc(c, v) / GF[v] - 1) for c in near])
    b1.plot(xi + rng.uniform(-0.18, 0.18, len(ev)), ev, "o", mfc="white", mec=BLUE, mew=0.8, ms=3.6)
    lov = []
    for c in CONFIGS:
        oth = [u for u in VIDEOS if u != v]
        k = sum(GF[u] for u in oth) / sum(pc(c, u) for u in oth)
        lov.append(100 * (k * pc(c, v) / GF[v] - 1))
    lo, q1, md, q3, hi = np.percentile(lov, [10, 25, 50, 75, 90])
    b2.plot([xi, xi], [lo, hi], color=GREY, lw=0.8)
    b2.add_patch(plt.Rectangle((xi - 0.2, q1), 0.4, q3 - q1, facecolor=LIGHT, edgecolor=GREY, lw=0.6))
    b2.plot([xi - 0.2, xi + 0.2], [md, md], color=INK, lw=1.0)
    c = chosen[v]
    oth = [u for u in VIDEOS if u != v]
    k = sum(GF[u] for u in oth) / sum(pc(c, u) for u in oth)
    b2.plot(xi, 100 * (k * pc(c, v) / GF[v] - 1), "D", color=VERM, ms=4.5, zorder=4)
for ax in (b1, b2):
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(xs, ["PP" + v[-1] for v in VIDEOS])
b1.set_ylabel("Count error of the vine (%)")
b1.set_title("a  Totals within 10%: 14 configurations", loc="left", fontweight="bold")
b2.set_ylabel("Calibrated error of the vine (%)")
b2.set_title("b  Calibrated, 120 configurations", loc="left", fontweight="bold")
b2.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=LIGHT, edgecolor=GREY, lw=0.6, label="120 configurations"),
                   Line2D([], [], marker="D", color=VERM, ls="", ms=4.5, label="Chosen on the other six vines")],
          loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.5)
save(fig, "fig5_vines")
print("figures written:", [p.name for p in sorted(OUT.glob("fig[125]_*.pdf"))])
