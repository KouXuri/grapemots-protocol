#!/usr/bin/env python3
"""Submission figures for Precision Agriculture (2026-11-21), drawn from results/ only.

PA is a small-format Springer journal (typeset page 155 x 235 mm, read from a 2025 PA PDF on 2026-11-21).
Its artwork rules for small-sized journals: figures 119 mm wide and at most 195 mm high; Helvetica or Arial
lettering of 8-12 pt, kept consistent; lines at least 0.3 pt; no titles inside the artwork; parts labelled
a, b; information readable in greyscale (colour is online only). All lettering here is Arial 8 pt
(TrueType embedded), every line >= 0.5 pt, every colour paired with a marker, fill or hatch.
Colours: Okabe-Ito blue #0072B2, vermilion #D55E00 and green #009E73 (validated earlier, all pairs, light
surface), ink #333333; grey #8C8C8C / #D9D9D9 only for reference marks and distributions.

Main text   Fig1.pdf/.eps  association alone (results/oracle_trackers_summary_1120.json)
            Fig2.pdf/.eps  missed bunches supplied alone (results/error_sources_summary_s3072.json)
            Fig3.pdf/.eps  error terms by interval; count error against coverage (results/journal_numbers.json,
                           results/crop_vs_whole.json)
            Fig4.pdf/.eps  vine-level error before and after calibration (tools/calibration_transfer_1120.py
                           records, results/calibration_robustness_1121.json)
Online Resource 1   FigS1.pdf  interval bracket of the sign change (results/journal_numbers.json)
                    FigS2.pdf  HOTA by interval; highest-HOTA vs smallest-|e| arm (results/identity_metrics.json)
Output folder: figures/pa_submission/.
    .venv/bin/python cadence_journal_1119/tools/make_figures_pa119_1121.py
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
OUT = HERE / "figures/pa_submission"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W = 119 * MM
BLUE, VERM, GREEN, GREY, LIGHT, INK = "#0072B2", "#D55E00", "#009E73", "#8C8C8C", "#D9D9D9", "#333333"
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
    "axes.edgecolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.color": INK, "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.0, "lines.markersize": 4, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
    "hatch.linewidth": 0.5, "legend.frameon": False, "legend.handletextpad": 0.4,
})
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)


def save(fig, name, eps=True):
    fig.savefig(OUT / f"{name}.pdf")
    if eps:
        fig.savefig(OUT / f"{name}.eps")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", pad=3)


O = json.loads((RES / "oracle_trackers_summary_1120.json").read_text())
N = json.loads((RES / "journal_numbers.json").read_text())
C = json.loads((RES / "crop_vs_whole.json").read_text())

# ----------------------------------------------------------------------------- Fig. 1
TRK = [("botsort", "BoT-SORT"), ("deepocsort", "Deep OC-SORT"), ("strongsort", "StrongSORT"),
       ("ocsort", "OC-SORT"), ("bytetrack", "ByteTrack")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 46 * MM), gridspec_kw=dict(wspace=0.12), sharey=True)
y = np.arange(len(TRK))[::-1]
for yi, (k, name) in zip(y, TRK):
    mv = 100 * O[k]["d1|tau1"]["groups"]["PP2-8"]["e"]
    fr = 100 * O[k]["d2|tau1"]["groups"]["NP1-3"]["e"]
    a1.plot([fr, mv], [yi, yi], color=LIGHT, lw=1.5, zorder=1, solid_capstyle="round")
    a1.plot(mv, yi, "o", color=BLUE, ms=4.5, zorder=3, label="Multi-view" if yi == y[0] else None)
    a1.plot(fr, yi, "s", mfc="white", mec=INK, mew=0.9, ms=4, zorder=3, label="Frontal" if yi == y[0] else None)
    a1.text(mv + 12, yi, f"{mv:.0f}", va="center", ha="left", fontsize=8)
a1.set_yticks(y, [n for _, n in TRK])
a1.axvline(0, color=INK, lw=0.6)
a1.set_xlim(-10, 345)
a1.set_xticks([0, 100, 200, 300])
a1.set_xlabel("Count error (%)")
a1.legend(loc="upper right", borderaxespad=0.1)
label(a1, "a")
cls = [("switch_in_view", "Switch in view", BLUE, None), ("reentry_beyond_buffer", "Re-entry", VERM, "////"),
       ("short_occlusion", "Short gap", GREEN, "....")]
G = O["botsort"]["d1|tau1"]["groups"]["PP2-8"]["G"]
for yi, (k, name) in zip(y, TRK):
    g = O[k]["d1|tau1"]["groups"]["PP2-8"]
    left = 0.0
    for key, lab, col, hatch in cls:
        w = g.get(key, 0) / G
        a2.barh(yi, w, left=left, height=0.56, color=col if hatch is None else "white", edgecolor=col if hatch else "white",
                hatch=hatch, linewidth=0.8 if hatch else 0.5, label=lab if yi == y[0] else None)
        left += w
    a2.text(left + 0.06, yi, f"{g['D_per_G']:.2f}", va="center", ha="left", fontsize=8)
a2.set_xlim(0, 3.3)
a2.set_xticks([0, 1, 2, 3])
a2.set_xlabel("Duplicates per reference trajectory")
a2.legend(loc="upper right", handlelength=1.2, borderaxespad=0.1)
label(a2, "b")
a2.tick_params(axis="y", length=0)
save(fig, "Fig1")

# ----------------------------------------------------------------------------- Fig. 2
S = json.loads((RES / "error_sources_summary_s3072.json").read_text())
SETS = [("crop_s0", "Crop, seed 0"), ("crop_s1", "Crop, seed 1"), ("crop_s2", "Crop, seed 2"),
        ("whole_s0", "Whole, seed 0"), ("whole_s1", "Whole, seed 1"), ("whole_s2", "Whole, seed 2")]
RATE = {"1": "15 Hz", "2": "7.5 Hz", "4": "3.75 Hz", "8": "1.9 Hz"}
fig, axs = plt.subplots(2, 2, figsize=(W, 64 * MM), sharey=True, sharex=True,
                        gridspec_kw=dict(wspace=0.08, hspace=0.36))
yy = np.arange(len(SETS))[::-1]
for ax, d in zip(axs.flat, ("1", "2", "4", "8")):
    ref = 100 * O["botsort"][f"d{d}|tau1"]["groups"]["PP2-8"]["e"]
    ax.axvline(0, color=INK, lw=0.6)
    ax.axvline(ref, color=GREY, lw=0.8, ls=(0, (3, 2)))
    for yi, (k, name) in zip(yy, SETS):
        w = S[k]["by_interval"][d]
        e0 = 100 * w["none"]["e"]
        e1 = 100 * (w["none"]["e"] + w["alone"]["miss"]["e"])
        ax.annotate("", xy=(e1, yi), xytext=(e0, yi),
                    arrowprops=dict(arrowstyle="-|>", color=VERM, lw=0.8, shrinkA=3, shrinkB=2, mutation_scale=6))
        ax.plot(e0, yi, "o", color=BLUE, ms=3.8, zorder=3)
        ax.plot(e1, yi, "^", mfc="white", mec=VERM, mew=0.9, ms=3.8, zorder=3)
    ax.set_title(RATE[d], pad=2)
    ax.set_xlim(-60, 340)
    ax.set_xticks([0, 100, 200, 300])
for ax in axs[:, 0]:
    ax.set_yticks(yy, [n for _, n in SETS])
for ax in axs[:, 1]:
    ax.tick_params(axis="y", length=0)
for ax in axs[1]:
    ax.set_xlabel("Count error (%)")
fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", ms=3.8, label="Real detections"),
                    Line2D([], [], marker="^", mfc="white", mec=VERM, ls="", ms=3.8, label="Missed bunches supplied"),
                    Line2D([], [], color=GREY, ls=(0, (3, 2)), lw=0.8, label="Reference boxes")],
           loc="lower center", bbox_to_anchor=(0.55, 0.955), ncol=3, columnspacing=1.0, handlelength=1.6)
save(fig, "Fig2")

# ----------------------------------------------------------------------------- Fig. 3
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 52 * MM), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1.05]))
P = N["pooled_terms_by_interval"]
spec = [("U_per_G", GREEN, "o", "U/G"), ("D_per_G", BLUE, "s", "D/G"), ("M_per_G", VERM, "^", "M/G")]
for fam, ls in (("whole", "-"), ("crop", "--")):
    for key, col, mk, lab in spec:
        ys = [P[f"{fam}|{d}"][key] for d in DELTAS]
        a.plot(DELTAS, ys, color=col, marker=mk, ls=ls, lw=1.0, ms=3.6, mfc=col if fam == "whole" else "white")
        if fam == "whole":
            at = (8, ys[-1])
            dy = {"U_per_G": -6, "D_per_G": 2, "M_per_G": 0}[key]
            a.annotate(lab, at, xytext=(5, dy), textcoords="offset points", va="center", fontsize=8,
                       color=col if key != "U_per_G" else GREEN)
a.set_xscale("log", base=2)
a.set_xticks(DELTAS, ["1", "2", "4", "8"])
a.minorticks_off()
a.set_xlim(0.85, 14)
a.set_ylim(0.25, 0.85)
a.set_xlabel(r"Processing interval $\Delta$")
a.set_ylabel("Per reference trajectory")
a.legend(handles=[Line2D([], [], color=INK, ls="-", lw=1.0, label="Whole frame"),
                  Line2D([], [], color=INK, ls="--", lw=1.0, label="Crop")], loc="upper right",
         ncol=1, borderaxespad=0.1, handlelength=1.8)
label(a, "a")
xs, ys, near = [], [], []
for k, ev in C["error"].items():
    xs.append(C["coverage"][k]); ys.append(ev); near.append(abs(ev) <= 0.10)
xs, ys, near = np.array(xs), np.array(ys), np.array(near)
b.scatter(xs[~near], ys[~near], s=5, color="#BDBDBD", lw=0, zorder=2, label="Other arms")
b.scatter(xs[near], ys[near], s=9, color=INK, marker="D", lw=0, zorder=3, label=r"$|e|\leq 0.10$")
b.axhline(0, color=INK, lw=0.6)
b.axvline(max(xs[near]), color=INK, lw=0.6, ls=":")
b.set_xlabel(r"Coverage $1-M/G$")
b.set_ylabel(r"Count error $e$")
b.set_xlim(0, 1)
b.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
b.legend(loc="upper left", markerscale=1.4, borderaxespad=0.1, handletextpad=0.2)
label(b, "b")
save(fig, "Fig3")

# ----------------------------------------------------------------------------- Fig. 4
NS = runpy.run_path(str(HERE / "tools/calibration_transfer_1120.py"), run_name="calib")
PP, GF, CONFIGS = NS["P"], NS["GF"], NS["CONFIGS"]
pc = lambda c, v: PP[(c[0], c[1], v, c[2], c[3])]
near_c = [c for c in CONFIGS if abs(sum(pc(c, v) for v in VIDEOS) / sum(GF.values()) - 1) <= 0.10]
assert len(near_c) == 14, len(near_c)
R = json.loads((RES / "calibration_robustness_1121.json").read_text())
chosen = {v: tuple(int(x) if x.isdigit() else x for x in s.split("|")) for v, s in R["nested_choice"]["chosen"].items()}
fig, (b1, b2) = plt.subplots(1, 2, figsize=(W, 48 * MM), gridspec_kw=dict(wspace=0.34))
rng = np.random.default_rng(0)
xs_ = np.arange(len(VIDEOS))
for xi, v in zip(xs_, VIDEOS):
    ev = np.array([100 * (pc(c, v) / GF[v] - 1) for c in near_c])
    b1.plot(xi + rng.uniform(-0.18, 0.18, len(ev)), ev, "o", mfc="white", mec=BLUE, mew=0.7, ms=3.0)
    lov = []
    for c in CONFIGS:
        oth = [u for u in VIDEOS if u != v]
        k = sum(GF[u] for u in oth) / sum(pc(c, u) for u in oth)
        lov.append(100 * (k * pc(c, v) / GF[v] - 1))
    lo, q1, md, q3, hi = np.percentile(lov, [10, 25, 50, 75, 90])
    b2.plot([xi, xi], [lo, hi], color=GREY, lw=0.8)
    b2.add_patch(plt.Rectangle((xi - 0.22, q1), 0.44, q3 - q1, facecolor=LIGHT, edgecolor=GREY, lw=0.6))
    b2.plot([xi - 0.22, xi + 0.22], [md, md], color=INK, lw=1.0)
    c = chosen[v]
    oth = [u for u in VIDEOS if u != v]
    k = sum(GF[u] for u in oth) / sum(pc(c, u) for u in oth)
    b2.plot(xi, 100 * (k * pc(c, v) / GF[v] - 1), "D", color=VERM, ms=3.8, zorder=4)
for ax in (b1, b2):
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(xs_, [v[-1] for v in VIDEOS])
    ax.set_xlabel("Vine (PP)")
b1.set_ylabel("Count error (%)")
label(b1, "a")
b2.set_ylabel("Error after calibration (%)")
label(b2, "b")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=LIGHT, edgecolor=GREY, lw=0.6, label="All 120 configurations"),
                    Line2D([], [], marker="D", color=VERM, ls="", ms=3.8, label="Configuration chosen on the other six vines")],
           loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, handlelength=1.2, columnspacing=1.2)
save(fig, "Fig4")

# ----------------------------------------------------------------------------- Fig. S1 (sign-change maps)
cats = ["<1", "(1,2)", "(2,4)", "(4,8)", ">8"]
labels_ = ["Under at every interval", "Changes 1→2", "Changes 2→4", "Changes 4→8", "Over at every interval"]
cmap = ListedColormap(["#D9D9D9", "#9ECAE1", "#4292C6", "#08519C", "#252525"])
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
fig, axes = plt.subplots(2, 3, figsize=(W, 82 * MM), sharex=True, sharey=True,
                         gridspec_kw=dict(wspace=0.08, hspace=0.25))
FAM = {"crop": "Crop", "whole": "Whole frame"}
for i, fam in enumerate(("crop", "whole")):
    for j, sd in enumerate(SEEDS):
        ax = axes[i, j]
        grid = np.array([[cats.index(N["crossing_map"][f"{fam}|{sd}|{v}|{s}"]) for s in SIGMAS] for v in VIDEOS])
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(5), ["1536", "2048", "2560", "3072", "3840"], rotation=90)
        ax.set_yticks(range(7), ["PP" + v[-1] for v in VIDEOS])
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.0)
        ax.tick_params(which="minor", length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{FAM[fam]}, seed {sd}", pad=2)
        if i == 1 and j == 1:
            ax.set_xlabel(r"Inference scale $\sigma$ (pixels)")
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(5)]
fig.legend(handles, labels_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.07), handlelength=1.2,
           columnspacing=1.0)
save(fig, "FigS1", eps=False)

# ----------------------------------------------------------------------------- Fig. S2 (HOTA)
I = json.loads((RES / "identity_metrics.json").read_text())
e_whole = lambda sd, v, s, d: C["error"][f"whole|{sd}|{v}|{s}|{d}"]
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 58 * MM), gridspec_kw=dict(wspace=0.36))
shades = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
marks = ["o", "s", "^", "D", "v"]
for s, col, mk in zip(SIGMAS, shades, marks):
    for mode, ls in (("processed", "-"), ("common", ":")):
        ys = [st.median(I[f"surface_seed{sd}|{v}|{s}|{d}"][mode]["HOTA"] for sd in SEEDS for v in VIDEOS) for d in DELTAS]
        a.plot(DELTAS, ys, color=col, ls=ls, marker=mk, ms=3.0, mfc=col if mode == "processed" else "white",
               label=f"σ = {s}" if mode == "processed" else None)
a.set_xscale("log", base=2)
a.set_xticks(DELTAS, ["1", "2", "4", "8"])
a.minorticks_off()
a.set_xlabel(r"Processing interval $\Delta$")
a.set_ylabel("HOTA, median of 21 cases")
label(a, "a")
for sd in SEEDS:
    for v in VIDEOS:
        arms = [(s, d) for s in SIGMAS for d in DELTAS]
        hh = {a_: I[f"surface_seed{sd}|{v}|{a_[0]}|{a_[1]}"]["processed"]["HOTA"] for a_ in arms}
        ee = {a_: abs(e_whole(sd, v, *a_)) for a_ in arms}
        bh = max(arms, key=hh.get)
        be = min(arms, key=ee.get)
        b.annotate("", xy=(ee[be], hh[be]), xytext=(ee[bh], hh[bh]),
                   arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
        b.scatter([ee[bh]], [hh[bh]], s=10, color=BLUE, marker="o", zorder=3, lw=0)
        b.scatter([ee[be]], [hh[be]], s=10, color=VERM, marker="s", zorder=3, lw=0)
b.set_xlabel(r"Absolute count error $|e|$")
b.set_ylabel("HOTA, processed frames")
label(b, "b")
h, _ = a.get_legend_handles_labels()
h += [Line2D([], [], color=INK, ls="-", label="Processed frames"),
      Line2D([], [], color=INK, ls=":", label="All annotated frames"),
      Line2D([], [], color=BLUE, marker="o", ls="none", ms=3.5, label="Highest HOTA"),
      Line2D([], [], color=VERM, marker="s", ls="none", ms=3.5, label="Smallest |e|")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, columnspacing=0.9, handlelength=1.6)
save(fig, "FigS2", eps=False)
print("written:", sorted(p.name for p in OUT.glob("*.pdf")))
