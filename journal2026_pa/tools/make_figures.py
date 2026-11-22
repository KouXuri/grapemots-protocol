#!/usr/bin/env python3
"""Figures for the journal manuscript, drawn from results/journal_numbers.json and the
frozen per-arm values in results/crop_vs_whole.json and results/identity_metrics.json.

Springer artwork rules: sans-serif lettering (Arial) at 8 pt, widths 84 mm (single
column) or 174 mm (full width), TrueType embedding (pdf.fonttype 42), readable in
greyscale: every colour is paired with a marker or a line style. Okabe-Ito colours,
checked with the dataviz validator (both sets pass; the three decomposition colours
are below 3:1 contrast, so their lines carry direct labels and distinct markers).

    .venv/bin/python cadence_journal_1119/tools/make_figures.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
FULL, SINGLE = 174 * MM, 84 * MM

BLUE, VERM, GREEN, PURPLE, ORANGE, SKY, INK = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                              "#E69F00", "#56B4E9", "#333333")
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "axes.edgecolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 4, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
})

N = json.loads((RES / "journal_numbers.json").read_text())
C = json.loads((RES / "crop_vs_whole.json").read_text())
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SHORT = {v: "PP" + v[-1] for v in VIDEOS}
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
FAM_STYLE = {"crop": dict(color=VERM, ls="--", label="crop-trained"),
             "whole": dict(color=BLUE, ls="-", label="whole-frame")}
SEED_MARK = {0: "o", 1: "s", 2: "^"}


def e(fam, sd, v, s, d):
    return C["error"][f"{fam}|{sd}|{v}|{s}|{d}"]


def cov(fam, sd, v, s, d):
    return C["coverage"][f"{fam}|{sd}|{v}|{s}|{d}"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2: error vs interval
def fig_surfaces(sigma=3072):
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 72 * MM), sharex=True)
    axes = axes.ravel()
    for ax, v in zip(axes, VIDEOS):
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        for fam in ("crop", "whole"):
            for sd in SEEDS:
                ys = [e(fam, sd, v, sigma, d) for d in DELTAS]
                ax.plot(DELTAS, ys, marker=SEED_MARK[sd], mfc="white" if fam == "crop" else FAM_STYLE[fam]["color"],
                        color=FAM_STYLE[fam]["color"], ls=FAM_STYLE[fam]["ls"], lw=1.0, ms=3.6, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DELTAS, [str(d) for d in DELTAS])
        ax.set_title(f"{SHORT[v]}  (G = {N['G_full'][v]})", pad=3)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    fig.supxlabel(r"Processing interval, $\Delta$ (annotated frames)", fontsize=8, y=0.01)
    fig.supylabel(r"Count error, $e=(P-G)/G$", fontsize=8, x=0.005)
    leg = axes[7]
    leg.axis("off")
    handles = [Line2D([], [], color=FAM_STYLE[f]["color"], ls=FAM_STYLE[f]["ls"], label=FAM_STYLE[f]["label"])
               for f in ("whole", "crop")]
    handles += [Line2D([], [], color=INK, ls="none", marker=SEED_MARK[sd], mfc="white", label=f"seed {sd}")
                for sd in SEEDS]
    leg.legend(handles=handles, loc="center", frameon=False, title=f"Detectors (σ = {sigma})",
               title_fontsize=8)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, "fig2_error_vs_interval")


# ------------------------------------------------------------------ Fig. 3: sign-change maps
def fig_maps():
    cats = ["<1", "(1,2)", "(2,4)", "(4,8)", ">8"]
    labels = ["under at every Δ", "changes 1→2", "changes 2→4", "changes 4→8", "over at every Δ"]
    cmap = ListedColormap(["#d9d9d9", "#9ecae1", "#4292c6", "#08519c", "#252525"])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 78 * MM), sharex=True, sharey=True)
    for i, fam in enumerate(("crop", "whole")):
        for j, sd in enumerate(SEEDS):
            ax = axes[i, j]
            grid = np.array([[cats.index(N["crossing_map"][f"{fam}|{sd}|{v}|{s}"]) for s in SIGMAS] for v in VIDEOS])
            ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(5), [str(s) for s in SIGMAS], rotation=0)
            ax.set_yticks(range(7), [SHORT[v] for v in VIDEOS])
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
            ax.grid(which="minor", color="white", lw=1.2)
            ax.tick_params(which="minor", length=0)
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{FAM_STYLE[fam]['label']}, seed {sd}", pad=3)
            if i == 1:
                ax.set_xlabel(r"Inference scale, $\sigma$ (px)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(5)]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.6, h_pad=0.8)
    save(fig, "fig3_sign_change_maps")


# ------------------------------------------------------------------ Fig. 4: terms and cancellation
def fig_terms():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 64 * MM), gridspec_kw={"width_ratios": [1, 1.15]})
    P = N["pooled_terms_by_interval"]
    spec = [("U_per_G", ORANGE, "o", "U/G ownerless"), ("D_per_G", PURPLE, "s", "D/G duplicate"),
            ("M_per_G", SKY, "^", "M/G unreached")]
    for fam in ("crop", "whole"):
        for key, col, mk, lab in spec:
            ys = [P[f"{fam}|{d}"][key] for d in DELTAS]
            a.plot(DELTAS, ys, color=col, marker=mk, ls=FAM_STYLE[fam]["ls"],
                   mfc="white" if fam == "crop" else col, ms=4)
            if fam == "whole":
                at = (1, ys[0]) if key != "M_per_G" else (8, ys[-1])
                off = {"U_per_G": (6, 4), "D_per_G": (6, -7), "M_per_G": (5, 0)}[key]
                a.annotate(lab, at, xytext=off, textcoords="offset points", va="center",
                           color=INK, fontsize=7.5)
    a.set_xscale("log", base=2)
    a.set_xticks(DELTAS, [str(d) for d in DELTAS])
    a.set_xlim(0.85, 30)
    a.set_xlabel(r"Processing interval, $\Delta$")
    a.set_ylabel("Pooled terms per reference trajectory")
    a.grid(axis="y", color="#e6e6e6", lw=0.5)
    a.legend(handles=[Line2D([], [], color=INK, ls=FAM_STYLE[f]["ls"], label=FAM_STYLE[f]["label"])
                      for f in ("whole", "crop")], frameon=False, loc="upper right")
    a.set_title("(a)", loc="left", fontweight="bold")
    xs, ys, near = [], [], []
    for k, ev in C["error"].items():
        cv = C["coverage"][k]
        xs.append(cv); ys.append(ev)
        near.append(abs(ev) <= 0.10)
    xs, ys, near = np.array(xs), np.array(ys), np.array(near)
    b.scatter(xs[~near], ys[~near], s=6, color="#bdbdbd", lw=0, label="other arms", zorder=2)
    b.scatter(xs[near], ys[near], s=10, color=VERM, marker="D", lw=0,
              label=r"$|e|\leq 0.10$", zorder=3)
    b.axhline(0, color=INK, lw=0.6)
    b.axvline(max(xs[near]), color=VERM, lw=0.6, ls=":")
    b.annotate(f"highest coverage among\nnear-zero arms: {max(xs[near]):.2f}", (max(xs[near]), -0.75),
               xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK, va="center")
    b.set_xlabel(r"Coverage, $1-M/G$")
    b.set_ylabel(r"Count error, $e$")
    b.set_xlim(0, 1)
    b.legend(frameon=False, loc="upper left", markerscale=1.5)
    b.set_title("(b)", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=2)
    save(fig, "fig4_terms_and_cancellation")


# ------------------------------------------------------------------ Fig. 6: count error vs HOTA
def fig_hota():
    I = json.loads((RES / "identity_metrics.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 72 * MM))
    shades = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    marks = ["o", "s", "^", "D", "v"]
    for s, col, mk in zip(SIGMAS, shades, marks):
        for mode, ls in (("processed", "-"), ("common", ":")):
            ys = [st.median(I[f"surface_seed{sd}|{v}|{s}|{d}"][mode]["HOTA"] for sd in SEEDS for v in VIDEOS)
                  for d in DELTAS]
            a.plot(DELTAS, ys, color=col, ls=ls, marker=mk, ms=3.5, mfc=col if mode == "processed" else "white",
                   label=f"σ = {s}" if mode == "processed" else None)
    a.set_xscale("log", base=2)
    a.set_xticks(DELTAS, [str(d) for d in DELTAS])
    a.set_xlabel(r"Processing interval, $\Delta$")
    a.set_ylabel("HOTA (median over 21 video–seed cases)")
    a.grid(axis="y", color="#e6e6e6", lw=0.5)
    h, l = a.get_legend_handles_labels()
    h += [Line2D([], [], color=INK, ls="-", label="processed frames"),
          Line2D([], [], color=INK, ls=":", label="all annotated frames")]
    a.legend(handles=h, frameon=False, fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.22),
             columnspacing=0.8, handlelength=1.6)
    a.set_title("(a)", loc="left", fontweight="bold")
    for sd in SEEDS:
        for v in VIDEOS:
            arms = [(s, d) for s in SIGMAS for d in DELTAS]
            hh = {a_: I[f"surface_seed{sd}|{v}|{a_[0]}|{a_[1]}"]["processed"]["HOTA"] for a_ in arms}
            ee = {a_: abs(e("whole", sd, v, *a_)) for a_ in arms}
            bh = max(arms, key=hh.get)
            be = min(arms, key=ee.get)
            b.annotate("", xy=(ee[be], hh[be]), xytext=(ee[bh], hh[bh]),
                       arrowprops=dict(arrowstyle="->", color="#9e9e9e", lw=0.6))
            b.scatter([ee[bh]], [hh[bh]], s=14, color=BLUE, marker="o", zorder=3, lw=0)
            b.scatter([ee[be]], [hh[be]], s=14, color=VERM, marker="s", zorder=3, lw=0)
    b.set_xlabel(r"Absolute count error, $|e|$")
    b.set_ylabel("HOTA (processed frames)")
    b.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="none", label="highest HOTA"),
                      Line2D([], [], color=VERM, marker="s", ls="none", label="smallest |e|")],
             frameon=False, loc="upper right")
    b.grid(color="#eeeeee", lw=0.5)
    b.set_title("(b)", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=2)
    fig.subplots_adjust(bottom=0.32)
    save(fig, "fig6_count_vs_hota")


if __name__ == "__main__":
    fig_surfaces()
    fig_maps()
    fig_terms()
    fig_hota()
    print("written to", OUT)
