#!/usr/bin/env python3
"""The three further figures of the full Precision Agriculture version, redrawn at the text width of the
Springer Nature template (131 mm) so that 8 pt lettering stays 8 pt in the typeset page (2026-11-21).

pa_fig_terms.pdf, pa_fig_maps.pdf: the unchanged drawing functions of make_figures.py (same data, same
encodings) run with the width set to 131 mm; panel labels set to bold a, b as in the other figures.
pa_fig_hota.pdf: the same data and encodings as make_figures.fig_hota (results/identity_metrics.json,
results/crop_vs_whole.json), re-laid out for the narrower width: one legend row under both panels.
    .venv/bin/python cadence_journal_1119/tools/make_figures_pa_extra_1121.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

plt, Line2D, MM = mf.plt, mf.Line2D, mf.MM
mf.FULL = 131 * MM
NAMES = {"fig4_terms_and_cancellation": "pa_fig_terms", "fig3_sign_change_maps": "pa_fig_maps"}
_save = mf.save


def save(fig, name):
    for ax in fig.axes:
        lab = ax.get_title(loc="left")
        if lab in ("(a)", "(b)"):
            ax.set_title(lab.strip("()"), loc="left", fontweight="bold")
    _save(fig, NAMES[name])


mf.save = save
mf.plt.rcParams.update({"xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
mf.fig_terms()
mf.fig_maps()

# ------------------------------------------------------------------ count error vs HOTA, 131 mm layout
I = json.loads((mf.RES / "identity_metrics.json").read_text())
fig, (a, b) = plt.subplots(1, 2, figsize=(mf.FULL, 60 * MM), gridspec_kw=dict(wspace=0.32))
shades = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
marks = ["o", "s", "^", "D", "v"]
for s, col, mk in zip(mf.SIGMAS, shades, marks):
    for mode, ls in (("processed", "-"), ("common", ":")):
        ys = [st.median(I[f"surface_seed{sd}|{v}|{s}|{d}"][mode]["HOTA"] for sd in mf.SEEDS for v in mf.VIDEOS)
              for d in mf.DELTAS]
        a.plot(mf.DELTAS, ys, color=col, ls=ls, marker=mk, ms=3.3, mfc=col if mode == "processed" else "white",
               label=f"σ = {s}" if mode == "processed" else None)
a.set_xscale("log", base=2)
a.set_xticks(mf.DELTAS, [str(d) for d in mf.DELTAS])
a.set_xlabel(r"Processing interval, $\Delta$")
a.set_ylabel("HOTA, median of 21 cases")
a.grid(axis="y", color="#e6e6e6", lw=0.5)
a.set_title("a", loc="left", fontweight="bold")
for sd in mf.SEEDS:
    for v in mf.VIDEOS:
        arms = [(s, d) for s in mf.SIGMAS for d in mf.DELTAS]
        hh = {a_: I[f"surface_seed{sd}|{v}|{a_[0]}|{a_[1]}"]["processed"]["HOTA"] for a_ in arms}
        ee = {a_: abs(mf.e("whole", sd, v, *a_)) for a_ in arms}
        bh = max(arms, key=hh.get)
        be = min(arms, key=ee.get)
        b.annotate("", xy=(ee[be], hh[be]), xytext=(ee[bh], hh[bh]),
                   arrowprops=dict(arrowstyle="->", color="#9e9e9e", lw=0.6))
        b.scatter([ee[bh]], [hh[bh]], s=12, color=mf.BLUE, marker="o", zorder=3, lw=0)
        b.scatter([ee[be]], [hh[be]], s=12, color=mf.VERM, marker="s", zorder=3, lw=0)
b.set_xlabel(r"Absolute count error, $|e|$")
b.set_ylabel("HOTA, processed frames")
b.grid(color="#eeeeee", lw=0.5)
b.set_title("b", loc="left", fontweight="bold")
h, _ = a.get_legend_handles_labels()
h += [Line2D([], [], color=mf.INK, ls="-", label="processed frames"),
      Line2D([], [], color=mf.INK, ls=":", label="all annotated frames"),
      Line2D([], [], color=mf.BLUE, marker="o", ls="none", ms=4, label="highest HOTA"),
      Line2D([], [], color=mf.VERM, marker="s", ls="none", ms=4, label="smallest |e|")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False,
           columnspacing=0.9, handlelength=1.6, handletextpad=0.4)
_save(fig, "pa_fig_hota")
print("written:", sorted(p.name for p in mf.OUT.glob("pa_fig_*.pdf")))
