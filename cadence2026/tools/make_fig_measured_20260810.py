#!/usr/bin/env python3
"""Figure 3: the measured cadence of a release, and the geometry it implies.

(a) Effective annotation rate of the 28 aligned sequences, defined as the source
    frame rate divided by the median source-frame gap between consecutive
    annotated frames, against the 1 Hz floor.
(b) Measured median overlap of consecutive same-identity reference boxes against
    the measured displacement relative to target size, for all 51 sequences of
    the four corpora, with the closed form (1-r)/(1+r) for two equal boxes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.8,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

ROOT = Path("/home/kou/my_env/yolo26")
OUT = ROOT / "figures" / "journal_0809"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_FPS = 59.94

align = json.loads((ROOT / "runs/bodegas_full_audit_0809/results"
                    / "bodegas_alignment_all28.json").read_text())
aligned = [v for v in align["sequences"].values() if v.get("status") == "aligned"]
rate = np.sort(np.array([SOURCE_FPS / v["source_gap_median"] for v in aligned]))

structure = json.loads((ROOT / "runs/grapemots_journal_0805/results"
                        / "sequence_structure.json").read_text())["sequences"]

LABELS = {"bodegas2023": "vineyard 2023", "grapemots": "GrapeMOTS",
          "mot17": "MOT17", "mot20": "MOT20"}
STYLE = {"bodegas2023": ("#c0392b", "o"), "grapemots": ("#1f77b4", "s"),
         "mot17": ("#7f7f7f", "^"), "mot20": ("#bcbd22", "v")}

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 4.3))

ax.bar(np.arange(rate.size), rate, color="#c0392b", width=0.75, linewidth=0)
ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
ax.text(0.4, 1.12, "1 Hz floor", fontsize=7, va="bottom")
ax.set_xlabel("the 28 aligned sequences, sorted")
ax.set_ylabel("annotation rate (Hz)")
ax.set_xlim(-0.8, rate.size - 0.2)
ax.set_xticks([])
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"median {np.median(rate):.2f} Hz, "
             f"{int((rate < 1).sum())} of {rate.size} below 1 Hz",
             fontsize=8, pad=4)

grid = np.linspace(0.02, 1.0, 200)
bx.plot(grid, (1 - grid) / (1 + grid), color="black", linewidth=1.1,
        label=r"$(1-r)/(1+r)$", zorder=1)
bx.plot([1.0, 20.0], [0.0, 0.0], color="black", linewidth=1.1, zorder=1)
for corpus in ("mot17", "mot20", "grapemots", "bodegas2023"):
    rows = [s for s in structure if s.get("corpus") == corpus]
    colour, marker = STYLE[corpus]
    bx.scatter([s["step_over_size_median"] for s in rows],
               [s["consecutive_iou_median"] for s in rows],
               s=16, c=colour, marker=marker, edgecolors="white",
               linewidths=0.4, label=f"{LABELS[corpus]} ({len(rows)})", zorder=2)
bx.set_xscale("log")
bx.set_xlabel(r"displacement / target size, $r$")
bx.set_ylabel("consecutive reference IoU")
bx.set_ylim(-0.05, 1.0)
bx.spines[["top", "right"]].set_visible(False)
bx.legend(frameon=False, loc="upper right", handletextpad=0.4, borderpad=0.2)

fig.tight_layout(h_pad=1.4)
fig.savefig(OUT / "fig_measured.pdf")
fig.savefig(OUT / "fig_measured.png")

summary = {
    "annotation_rate_hz": {"median": float(np.median(rate)),
                           "min": float(rate.min()), "max": float(rate.max()),
                           "below_1hz": int((rate < 1).sum()),
                           "sequences": int(rate.size)},
    "per_corpus": {c: {"sequences": len([s for s in structure if s.get("corpus") == c]),
                       "r_median": float(np.median([s["step_over_size_median"]
                                                    for s in structure
                                                    if s.get("corpus") == c])),
                       "iou_median": float(np.median([s["consecutive_iou_median"]
                                                      for s in structure
                                                      if s.get("corpus") == c]))}
                   for c in LABELS},
}
(OUT / "fig_measured_values.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
print(f"wrote {OUT/'fig_measured.pdf'}")
