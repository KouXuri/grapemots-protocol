#!/usr/bin/env python3
"""The two figures the argument actually needs.

Figure 1 -- annotation density. One corpus, one detector, one tracker; only how
densely the sequence is labelled changes. It carries the paper's central claim in
one panel pair: the reported count error walks to zero and past it while the
fraction of annotated trajectories the system reaches falls the whole way. The
U/D/M panel shows why, which is the part a single accuracy number cannot say.

Figure 2 -- the cadence contrast on a published release. Same footage, same
checkpoint, same scoring instants, processing cadence changed. This is the
intervention, so it is drawn as paired points rather than as two distributions.

Style follows the manuscript's existing figures: serif, no chartjunk, colours
that survive greyscale printing.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
INK = "#1a1a1a"
BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#999999"

ROOT = pathlib.Path("/home/kou/my_env/yolo26")
OUT = ROOT / "figures" / "journal_0809"
OUT.mkdir(parents=True, exist_ok=True)


def figure_density():
    """Real pipeline against annotation density, with the decomposition beneath."""
    summary = json.loads(
        (ROOT / "runs/final_analyses_0809/results/density_realpipeline.json").read_text())
    pooled = summary["pooled_tau1"]
    order = sorted(pooled, key=lambda k: int(k[1:]))
    ks = np.array([int(k[1:]) for k in order])
    err = np.array([pooled[k]["signed_error"] for k in order])
    assigned = np.array([pooled[k]["assigned_fraction"] for k in order])
    U = np.array([pooled[k]["U"] for k in order])
    D = np.array([pooled[k]["D"] for k in order])
    M = np.array([pooled[k]["M"] for k in order])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.12})

    top.axhline(0, color=GREY, lw=0.6, zorder=1)
    top.plot(ks, err, "o-", color=RED, lw=1.4, ms=4, label="signed count error $e$", zorder=3)
    top.set_ylabel("signed count error $e$", color=RED)
    top.tick_params(axis="y", colors=RED)
    twin = top.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(ks, assigned, "s--", color=BLUE, lw=1.4, ms=3.6,
              label="assigned fraction", zorder=3)
    twin.set_ylabel("assigned fraction $1-M/G$", color=BLUE)
    twin.tick_params(axis="y", colors=BLUE)
    twin.set_ylim(0, 1)

    # The point of the figure: the flattering number and the worst coverage coincide.
    best = int(np.argmin(np.abs(err)))
    top.annotate(f"$e={err[best]:+.2f}$\nreaches {assigned[best]:.0%}",
                 xy=(ks[best], err[best]), xytext=(ks[best] * 1.05, err[best] + 0.95),
                 fontsize=7.5, color=INK,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))

    bottom.plot(ks, U, "o-", color=INK, lw=1.2, ms=3.4, label="$U$ ownerless")
    bottom.plot(ks, D, "^-", color=RED, lw=1.2, ms=3.4, label="$D$ duplicate")
    bottom.plot(ks, M, "s-", color=BLUE, lw=1.2, ms=3.4, label="$M$ unassigned")
    bottom.set_xscale("log", base=2)
    bottom.set_xticks(ks)
    bottom.set_xticklabels([str(k) for k in ks])
    bottom.set_xlabel(r"annotation thinning factor $k$   (1 frame kept in $k$)")
    bottom.set_ylabel("tracks")
    bottom.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.savefig(OUT / "fig_density.pdf")
    fig.savefig(OUT / "fig_density.png")
    plt.close(fig)
    print(f"wrote {OUT/'fig_density.pdf'}")
    return {"k": ks.tolist(), "e": err.tolist(), "assigned": assigned.tolist()}


def figure_cadence():
    """Released annotation cadence against source rate, paired by sequence."""
    path = ROOT / "runs/bodegas_round2_0809/results/cadence_contrast_release.json"
    if not path.is_file():
        # The whole-release contrast is still running; fall back to the six
        # sequences that are already complete rather than drawing nothing.
        path = ROOT / "runs/grapemots_journal_phase2/results/bodegas_track_botsort.json"
        print(f"whole-release contrast not ready; figure 2 deferred")
        return None
    report = json.loads(path.read_text())
    rows = [r for r in report["sequences"] if r["model_unseen"]]
    if not rows:
        rows = report["sequences"]

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhline(0, color=GREY, lw=0.6)
    for index, row in enumerate(sorted(rows, key=lambda r: r["released"])):
        ax.plot([index, index], [row["released"], row["source_rate"]],
                color=GREY, lw=0.8, zorder=1)
        ax.plot(index, row["released"], "o", color=BLUE, ms=4, zorder=3)
        ax.plot(index, row["source_rate"], "^", color=RED, ms=4.5, zorder=3)
    ax.plot([], [], "o", color=BLUE, ms=4, label="released cadence")
    ax.plot([], [], "^", color=RED, ms=4.5, label="every source frame")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["video"] for r in sorted(rows, key=lambda x: x["released"])],
                       rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("signed count error $e$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.savefig(OUT / "fig_cadence.pdf")
    fig.savefig(OUT / "fig_cadence.png")
    plt.close(fig)
    print(f"wrote {OUT/'fig_cadence.pdf'}")
    return report["groups"]


if __name__ == "__main__":
    a = figure_density()
    b = figure_cadence()
    (OUT / "figure_values.json").write_text(
        json.dumps({"density": a, "cadence": b}, indent=1) + "\n")
    print(f"wrote {OUT/'figure_values.json'}")
