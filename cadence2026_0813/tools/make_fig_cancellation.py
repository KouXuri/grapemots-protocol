#!/usr/bin/env python3
"""Where a count error reaches zero, and how much of the reference is left there.

The paper's claim in one frame. Three knobs move a counting pipeline: how often
frames reach the tracker, how densely the corpus is annotated, and where the
detector's operating point sits. Plotted against the share of reference
trajectories the pipeline actually reaches, all three run the same way and all
three cross zero error while under half the trajectories have been found. A
reported error of zero is therefore not evidence of a correct count, which is the
argument the rest of the paper makes term by term.

Every point is read from a frozen result file, and the crossings are linear
interpolations between the two measured points that bracket them, written out
beside the figure so the annotation can be checked rather than trusted.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from paperstyle import apply, C_ERR, C_GT, C_PRED, C_NEUTRAL  # noqa: E402

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "figures"
LADDER = ROOT / "runs/final_analyses_0809/results/density_realpipeline.json"
PANEL_A = ROOT / "grapemots-protocol/cbdcom2026_r3/results/hota_panelA.json"
ARMS = [ROOT / "runs/adaptive_0813/results/arms_fold1_six.json",
        ROOT / "runs/adaptive_0813/results/arms_fold2_eleven.json"]


def cadence_series() -> list[tuple[float, float, str]]:
    """The processing-cadence path: the sparse arm, three budgets, the source rate."""
    pooled: dict[str, Counter] = {}
    for path in ARMS:
        payload = json.loads(path.read_text())
        for record in payload["runs"]:
            one = record["decomposition"]["1"]
            terms = pooled.setdefault(record["arm"], Counter())
            for key in ("P", "G", "M"):
                terms[key] += one[key]
    order = [("rel", "released"), ("uni2", r"$2\times$"), ("uni4", r"$4\times$"),
             ("uni8", r"$8\times$"), ("src", "source")]
    series = []
    for arm, label in order:
        terms = pooled[arm]
        G = terms["G"]
        series.append((1 - terms["M"] / G, (terms["P"] - G) / G, label))
    return series


def ladder_series() -> list[tuple[float, float, str]]:
    """The annotation-thinning path, densest first."""
    pooled = json.loads(LADDER.read_text())["pooled_tau1"]
    series = []
    for key in sorted(pooled, key=lambda s: int(s[1:]), reverse=True):
        cell = pooled[key]
        series.append((cell["assigned_fraction"], cell["signed_error"],
                       f"$k={key[1:]}$"))
    return series


def confidence_series() -> list[tuple[float, float, str]]:
    """The detector operating point, over one cache of detections."""
    rows = json.loads(PANEL_A.read_text())["rows"]
    wanted = [("Confidence 0.85", "0.85"), ("Confidence 0.70", "0.70"),
              ("Confidence 0.55", "0.55"), ("Confidence 0.40", "0.40"),
              ("BoT-SORT, buffer 30", "0.25")]
    return [(rows[key]["assigned_fraction"], rows[key]["signed_error"], label)
            for key, label in wanted]


def crossing(series) -> float | None:
    """Where a series changes sign, interpolated between the bracketing points."""
    ordered = sorted(series)
    for (x0, y0, _), (x1, y1, _) in zip(ordered, ordered[1:]):
        if (y0 <= 0 <= y1) or (y1 <= 0 <= y0):
            if y1 == y0:
                return x0
            return x0 + (x1 - x0) * (0 - y0) / (y1 - y0)
    return None


def main() -> None:
    cadence, ladder, confidence = cadence_series(), ladder_series(), confidence_series()
    crossings = {"processing cadence": crossing(cadence),
                 "annotation thinning": crossing(ladder),
                 "detector confidence": crossing(confidence)}
    low, high = min(crossings.values()), max(crossings.values())

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.axhspan(-0.05, 0.05, color=C_NEUTRAL, alpha=0.10, lw=0)
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.6, zorder=1)
    ax.axvspan(low, high, color=C_NEUTRAL, alpha=0.13, lw=0, zorder=0)

    for series, colour, marker, style, label in (
        (cadence, C_ERR, "o", "-", "processing cadence, 2023"),
        (ladder, C_GT, "s", "--", "annotation thinning, 2024"),
        (confidence, C_PRED, "^", ":", "detector confidence, 2024"),
    ):
        xs = [point[0] for point in series]
        ys = [point[1] for point in series]
        ax.plot(xs, ys, style, color=colour, marker=marker, markersize=3.2,
                lw=1.1, label=label, zorder=3, markeredgewidth=0)

    for name, value in crossings.items():
        ax.plot([value], [0.0], marker="o", markersize=5.2, markerfacecolor="white",
                markeredgecolor=C_NEUTRAL, markeredgewidth=0.9, zorder=4)

    ax.annotate(f"zero error at {low:.2f}\u2013{high:.2f}\nof the reference reached",
                xy=(high, -0.30), xytext=(0.55, -0.78),
                textcoords="data", ha="left", va="center", fontsize=6.6,
                color=C_NEUTRAL,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_NEUTRAL,
                                shrinkA=2, shrinkB=2))

    ax.set_xlabel(r"annotated trajectories reached, $1-M/G$")
    ax.set_ylabel(r"signed count error $e$")
    ax.set_xlim(0.03, 0.86)
    ax.set_ylim(-1.05, 2.9)
    ax.legend(loc="upper left", frameon=False, handlelength=2.0,
              borderaxespad=0.3, labelspacing=0.3)
    fig.tight_layout(pad=0.15)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_cancellation.pdf")
    (OUT / "fig_cancellation_data.json").write_text(json.dumps({
        "crossings": crossings,
        "series": {"processing cadence": cadence, "annotation thinning": ladder,
                   "detector confidence": confidence},
    }, indent=1) + "\n")
    print("crossings:", {k: round(v, 4) for k, v in crossings.items()})
    print("wrote", OUT / "fig_cancellation.pdf")


if __name__ == "__main__":
    main()
