#!/usr/bin/env python3
"""Rebuild the five figures used by the CBDCom 2026 paper.

All values come from the frozen release results. Figure text follows the IEEE
conference template: Times New Roman at no less than 8 pt, with TrueType fonts
embedded in PDF output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter


def plain_log_axis(axis, ticks: tuple, labels: tuple) -> None:
    """Label a log axis with plain numbers.

    Matplotlib's default $10^{n}$ labels set the exponent at 70 percent of the
    base size, i.e. 5.6 pt here, below the 8 pt floor the IEEE template sets for
    figure text. Plain tick labels keep every glyph at 8 pt and read better on
    an axis whose values the caption quotes directly.
    """
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FixedFormatter(labels))
    axis.set_minor_formatter(NullFormatter())


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "grapemots-protocol" / "results"
OUT = ROOT / "figures"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

RED = "#B2182B"
ORANGE = "#EF8A62"
BLUE = "#2166AC"
LIGHT_BLUE = "#67A9CF"
GREEN = "#1B7837"
PURPLE = "#7B3294"
GREY = "#4D4D4D"
SEQ = ("#1B9E77", "#D95F02", "#7570B3", "#E7298A")


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def configure_style() -> None:
    for name in (
        "Times New Roman.ttf",
        "Times New Roman Bold.ttf",
        "Times New Roman Italic.ttf",
        "Times New Roman Bold Italic.ttf",
    ):
        font_manager.fontManager.addfont(FONT_DIR / name)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": GREY,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "savefig.dpi": 300,
        }
    )


def finish(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_teaser() -> None:
    data = load_json("arm_botsort_tiled.json")
    video = next(item for item in data["videos"] if item["video"] == "PathPlanning_4")
    # Shape and colour both encode tau. Conference proceedings are routinely
    # printed in greyscale, where RED and BLUE sit at a similar lightness, so
    # colour alone would not separate the series.
    colours = {1: RED, 3: ORANGE, 5: LIGHT_BLUE, 8: BLUE}
    markers = {1: "o", 3: "s", 5: "D", 8: "^"}

    fig, ax = plt.subplots(figsize=(3.45, 2.42))
    series = {}
    for tau in (1, 3, 5, 8):
        rows = [row for row in video["count_error_surface"] if row["min_track_len"] == tau]
        xs = [row["window_frames"] for row in rows]
        ys = [row["prefix_signed_relative_error"] for row in rows]
        series[tau] = dict(zip(xs, ys))
        ax.plot(
            xs,
            ys,
            color=colours[tau],
            marker=markers[tau],
            markersize=3.3,
            linewidth=1.2,
            label=rf"$\tau={tau}$",
        )

    ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    plain_log_axis(
        ax.xaxis,
        (10, 20, 30, 50, 75, 150, 374),
        ("10", "20", "30", "50", "75", "150", "374"),
    )
    ax.set_xlim(8.5, 450)
    ax.set_ylim(-0.72, 1.58)
    ax.set_xlabel("Sequence length (annotated frames)")
    ax.set_ylabel("Signed count error")
    ax.grid(True, which="major", color="#E2E2E2", linewidth=0.5)
    ax.legend(loc="upper left", ncol=4, handlelength=1.5, columnspacing=0.9)
    # The two cells the caption quotes. Both satisfy the retained-cell rule
    # tau <= L/2 of Section II-D; (L=10, tau=8) does not, so it is never
    # annotated even though it is the lowest point drawn.
    ax.annotate(
        "+147%",
        xy=(374, series[1][374]),
        xytext=(-32, -15),
        textcoords="offset points",
        color=RED,
        fontsize=8,
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 0.6,
                    "shrinkA": 1, "shrinkB": 3},
    )
    ax.annotate(
        "+2%",
        xy=(75, series[8][75]),
        xytext=(-40, -26),
        textcoords="offset points",
        color=BLUE,
        fontsize=8,
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 0.6,
                    "shrinkA": 1, "shrinkB": 3},
    )
    despine(ax)
    fig.tight_layout(pad=0.35)
    finish(fig, "fig_teaser")


def figure_identity_break() -> None:
    source = mpimg.imread(OUT / "fig_identity_break_source.png")
    panels = (
        source[75:632, 24:1009],
        source[75:632, 1039:1604],
        source[75:632, 1634:2199],
    )
    titles = (
        "(a) Released 4K frame, 26 annotated bunches",
        "(b) First appearance",
        "(c) 192 annotated frames later",
    )

    fig = plt.figure(figsize=(7.0, 2.15))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.75, 1.0, 1.0), wspace=0.05)
    axes = [fig.add_subplot(grid[index]) for index in range(3)]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.6)

    # The yellow rectangle in (a) is the crop window, not a second detection.
    # Unlabelled it reads as an annotation, which is the opposite of its meaning.

    # Opaque replacements cover the original raster labels exactly.
    for ax, x, width, label in (
        (axes[1], 202, 160, "track 12"),
        (axes[2], 191, 182, "track 975"),
    ):
        ax.add_patch(Rectangle((x, 10), width, 44, facecolor=RED, edgecolor="none"))
        ax.text(x + width / 2, 32, label, color="white", fontsize=8, ha="center", va="center")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01)
    finish(fig, "fig_identity_break")


def figure_miss_modes() -> None:
    paper = load_json("paper_numbers.json")
    # Three redundant channels: line style separates the two families, colour
    # and marker separate the modes within a family. In greyscale the two
    # members of a family differ only by marker, so it carries real load.
    modes = (
        ("dropout_bernoulli", "independent frame miss", RED, "o", "-"),
        ("dropout_block", "15-frame block miss", ORANGE, "s", "-"),
        ("dropout_identity", "identity removed", BLUE, "^", "--"),
        ("dropout_size", "small identities removed", LIGHT_BLUE, "D", "--"),
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.67))
    for key, label, colour, marker, line_style in modes:
        rows = paper[key]
        xs = sorted(float(value) for value in rows)
        medians = [rows[str(value)]["median_of_video_medians"] for value in xs]
        per_video = [list(rows[str(value)]["per_video_median"].values()) for value in xs]
        lower = [float(np.percentile(values, 25)) for values in per_video]
        upper = [float(np.percentile(values, 75)) for values in per_video]
        ax.fill_between(xs, lower, upper, color=colour, alpha=0.13, linewidth=0)
        ax.plot(
            xs,
            medians,
            line_style,
            color=colour,
            marker=marker,
            markersize=3.4,
            linewidth=1.2,
            label=label,
        )

    ax.axhline(0, color=GREY, linewidth=0.6, linestyle=":")
    ax.set_xlim(-0.01, 0.42)
    ax.set_ylim(-0.62, 2.55)
    ax.set_xlabel("Miss rate")
    ax.set_ylabel("Signed count error")
    ax.legend(loc="upper left")
    despine(ax)
    fig.tight_layout(pad=0.35)
    finish(fig, "fig_miss_modes")


def figure_metric_alignment() -> None:
    hota = load_json("hota_arms.json")
    decomp = load_json("count_decomposition.json")["arms"]
    # Marker and colour both identify the arm. The Resize and confidence-0.55
    # arms nearly coincide, so they must differ in shape as well as in colour;
    # the two confidence arms share a shape because they are one knob at two
    # values, and they sit far apart vertically.
    arms = (
        ("Resize + BoT-SORT", "Resize", "Resize", GREY, "s", (-7, 13)),
        ("Tiles, conf 0.55", "conf 0.55", "confidence 0.55", BLUE, "v", (7, 2)),
        ("Tiles + ByteTrack", "ByteTrack", "ByteTrack", LIGHT_BLUE, "^", (5, 5)),
        ("Tiles, conf 0.40", "conf 0.40", "confidence 0.40", SEQ[0], "v", (-4, 5)),
        ("Tiles + YOLO11s", "YOLO11s", "YOLO11s", PURPLE, "D", (-5, 5)),
        ("Tiles + IoS merge", "IoS merge", "IoS merge", GREEN, "P", (6, 5)),
        ("Tiles + BoT-SORT", "BoT-SORT", "BoT-SORT", ORANGE, "o", (-5, 5)),
        ("Tiles + ReID", "+ ReID", "+ ReID", RED, "*", (-5, 5)),
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.52))
    for full_name, decomp_name, label, colour, marker, offset in arms:
        assa = hota[full_name]["AssA"]
        pooled = decomp[decomp_name]["pooled"]
        error = (pooled["P"] - pooled["G"]) / pooled["G"]
        ax.scatter(
            assa,
            error,
            s=34,
            color=colour,
            marker=marker,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.annotate(
            label,
            (assa, error),
            textcoords="offset points",
            xytext=offset,
            ha="right" if offset[0] < 0 else "left",
            fontsize=8,
            color="#303030",
        )

    ax.axhline(0, color=GREY, linewidth=0.6, linestyle=":")
    ax.set_xlim(0.198, 0.378)
    ax.set_ylim(-0.16, 2.82)
    ax.set_xlabel("Association accuracy")
    ax.set_ylabel("Signed count error")
    # The eight arms share two videos and most of the pipeline, so they are not
    # exchangeable replicates; the title reports the coefficient descriptively
    # and carries no p value.
    ax.set_title(r"Spearman $\rho=+0.43$ (descriptive, $n=8$ nested arms)")
    despine(ax)
    fig.tight_layout(pad=0.35)
    finish(fig, "fig_metric_alignment")


def figure_cadence() -> None:
    paper = load_json("paper_numbers.json")
    cadence_control = load_json("cadence_control.json")
    phi = paper["phi"]["per_video"]
    cadence = {
        "NoPathPlanning_1": 1,
        "NoPathPlanning_2": 1,
        "NoPathPlanning_3": 1,
        "PathPlanning_1": 1,
        "PathPlanning_2": 2,
        "PathPlanning_3": 2,
        "PathPlanning_4": 2,
        "PathPlanning_5": 2,
        "PathPlanning_6": 2,
        "PathPlanning_7": 2,
        "PathPlanning_8": 2,
    }
    cadence_one = [phi[name]["source"] for name, value in cadence.items() if value == 1]
    cadence_two = [phi[name]["source"] for name, value in cadence.items() if value == 2]
    gap_low = max(cadence_one)
    gap_high = min(cadence_two)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.45, 2.48),
        sharey=True,
        gridspec_kw={"width_ratios": (1.0, 1.35), "wspace": 0.08},
    )
    ax = axes[0]
    offsets = {1: (-0.08, -0.03, 0.03, 0.08), 2: (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)}
    counters = {1: 0, 2: 0}
    for name, group in sorted(cadence.items(), key=lambda item: (item[1], phi[item[0]]["source"])):
        offset = offsets[group][counters[group]]
        counters[group] += 1
        ax.scatter(
            group + offset,
            phi[name]["source"],
            s=22,
            color=BLUE if group == 1 else RED,
            marker="o" if group == 1 else "s",
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax.axhspan(gap_low, gap_high, color="#D9D9D9", alpha=0.65, zorder=0)
    ax.text(1.5, np.sqrt(gap_low * gap_high), "11.4-fold gap", ha="center", va="center")
    ax.set_xticks((1, 2))
    ax.set_xticklabels(("1", "2"))
    ax.set_xlim(0.52, 2.48)
    ax.set_xlabel("Released cadence")
    ax.set_ylabel(r"Fitted drift rate $\varphi$ (per source frame)")
    ax.set_title("Released sequences")

    ax = axes[1]
    names = ("NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3", "PathPlanning_1")
    labels = ("NPP1", "NPP2", "NPP3", "PP1")
    for name, label, hue, marker in zip(names, labels, SEQ, ("o", "s", "^", "D")):
        steps = (1, 2, 3)
        values = [cadence_control[name][str(step)]["phi_per_source_frame"] for step in steps]
        ax.plot(
            steps,
            values,
            marker=marker,
            markersize=3.5,
            linewidth=1.0,
            color=hue,
            alpha=0.9,
            label=label,
        )
    # Without this the four lines are anonymous and the two that fall from
    # interval 2 to 3 look like plotting noise rather than named sequences.
    ax.legend(loc="lower right", ncol=2, handlelength=1.4, columnspacing=0.8,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.2)
    ax.axhspan(min(cadence_two), max(cadence_two), color=RED, alpha=0.11, zorder=0)
    ax.text(1.08, min(cadence_two) * 1.35, "Released cadence-2 range", color=RED, va="bottom")
    ax.set_xticks((1, 2, 3))
    ax.set_xlim(0.85, 3.15)
    ax.set_xlabel("Retained-frame interval")
    ax.set_title("Same footage, thinned")

    for axis in axes:
        axis.set_yscale("log")
        plain_log_axis(
            axis.yaxis,
            (0.005, 0.01, 0.05, 0.1, 0.5),
            ("0.005", "0.01", "0.05", "0.1", "0.5"),
        )
        axis.grid(True, which="major", axis="y", color="#D9D9D9", linewidth=0.5)
        despine(axis)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.87, bottom=0.25, wspace=0.08)
    finish(fig, "fig_cadence")


def main() -> None:
    configure_style()
    figure_teaser()
    figure_identity_break()
    figure_miss_modes()
    figure_metric_alignment()
    figure_cadence()
    print("Rebuilt all five paper figures")


if __name__ == "__main__":
    main()
