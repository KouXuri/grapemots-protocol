#!/usr/bin/env python3
"""Create the GrapeMOTS acquisition/annotation and tiling method figures.

The figures use a real RGB-mask pair from PathPlanning_5/frame_000132. The
acquisition panel is explicitly schematic; it communicates the two collection
modes without claiming an unreleased UAV trajectory. Instance-mask colours are
visualisation colours for track IDs, whereas detection boxes use one colour
because this paper retains one semantic class (grape).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperstyle import apply as apply_paper_style  # noqa: E402


INK = "#17232D"
MUTED = "#60717D"
LINE = "#C7D0D5"
PAPER = "#FFFFFF"
GREEN = "#2F7D64"
GREEN_LIGHT = "#DCEDE5"
ORANGE = "#E8752E"
ORANGE_LIGHT = "#FCE5D6"
RED = "#D92D4A"
BLUE = "#277DA1"
YELLOW = "#F4C542"
MASK_COLOURS = (
    "#00A6A6",
    "#E76F51",
    "#577590",
    "#F2B134",
    "#7B61A8",
    "#43AA8B",
    "#D1495B",
    "#4D908E",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("tmp/grapemots_figure_assets/pathplanning5_image_frame_000132.PNG"),
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=Path("tmp/grapemots_figure_assets/pathplanning5_mask_frame_000132.png"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path))
    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f"RGB/mask shape mismatch: {rgb.shape[:2]} vs {mask.shape[:2]}")
    if mask.dtype != np.uint16:
        raise ValueError(f"Expected a 16-bit MOTS mask, got {mask.dtype}")
    return rgb, mask


def grape_instances(mask: np.ndarray) -> list[dict[str, int]]:
    instances: list[dict[str, int]] = []
    for raw_value in np.unique(mask):
        value = int(raw_value)
        if value == 0 or value // 1000 != 1:
            continue
        ys, xs = np.where(mask == value)
        if xs.size < 20:
            continue
        instances.append(
            {
                "value": value,
                "track": value % 1000,
                "area": int(xs.size),
                "x0": int(xs.min()),
                "y0": int(ys.min()),
                "x1": int(xs.max()),
                "y1": int(ys.max()),
            }
        )
    return instances


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label} {title}", loc="left", fontsize=7.6, fontweight="bold", pad=3, color=INK)


def style_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(LINE)
        spine.set_linewidth(0.7)


def draw_drone(ax: plt.Axes, x: float, y: float, scale: float, colour: str, alpha: float = 1.0,
               yscale: float = 1.0) -> None:
    """Draw a compact UAV symbol in axis coordinates.

    `yscale` compensates the vertical squash of a wide, short panel, whose axis
    box is far from square: without it the rotors collapse onto the arms.
    """
    sy = scale * yscale
    lw = 1.25 * scale
    ax.plot([x - 0.055 * scale, x + 0.055 * scale], [y, y], color=colour, lw=lw, alpha=alpha)
    ax.plot([x - 0.04 * scale, x + 0.04 * scale], [y - 0.026 * sy, y + 0.026 * sy],
            color=colour, lw=lw, alpha=alpha)
    ax.add_patch(Rectangle((x - 0.018 * scale, y - 0.012 * sy), 0.036 * scale, 0.024 * sy,
                           facecolor=colour, edgecolor="none", alpha=alpha))
    for dx, dy in ((-0.055, 0), (0.055, 0), (-0.04, -0.026), (0.04, 0.026)):
        ax.add_patch(Ellipse((x + dx * scale, y + dy * sy), 0.024 * scale, 0.024 * sy,
                             facecolor=PAPER, edgecolor=colour, lw=0.8, alpha=alpha))
    ax.add_patch(Ellipse((x, y - 0.019 * sy), 0.016 * scale, 0.016 * sy,
                         facecolor=ORANGE, edgecolor=PAPER, lw=0.35, alpha=alpha))


def draw_vine_row(ax: plt.Axes, x0: float, x1: float, y: float, canopy_h: float = 0.16) -> None:
    span = x1 - x0
    ax.plot([x0 + 0.06 * span, x1 - 0.06 * span], [y - 0.09, y - 0.09], color=INK, lw=0.9)
    for x in np.linspace(x0 + 0.13 * span, x1 - 0.13 * span, 5):
        ax.plot([x, x], [y - 0.20, y - 0.015], color="#6B5037", lw=1.1)
        ax.add_patch(Ellipse((x, y), 0.17 * span, canopy_h, facecolor=GREEN_LIGHT,
                             edgecolor=GREEN, lw=0.65))
        ax.add_patch(Ellipse((x - 0.032 * span, y - 0.045), 0.045 * span, 0.045,
                             facecolor=ORANGE_LIGHT, edgecolor=ORANGE, lw=0.55))


def fov_polygon(camera: tuple[float, float], target_left: tuple[float, float],
                target_right: tuple[float, float], colour: str) -> Polygon:
    return Polygon([camera, target_left, target_right], closed=True, facecolor=colour,
                   edgecolor="none", alpha=0.22)


def draw_acquisition_mode(ax: plt.Axes, x0: float, x1: float, name: str, descriptor: str,
                          colour: str, cameras: tuple[float, ...], motion: str,
                          arrow_style: str) -> None:
    """Draw one collection mode inside the horizontal band [x0, x1]."""
    span = x1 - x0
    ax.add_patch(Rectangle((x0, 0.02), span, 0.88, facecolor="#F7F9FA", edgecolor=LINE, lw=0.6))
    ax.text(x0 + 0.035 * span, 0.870, name, fontsize=5.8, fontweight="bold", color=INK, va="top")
    ax.text(x0 + 0.035 * span, 0.755, descriptor, fontsize=5.1, color=MUTED, va="top")
    draw_vine_row(ax, x0, x1, 0.545, canopy_h=0.15)

    # Every camera looks at the same central vine, which is what makes the
    # multi-view mode multi-view: one bunch, several viewpoints, several chances
    # for the tracker to issue it a fresh identity.
    target = x0 + 0.5 * span
    for offset in cameras:
        camera = (x0 + offset * span, 0.235)
        active = abs(offset - 0.5) < 1e-6
        ax.add_patch(fov_polygon(camera, (target - 0.06 * span, 0.47),
                                 (target + 0.06 * span, 0.47), colour))
        draw_drone(ax, *camera, 0.88 * span, colour if active else MUTED,
                   1.0 if active else 0.60, yscale=2.6)

    ax.add_patch(FancyArrowPatch((x0 + 0.16 * span, 0.115), (x1 - 0.16 * span, 0.115),
                                 arrowstyle=arrow_style, mutation_scale=6, lw=0.8, color=colour))
    ax.text(x0 + 0.5 * span, 0.095, motion, fontsize=5.0, color=colour, ha="center", va="top")


def draw_acquisition_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "(a)", "Acquisition modes (schematic)")
    draw_acquisition_mode(ax, 0.005, 0.492, "NoPathPlanning", "frontal control", BLUE,
                          (0.5,), "along-row motion", "-|>")
    draw_acquisition_mode(ax, 0.508, 0.995, "PathPlanning", "planned multi-view", ORANGE,
                          (0.26, 0.5, 0.74), "viewpoint variation", "<->")


def visible_instances(instances: list[dict[str, int]], crop: tuple[int, int, int, int]) -> list[dict[str, int]]:
    x0, y0, x1, y1 = crop
    return [item for item in instances if item["x1"] >= x0 and item["x0"] < x1
            and item["y1"] >= y0 and item["y0"] < y1]


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025,
                facecolor=PAPER)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025,
                facecolor=PAPER)
    plt.close(fig)


def make_acquisition_annotation(rgb: np.ndarray, mask: np.ndarray,
                                instances: list[dict[str, int]], out_dir: Path) -> None:
    crop = (1500, 1080, 2260, 1480)
    x0, y0, x1, y1 = crop
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    shown = [
        item
        for item in visible_instances(instances, crop)
        if item["x0"] >= x0 and item["x1"] < x1 and item["y0"] >= y0 and item["y1"] < y1
    ]

    fig = plt.figure(figsize=(7.16, 1.22), facecolor=PAPER)
    grid = fig.add_gridspec(1, 4, width_ratios=(1.55, 1.0, 1.0, 1.0), wspace=0.085)
    acquisition = fig.add_subplot(grid[0, 0])
    raw_ax = fig.add_subplot(grid[0, 1])
    mask_ax = fig.add_subplot(grid[0, 2])
    box_ax = fig.add_subplot(grid[0, 3])
    draw_acquisition_schematic(acquisition)

    raw_ax.imshow(crop_rgb)
    panel_label(raw_ax, "(b)", "RGB crop")
    raw_ax.text(0.025, 0.035, "PathPlanning_5 / frame_000132", transform=raw_ax.transAxes,
                fontsize=5.2, color=PAPER, ha="left", va="bottom",
                bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.78, "pad": 1.4})
    style_image_axis(raw_ax)

    mask_ax.imshow(crop_rgb)
    overlay = np.zeros((*crop_mask.shape, 4), dtype=float)
    for item in shown:
        local = crop_mask == item["value"]
        colour = matplotlib.colors.to_rgba(MASK_COLOURS[item["track"] % len(MASK_COLOURS)], 0.64)
        overlay[local] = colour
        cx = (max(item["x0"], x0) + min(item["x1"], x1 - 1)) / 2 - x0
        cy = (max(item["y0"], y0) + min(item["y1"], y1 - 1)) / 2 - y0
        mask_ax.text(cx, cy, str(item["track"]), color=PAPER, fontsize=5.1, ha="center", va="center",
                     fontweight="bold", bbox={"facecolor": INK, "edgecolor": "none", "pad": 0.65, "alpha": 0.78})
    mask_ax.imshow(overlay)
    panel_label(mask_ax, "(c)", "MOTS instances")
    mask_ax.text(0.025, 0.035, "value = 1000 x class + track ID", transform=mask_ax.transAxes,
                 fontsize=5.1, color=PAPER, ha="left", va="bottom",
                 bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.82, "pad": 1.4})
    style_image_axis(mask_ax)

    box_ax.imshow(crop_rgb)
    for item in shown:
        bx0 = max(item["x0"], x0) - x0
        by0 = max(item["y0"], y0) - y0
        bx1 = min(item["x1"], x1 - 1) - x0
        by1 = min(item["y1"], y1 - 1) - y0
        box_ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                                   edgecolor=RED, lw=1.0))
    # Track 2 is the instance in this crop closest to the frame median width, so the
    # printed box can be checked against the size quoted in the corpora section.
    focus = next(item for item in shown if item["track"] == 2)
    info = (f"v={focus['value']} -> class=1, track={focus['track']}\n"
            f"box=[{focus['x0']},{focus['y0']},{focus['x1']},{focus['y1']}], "
            f"{focus['x1'] - focus['x0']} x {focus['y1'] - focus['y0']} px")
    box_ax.text(0.025, 0.035, info, transform=box_ax.transAxes, fontsize=5.0, color=PAPER,
                ha="left", va="bottom",
                bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.84, "pad": 1.4})
    panel_label(box_ax, "(d)", "One-class boxes")
    style_image_axis(box_ax)

    fig.subplots_adjust(left=0.008, right=0.995, top=0.91, bottom=0.02)
    save_figure(fig, out_dir / "fig_acquisition_annotation")


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def make_tiling_pipeline(rgb: np.ndarray, instances: list[dict[str, int]], out_dir: Path) -> None:
    height, width = rgb.shape[:2]
    tile, stride = 1280, 960
    x_starts = tile_starts(width, tile, stride)
    y_starts = tile_starts(height, tile, stride)
    focus = next(item for item in instances if item["track"] == 3)
    zoom = (1780, 1110, 2150, 1420)
    zx0, zy0, zx1, zy1 = zoom
    zoom_rgb = rgb[zy0:zy1, zx0:zx1]

    fig = plt.figure(figsize=(7.16, 2.85), facecolor=PAPER)
    grid = fig.add_gridspec(2, 4, height_ratios=(4.2, 1.0), width_ratios=(1.42, 1.0, 1.0, 1.18),
                            hspace=0.20, wspace=0.12)
    full_ax = fig.add_subplot(grid[0, 0])
    tile_ax = fig.add_subplot(grid[0, 1])
    restore_ax = fig.add_subplot(grid[0, 2])
    merge_ax = fig.add_subplot(grid[0, 3])
    lane_ax = fig.add_subplot(grid[1, :])

    full_ax.imshow(rgb)
    tile_colours = (BLUE, ORANGE)
    tile_index = 1
    for row, y_start in enumerate(y_starts):
        for col, x_start in enumerate(x_starts):
            colour = tile_colours[(row + col) % 2]
            full_ax.add_patch(Rectangle((x_start, y_start), tile, tile, fill=False,
                                        edgecolor=colour, lw=1.0))
            full_ax.text(x_start + 32, y_start + 72, f"T{tile_index}", fontsize=5.0, color=PAPER,
                         fontweight="bold", bbox={"facecolor": colour, "edgecolor": "none", "pad": 0.8})
            tile_index += 1
    full_ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                                edgecolor=YELLOW, lw=1.4))
    panel_label(full_ax, "(a)", "4K frame -> 8 overlapping tiles")
    full_ax.text(0.02, 0.04, "tile 1280 x 1280 | stride 960\nx: 0, 960, 1920, 2560 | y: 0, 880",
                 transform=full_ax.transAxes, fontsize=5.0, color=PAPER, va="bottom",
                 bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.84, "pad": 1.5})
    style_image_axis(full_ax)

    tile_ax.imshow(zoom_rgb)
    full_box = (focus["x0"] - zx0, focus["y0"] - zy0,
                focus["x1"] - focus["x0"], focus["y1"] - focus["y0"])
    clipped_x0 = max(focus["x0"], 1920)
    clipped_box = (clipped_x0 - zx0, focus["y0"] - zy0,
                   focus["x1"] - clipped_x0, focus["y1"] - focus["y0"])
    tile_ax.add_patch(Rectangle((full_box[0], full_box[1]), full_box[2], full_box[3], fill=False,
                                edgecolor=BLUE, lw=1.5, label="T6: full instance"))
    tile_ax.add_patch(Rectangle((clipped_box[0], clipped_box[1]), clipped_box[2], clipped_box[3], fill=False,
                                edgecolor=ORANGE, lw=1.5, linestyle="--", label="T7: clipped instance"))
    tile_ax.axvline(1920 - zx0, color=PAPER, lw=0.9, linestyle=":")
    tile_ax.text(1920 - zx0 + 4, 10, "T7 starts", fontsize=4.9, color=INK, va="top",
                 bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.86, "pad": 0.8})
    panel_label(tile_ax, "(b)", "Tile-level boxes")
    tile_ax.legend(loc="lower left", fontsize=4.8, framealpha=0.84, borderpad=0.25,
                   handlelength=1.3, labelspacing=0.25)
    style_image_axis(tile_ax)

    restore_ax.imshow(zoom_rgb)
    restore_ax.add_patch(Rectangle((full_box[0], full_box[1]), full_box[2], full_box[3], fill=False,
                                   edgecolor=BLUE, lw=1.5))
    restore_ax.add_patch(Rectangle((clipped_box[0], clipped_box[1]), clipped_box[2], clipped_box[3], fill=False,
                                   edgecolor=ORANGE, lw=1.5, linestyle="--"))
    panel_label(restore_ax, "(c)", "Restore coordinates")
    restore_ax.text(0.03, 0.04, "x_frame = x_tile + x0\ny_frame = y_tile + y0",
                    transform=restore_ax.transAxes, fontsize=5.2, color=PAPER, va="bottom",
                    bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.84, "pad": 1.5})
    style_image_axis(restore_ax)

    merge_ax.imshow(zoom_rgb)
    merge_ax.add_patch(Rectangle((full_box[0], full_box[1]), full_box[2], full_box[3], fill=False,
                                 edgecolor=GREEN, lw=1.8))
    merge_ax.text(full_box[0] + full_box[2] / 2, full_box[1] - 7, "one retained box",
                  fontsize=4.9, color=PAPER, ha="center", va="bottom",
                  bbox={"facecolor": GREEN, "edgecolor": "none", "pad": 0.8})
    panel_label(merge_ax, "(d)", "Merge and evaluate")
    merge_ax.text(0.03, 0.04, "overlap NMS / IoS merge\nthen full-frame COCO evaluation",
                  transform=merge_ax.transAxes, fontsize=5.1, color=PAPER, va="bottom",
                  bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.84, "pad": 1.5})
    style_image_axis(merge_ax)

    lane_ax.set_xlim(0, 1)
    lane_ax.set_ylim(0, 1)
    lane_ax.axis("off")
    lane_ax.add_patch(Rectangle((0.0, 0.53), 1.0, 0.39, facecolor=GREEN_LIGHT, edgecolor="none"))
    lane_ax.add_patch(Rectangle((0.0, 0.06), 1.0, 0.39, facecolor=ORANGE_LIGHT, edgecolor="none"))
    lane_ax.text(0.018, 0.725, "TRAIN", fontsize=5.8, fontweight="bold", color=GREEN, va="center")
    lane_ax.text(0.105, 0.725, "MOTS masks", fontsize=5.6, color=INK, va="center")
    lane_ax.text(0.275, 0.725, "clipped tile labels", fontsize=5.6, color=INK, va="center")
    lane_ax.text(0.505, 0.725, "YOLO training", fontsize=5.6, color=INK, va="center")
    lane_ax.text(0.018, 0.255, "TEST", fontsize=5.8, fontweight="bold", color=ORANGE, va="center")
    lane_ax.text(0.105, 0.255, "tile detections", fontsize=5.6, color=INK, va="center")
    lane_ax.text(0.305, 0.255, "coordinate restore", fontsize=5.6, color=INK, va="center")
    lane_ax.text(0.535, 0.255, "overlap NMS/IoS", fontsize=5.6, color=INK, va="center")
    lane_ax.text(0.735, 0.255, "full-frame COCO AP", fontsize=5.6, color=INK, va="center")
    for y, starts in ((0.725, (0.225, 0.445)), (0.255, (0.255, 0.485, 0.685))):
        for x in starts:
            lane_ax.add_patch(FancyArrowPatch((x, y), (x + 0.035, y), arrowstyle="-|>",
                                              mutation_scale=6, lw=0.7, color=MUTED))

    fig.subplots_adjust(left=0.008, right=0.995, top=0.92, bottom=0.025)
    save_figure(fig, out_dir / "fig_tiling_pipeline")


def main() -> None:
    # Serif to match the plotted figures, and fonttype 42: matplotlib's default
    # Type 3 fonts are rejected by IEEE PDF eXpress. The grid these axes would
    # otherwise inherit would be drawn over the imagery.
    apply_paper_style()
    plt.rcParams.update({"axes.grid": False})
    args = parse_args()
    rgb, mask = load_pair(args.image, args.mask)
    instances = grape_instances(mask)
    if len(instances) != 29:
        raise ValueError(f"Expected 29 grape instances in frame_000132, found {len(instances)}")
    make_acquisition_annotation(rgb, mask, instances, args.out_dir)
    make_tiling_pipeline(rgb, instances, args.out_dir)
    print(f"Wrote {args.out_dir / 'fig_acquisition_annotation.png'} and PDF")
    print(f"Wrote {args.out_dir / 'fig_tiling_pipeline.png'} and PDF")


if __name__ == "__main__":
    main()
