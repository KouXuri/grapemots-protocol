#!/usr/bin/env python3
"""Figure: sequence observation structure explains the counting drift rate.

Three panels per column, one column per representative video:
  top    - each GT identity's horizontal centroid against annotated-frame order.
           Frontal passes show parallel diagonal streaks (bunches sweep across
           the frame once and leave); planned multi-view passes show long flat
           bands that fold back on themselves (the drone orbits and the same
           bunches stay in view).
  bottom - the same identities as (x, y) paths.

The header of each column carries the three numbers that matter for counting:
turnover (= trajectories / mean simultaneously visible), median lifetime, and
phi, the fitted drift slope of P-G against window length. phi varies by two
orders of magnitude across these sequences and tracks turnover almost exactly
(r = -0.90), which is the point of the figure: the amount by which unique-track
counting over-counts is a property of how the sequence was flown, not only of
the tracker.

Lifetimes are printed in SOURCE frames: PathPlanning videos other than _1 are
annotated every second frame, so annotated-frame lifetimes are not comparable
across columns without this conversion.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# videos chosen to span the turnover range, not cherry-picked for looks
VIDEOS = [
    ("NoPathPlanning_1", "frontal pass"),
    ("PathPlanning_5", "planned multi-view"),
    ("PathPlanning_8", "planned multi-view"),
]
# Turnover and median lifetime come from the ground-truth audit. phi is read
# from the frozen number file rather than pasted here: an earlier version of this
# figure carried annotated-frame slopes while the text had already moved to
# source-frame units, so the caption contradicted the abstract.
STATS = {  # turnover, median lifetime in source frames
    "NoPathPlanning_1": (5.92, 158),
    "PathPlanning_5": (2.66, 184),
    "PathPlanning_8": (1.90, 354),
}
PHI = json.loads(
    Path("runs/cbdcom2026_queue/results/paper_numbers.json").read_text()
)["phi"]["per_video"]
MAX_FRAMES = 150  # keep columns visually comparable


def load(track_dir: Path, video: str):
    """{track_id: [(frame_order, xc, yc), ...]} for one video."""
    per_frame = defaultdict(list)
    for path in track_dir.rglob(f"{video}__frame_*.txt"):
        frame_no = int(path.stem.split("__frame_")[1].split("__")[0])
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) >= 6:
                per_frame[frame_no].append((int(p[1]), float(p[2]), float(p[3])))
    order = sorted(per_frame)[:MAX_FRAMES]
    tracks = defaultdict(list)
    for idx, frame_no in enumerate(order):
        for tid, xc, yc in per_frame[frame_no]:
            tracks[tid].append((idx, xc, yc))
    return tracks, len(order)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/grapemots_det_721/tracks")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures/fig_sequence_structure.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.linewidth": 0.6, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), squeeze=False)

    for col, (video, mode) in enumerate(VIDEOS):
        tracks, n_frames = load(root, video)
        turnover, med_life = STATS[video]
        phi = PHI[video]["source"]
        cmap = plt.get_cmap("tab20")

        ax = axes[0][col]
        for i, (tid, pts) in enumerate(sorted(tracks.items())):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, lw=0.5, alpha=0.75, color=cmap(i % 20))
        ax.set_title(f"{video}\n{mode}\n"
                     f"turnover {turnover:.2f},  median life {med_life} src fr\n" r"$\varphi=$" f"{phi:.4f}/source frame",
                     fontsize=7, linespacing=1.4)
        ax.set_xlim(0, n_frames)
        ax.set_ylim(0, 1)
        ax.set_xlabel("annotated-frame order")
        if col == 0:
            ax.set_ylabel("centroid x (normalised)")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {out} and {pdf}")


if __name__ == "__main__":
    main()
