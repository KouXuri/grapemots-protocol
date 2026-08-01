#!/usr/bin/env python3
"""Is the 11.4-fold drift gap between cadence groups caused by the cadence itself?

The four sequences annotated at every source frame all have phi below 0.017; the
seven annotated at every second frame all have phi above 0.18. Those are different
videos, so the gap could be structure rather than sampling. This decides it on the
same footage: take only the cadence-1 sequences and thin their annotated frames by
a factor s, so the tracker sees the same vineyard with s times the inter-frame
displacement. phi is always converted to a per-source-frame rate, exactly as the
paper does for the released cadence-2 videos, so the numbers stay comparable.

No model inference: ground-truth boxes go straight to the tracker.
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools"))
from oracle_count_surface import (video_frames, read_resolutions,  # noqa: E402
                                  prefix_surface, fit_drift, run_sequence)

DATA = ROOT / "datasets/grapemots_det_721"
CADENCE1 = ["NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3", "PathPlanning_1"]
LENGTHS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300,
           350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
TAUS = [1, 2, 3, 5, 8]
TRACKER = "bytetrack.yaml"          # the arm the paper's oracle numbers come from


def main():
    sizes = read_resolutions(DATA)
    out = {}
    print(f"{'video':20s} {'step':>4s} {'ann.frames':>10s} {'phi/ann':>9s} {'phi/src':>9s}")
    for video in CADENCE1:
        frames_all = video_frames(DATA, video)
        out[video] = {}
        for step in (1, 2, 3):
            frames = frames_all[::step]
            rng = np.random.default_rng(0)
            pred, truth = run_sequence(frames, sizes[video], TRACKER, 0.0, 0.0, rng)
            grid = prefix_surface(pred, truth, LENGTHS, TAUS)
            fits = fit_drift(grid, TAUS)
            phi_ann = fits["1"]["phi_per_frame"]
            phi_src = phi_ann / step      # one annotated step now spans `step` source frames
            out[video][step] = {"annotated_frames": len(frames),
                                "phi_per_annotated_frame": phi_ann,
                                "phi_per_source_frame": phi_src,
                                "r2": fits["1"]["r2"]}
            print(f"{video:20s} {step:4d} {len(frames):10d} {phi_ann:9.4f} {phi_src:9.4f}",
                  flush=True)
    # the headline: how much does phi/source frame move when only the sampling changes
    ratios = [out[v][2]["phi_per_source_frame"] / out[v][1]["phi_per_source_frame"]
              for v in CADENCE1]
    ratios3 = [out[v][3]["phi_per_source_frame"] / out[v][1]["phi_per_source_frame"]
               for v in CADENCE1]
    out["_summary"] = {
        "step2_over_step1": {v: round(r, 2) for v, r in zip(CADENCE1, ratios)},
        "step2_median": float(np.median(ratios)),
        "step3_over_step1": {v: round(r, 2) for v, r in zip(CADENCE1, ratios3)},
        "step3_median": float(np.median(ratios3)),
        "released_between_group_gap": 11.4,
    }
    print("\nphi per SOURCE frame, step2 / step1:",
          {v: round(r, 2) for v, r in zip(CADENCE1, ratios)},
          f"median {np.median(ratios):.2f}x")
    print("phi per SOURCE frame, step3 / step1:",
          {v: round(r, 2) for v, r in zip(CADENCE1, ratios3)},
          f"median {np.median(ratios3):.2f}x")
    dest = ROOT / "runs/cbdcom2026_queue/results/cadence_control.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
