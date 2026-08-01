#!/usr/bin/env python3
"""Reviewer question 7: Fig. 5 fits phi(tau) but only ever reports tau = 1.

fit_drift already returns every tau, so this reruns the cadence control and keeps
all of them, together with the fit diagnostics (R^2, number of fitted lengths) the
review asks for. Same tracker, same lengths, same thinning as cadence_control.py.
"""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools"))
from oracle_count_surface import (  # noqa: E402
    fit_drift, prefix_surface, read_resolutions, run_sequence, video_frames,
)

DATA = ROOT / "datasets/grapemots_det_721"
CADENCE1 = ["NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3", "PathPlanning_1"]
LENGTHS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300,
           350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
TAUS = [1, 2, 3, 5, 8]
TRACKER = "bytetrack.yaml"


def main() -> None:
    sizes = read_resolutions(DATA)
    out: dict = {}
    for video in CADENCE1:
        frames_all = video_frames(DATA, video)
        out[video] = {}
        for step in (1, 2, 3):
            frames = frames_all[::step]
            rng = np.random.default_rng(0)
            pred, truth = run_sequence(frames, sizes[video], TRACKER, 0.0, 0.0, rng)
            grid = prefix_surface(pred, truth, LENGTHS, TAUS)
            fits = fit_drift(grid, TAUS)
            out[video][step] = {
                "annotated_frames": len(frames),
                "per_tau": {str(t): {
                    "phi_per_annotated_frame": fits[str(t)]["phi_per_frame"],
                    "phi_per_source_frame": fits[str(t)]["phi_per_frame"] / step,
                    "delta": fits[str(t)].get("delta"),
                    "r2": fits[str(t)].get("r2"),
                } for t in TAUS},
            }
            print(f"  {video} step {step} done", flush=True)

    print("\nthinning multiplier on phi per source frame, by tau")
    print(f"{'tau':>4s} {'median x2/x1':>13s} {'range':>16s} {'median x3/x1':>13s}")
    summary = {}
    for tau in TAUS:
        k = str(tau)
        r2 = [out[v][2]["per_tau"][k]["phi_per_source_frame"] /
              out[v][1]["per_tau"][k]["phi_per_source_frame"] for v in CADENCE1]
        r3 = [out[v][3]["per_tau"][k]["phi_per_source_frame"] /
              out[v][1]["per_tau"][k]["phi_per_source_frame"] for v in CADENCE1]
        summary[k] = {"median_step2": float(np.median(r2)),
                      "range_step2": [float(min(r2)), float(max(r2))],
                      "median_step3": float(np.median(r3))}
        print(f"{tau:4d} {np.median(r2):13.2f} {min(r2):7.1f}-{max(r2):<7.1f} "
              f"{np.median(r3):13.2f}")

    r2s = [out[v][1]["per_tau"]["1"]["r2"] for v in CADENCE1]
    print(f"\ntau=1 fit R^2 at step 1: {[round(x,3) for x in r2s]}  "
          f"({len(LENGTHS)} lengths offered)")

    dest = ROOT / "runs/cbdcom2026_queue/results/cadence_control_all_taus.json"
    dest.write_text(json.dumps({"per_video": out, "multipliers": summary}, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
