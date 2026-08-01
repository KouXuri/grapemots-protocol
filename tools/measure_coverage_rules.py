#!/usr/bin/env python3
"""Reviewer question 3/4: is M really "uncovered", or only "unassigned"?

M counts trajectories that own no predicted track under the dominant-overlap
rule. A trajectory that a predicted track touched, but did not touch most, still
lands in M. This reports, per arm and per video:

  touched     trajectories matched by any predicted track in any frame
  owned       trajectories that own a predicted track (G - M)
  M           G - owned, i.e. what the paper currently calls uncovered

so the manuscript can say how much of M is genuine non-detection and how much is
an artefact of the ownership rule.
"""
from __future__ import annotations

import os
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools"))
from decompose_count_error import frame_matches  # noqa: E402

RESULTS = ROOT / "runs/cbdcom2026_queue/results"
ARMS = {
    "conf 0.55": "arm_conf0.55.json",
    "Resize": "arm_resize.json",
    "conf 0.40": "arm_conf0.40.json",
    "IoS merge": "arm_ios_tiled.json",
    "BoT-SORT": "arm_botsort_tiled.json",
    "YOLO11s": "arm_yolo11s_tiled.json",
    "ByteTrack": "arm_bytetrack_tiled.json",
    "+ ReID": "arm_reid.json",
}


def coverage(video, threshold=0.5, tau=1):
    pred_by_frame = video["frame_predicted_ids"]
    gt_by_frame = video["frame_gt_ids"]
    pred_life = Counter(t for frame in pred_by_frame for t in frame)
    kept = {t for t, n in pred_life.items() if n >= tau}

    overlap: dict[int, Counter] = defaultdict(Counter)
    for pb, pi, gb, gi in zip(video["frame_predicted_boxes"], pred_by_frame,
                              video["frame_gt_boxes"], gt_by_frame):
        live = [(b, t) for b, t in zip(pb, pi) if t in kept]
        if not live:
            continue
        for pred, truth in frame_matches([b for b, _ in live], [t for _, t in live],
                                         gb, gi, threshold):
            overlap[pred][truth] += 1

    all_gt = sorted({t for frame in gt_by_frame for t in frame})
    owner = {p: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
             for p, c in overlap.items() if c}
    owned = set(owner.values())
    touched = {truth for c in overlap.values() for truth in c}
    G = len(all_gt)
    return {
        "video": video["video"], "G": G,
        "owned": len(owned), "M": G - len(owned),
        "touched": len(touched), "never_touched": G - len(touched),
        "owned_recall": len(owned) / G if G else None,
        "any_overlap_recall": len(touched) / G if G else None,
    }


def main() -> None:
    out = {}
    print(f"{'arm':11s} {'G':>4s} {'M(unassigned)':>14s} {'never touched':>14s} "
          f"{'owned rec':>10s} {'any-overlap rec':>16s}")
    for label, filename in ARMS.items():
        data = json.loads((RESULTS / filename).read_text())
        rows = [coverage(v) for v in data["videos"]]
        G = sum(r["G"] for r in rows)
        M = sum(r["M"] for r in rows)
        never = sum(r["never_touched"] for r in rows)
        out[label] = {"videos": rows, "pooled": {
            "G": G, "M": M, "never_touched": never,
            "owned_recall": (G - M) / G, "any_overlap_recall": (G - never) / G}}
        print(f"{label:11s} {G:4d} {M:14d} {never:14d} "
              f"{(G-M)/G:10.3f} {(G-never)/G:16.3f}")
    dest = RESULTS / "coverage_rules.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
