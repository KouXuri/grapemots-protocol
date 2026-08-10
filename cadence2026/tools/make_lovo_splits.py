#!/usr/bin/env python3
"""Leave-one-video-out split specifications for GrapeMOTS.

The manuscript's five bespoke assignments (A--E) leave six of the eleven videos
model-unseen and repeat four of them, so the real-pipeline evidence rests on
1,738 of the 5,755 annotated frames and has to be described as exploratory. A
plain leave-one-video-out design removes that limitation: every video is held
out exactly once, every video is scored by a checkpoint that never saw it, and
the real-pipeline cohort becomes the whole release.

What is given up is the fixed 7:2:1 frame ratio. That ratio was never reachable
by choice anyway -- the three frontal videos are 890--927 frames each, about 15%
of the corpus, so video granularity, not sloppiness, sets what is possible. Under
leave-one-video-out the test share is simply whatever the held-out video is, and
it is reported per video rather than pretended to be constant.

Validation still needs two videos, and which two is a free choice that must not
become a hidden degree of freedom. The rule here is fixed in advance and applied
mechanically: rotate through the videos of each acquisition mode in the released
order, take the first candidate of each mode that is not the test video, so every
fold validates on exactly one frontal and one multi-view video. No fold is
allowed to validate on the video it tests.

The multi-view rotation draws only from the six 4K sequences. PathPlanning_1 and
PathPlanning_3 are 1920x1080, and AP is never pooled across resolutions, so a
validation set whose resolution mix changed from fold to fold would make
checkpoint selection mean something different in each fold. Those two videos stay
in training everywhere except in the two folds that hold them out for test, and
those two folds are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Released order. Frontal controls first so the rotation is easy to check by eye.
FRONTAL = ["NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3"]
MULTIVIEW = ["PathPlanning_1", "PathPlanning_2", "PathPlanning_3", "PathPlanning_4",
             "PathPlanning_5", "PathPlanning_6", "PathPlanning_7", "PathPlanning_8"]
ALL = FRONTAL + MULTIVIEW
LOW_RESOLUTION = {"PathPlanning_1", "PathPlanning_3"}


def validation_for(test: str, index: int) -> list[str]:
    """One frontal plus one 4K multi-view video, rotated, never the test video."""
    frontal = [name for name in FRONTAL if name != test]
    multiview = [name for name in MULTIVIEW
                 if name != test and name not in LOW_RESOLUTION]
    if not frontal or not multiview:
        raise SystemExit(f"fold {test}: no candidate left for validation")
    return [frontal[index % len(frontal)], multiview[index % len(multiview)]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", type=Path, default=Path("splits"))
    ap.add_argument("--prefix", default="lovo")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, test in enumerate(ALL):
        val = validation_for(test, index)
        train = [name for name in ALL if name != test and name not in val]
        spec = {"train": sorted(train), "val": sorted(val), "test": [test]}

        assigned = spec["train"] + spec["val"] + spec["test"]
        if len(assigned) != len(set(assigned)) or len(assigned) != len(ALL):
            raise SystemExit(f"fold {test}: assignment is not a partition")

        path = args.dst / f"{args.prefix}_{test}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        summary.append({
            "fold": test,
            "spec": str(path),
            "val": spec["val"],
            "train_size": len(train),
            "test_is_1080p": test in LOW_RESOLUTION,
            "val_has_1080p": bool(set(val) & LOW_RESOLUTION),
        })
        flag = "  [1080p test]" if test in LOW_RESOLUTION else ""
        print(f"{test:<18} val={val[0]:<18}{val[1]:<18} train={len(train)}{flag}")

    index_path = args.dst / f"{args.prefix}_index.json"
    index_path.write_text(json.dumps({
        "design": "leave-one-video-out, 11 folds",
        "validation_rule": "rotate released order per acquisition mode, skip the test video",
        "note": ("PathPlanning_1 and PathPlanning_3 are 1920x1080; under this design they are "
                 "held out like any other video. AP is therefore never pooled across "
                 "resolutions -- report those two folds separately."),
        "folds": summary,
    }, indent=2) + "\n")
    print(f"\nwrote {len(summary)} fold specs and {index_path}")
    low = [entry["fold"] for entry in summary if entry["test_is_1080p"]]
    print(f"folds whose test video is 1920x1080 (report separately): {low}")


if __name__ == "__main__":
    main()
