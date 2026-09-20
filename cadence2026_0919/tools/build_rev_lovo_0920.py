#!/usr/bin/env python3
"""Training sets for the 2026-09-20 retraining: seeds, and plant-disjoint folds.

Two limitations of the camera-ready paper are experimental, so they are run
rather than stated:

1. Each out-of-fold sequence was read by ONE trained checkpoint (seed 0), so the
   spread between training seeds was not sampled.
2. Out of fold held for the video, not the plant. GrapeMOTS's frontal videos
   (NoPathPlanning_*) film the same vines as the multi-view ones (Ariza-Sentis et
   al., Data in Brief 54, 2024: "These record the same plants"), and every
   original fold trained on frontal videos.

Design, fixed before any run:
  * multi-view folds (test = PathPlanning_2..8): train and validate on
    PathPlanning videos only. The release films each PathPlanning video on a
    different vine, so no plant is shared with the test video. Validation is one
    4K multi-view video, chosen by the same rotation rule as
    tools/make_lovo_splits.py (skip the test video and the 1080p videos).
    Seeds 0, 1, 2.
  * frontal folds (test = NoPathPlanning_1..3): every frontal plant also appears
    in the multi-view videos, and the release gives no cross-view identity, so no
    plant-disjoint training set exists. The original composition is kept and
    seeds 1, 2 are added to the existing seed 0.
Everything else is the original LOVO recipe (tools/run_grapemots_journal_queue_
20260805.sh: tiles 1280, every 3rd annotated frame, 15 epochs, batch 8, one GPU).

Each run gets its own symlinked image/label tree so that Ultralytics' label cache,
which is written next to the images, is never shared between two runs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "tools")
from make_split_manifests import POOL, load_pool, select  # noqa: E402

DST = Path("datasets/rev_lovo_0920")
MULTIVIEW = [f"PathPlanning_{i}" for i in range(1, 9)]
LOW_RES = {"PathPlanning_1", "PathPlanning_3"}
MV_TESTS = [f"PathPlanning_{i}" for i in range(2, 9)]
FRONT_TESTS = [f"NoPathPlanning_{i}" for i in range(1, 4)]
ALL_ORDER = ["NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3"] + MULTIVIEW


def mv_spec(test: str) -> dict:
    index = ALL_ORDER.index(test)
    candidates = [v for v in MULTIVIEW if v != test and v not in LOW_RES]
    val = candidates[index % len(candidates)]
    train = [v for v in MULTIVIEW if v not in (test, val)]
    return {"train": sorted(train), "val": [val], "test": [test], "design": "plant-disjoint"}


def front_spec(test: str) -> dict:
    spec = json.loads(Path(f"splits/lovo_{test}.json").read_text())
    spec["design"] = "original composition (no plant-disjoint set exists)"
    return spec


def link_tree(run_dir: Path, split: str, tiles: list[str]) -> list[str]:
    img_dir = run_dir / "images" / split
    lab_dir = run_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for t in tiles:
        src = Path(t)
        lab = Path(str(src).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lab.exists():
            raise SystemExit(f"label missing for {src}")
        di = img_dir / src.name
        dl = lab_dir / (src.stem + ".txt")
        if not di.exists():
            os.symlink(src.resolve(), di)
        if not dl.exists():
            os.symlink(lab.resolve(), dl)
        out.append(str(di.absolute()))  # the symlink path, so labels resolve inside this run
    return out


def main() -> None:
    by_video = load_pool(POOL)
    runs = []
    for test in MV_TESTS:
        runs += [(test, mv_spec(test), s) for s in (0, 1, 2)]
    for test in FRONT_TESTS:
        runs += [(test, front_spec(test), s) for s in (1, 2)]
    index = []
    for test, spec, seed in runs:
        name = f"{test}_s{seed}"
        run_dir = DST / name
        run_dir.mkdir(parents=True, exist_ok=True)
        tr = link_tree(run_dir, "train", select(by_video, spec["train"], 3))
        va = link_tree(run_dir, "val", select(by_video, spec["val"], 3))
        (run_dir / "train.txt").write_text("\n".join(tr) + "\n")
        (run_dir / "val.txt").write_text("\n".join(va) + "\n")
        (run_dir / "data.yaml").write_text(
            f"path: {run_dir.absolute()}\ntrain: train.txt\nval: val.txt\nnc: 1\nnames: ['grape']\n")
        (run_dir / "split_spec.json").write_text(json.dumps({**spec, "seed": seed,
                                                             "train_tiles": len(tr), "val_tiles": len(va)}, indent=2))
        index.append({"run": name, "test": test, "seed": seed, "design": spec["design"],
                      "train": spec["train"], "val": spec["val"], "train_tiles": len(tr), "val_tiles": len(va)})
        print(f"{name:22s} {spec['design'][:14]:14s} train {len(tr):5d} tiles ({','.join(v.replace('PathPlanning_', 'PP').replace('NoPP', 'NP') for v in spec['train'])})  val {spec['val']}")
    (DST / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
