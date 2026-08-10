#!/usr/bin/env python3
"""Build a Bodegas box-detection dataset on Piazolo et al.'s own train/test split.

Why this dataset has to exist. The Bodegas evidence so far is oracle-only: it
shows that the released annotation cadence leaves consecutive boxes of one
trajectory non-overlapping, so an IoU-gated associator is working blind. That is
a statement about the reference. The published headline it explains -- Piazolo
et al. (CEA 2026) raising counting accuracy from 33% to 96% -- was produced by a
real detector and a real tracker, so answering it needs a real pipeline too.

Two decisions keep the comparison honest:

  The split is theirs, not ours. Their Table 2 lists 23 training rows and 6 test
  rows; those exact lists are hard-coded below. Validation is carved out of THEIR
  training rows, never out of their test rows, so their test sequences stay
  untouched by checkpoint selection.

  Nothing is re-tiled. The 1280/960 tiles already exist with polygon labels from
  the segmentation work; only the label geometry changes, from polygon to its
  axis-aligned bounding box, and the images are reached through symlinks. Tiling
  is per-frame and independent of the split, so a different split costs three
  text files rather than an hour of cropping.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Piazolo et al., CEA 245 (2026) 111529, Table 2.
PIAZOLO_TEST = ["row_4.2_1", "row_6.1_1", "row_6.1_2", "row_7.1_1", "row_7.1_2", "row_8_1"]
PIAZOLO_TRAIN = [
    "row_4.3_2", "row_4.4_2", "row_4.4_4", "row_6.1_3", "row_6.1_4", "row_6.2_1",
    "row_6.2_2", "row_6.3", "row_7.1_3", "row_7.1_4", "row_7.2_1", "row_7.2_2",
    "row_7.2_3", "row_7.2_4", "row_7.3_1", "row_7.3_2", "row_7.3_3", "row_7.3_4",
    "row_7.4_1", "row_7.4_2", "row_8_2", "row_8_3", "row_8_4",
]
# Held out of training for checkpoint selection. Taken from their training rows.
VALIDATION = ["row_7.4_1", "row_7.4_2", "row_8_2"]

TILE_RE = re.compile(r"^(?P<seq>row_[0-9._]+?)_(?P<frame>\d{6})_x(?P<x>\d+)_y(?P<y>\d+)$")


def polygon_to_box(parts: list[float]) -> tuple[float, float, float, float] | None:
    """Normalised polygon -> normalised xc yc w h, clipped to the tile."""
    xs = [min(max(v, 0.0), 1.0) for v in parts[0::2]]
    ys = [min(max(v, 0.0), 1.0) for v in parts[1::2]]
    if len(xs) < 3 or len(ys) < 3:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=Path,
                    default=Path("datasets/bodegas_grape_bunch_seg_tiles1280_s960"))
    ap.add_argument("--dst", type=Path, default=Path("datasets/bodegas_det_piazolo"))
    ap.add_argument("--min-box-frac", type=float, default=0.0,
                    help="drop boxes narrower or shorter than this fraction of the tile")
    ap.add_argument("--split-spec", type=Path,
                    help="JSON with train/val/test sequence lists; defaults to the "
                         "split Piazolo et al. published")
    args = ap.parse_args()

    images_out = args.dst / "images" / "all"
    labels_out = args.dst / "labels" / "all"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    by_sequence: dict[str, list[str]] = defaultdict(list)
    tiles = polygons = boxes = dropped = 0
    unparsed: list[str] = []

    for split in ("train", "val", "test"):
        for image in sorted((args.tiles / "images" / split).glob("*.png")):
            match = TILE_RE.match(image.stem)
            if not match:
                unparsed.append(image.name)
                continue
            label = args.tiles / "labels" / split / f"{image.stem}.txt"
            rows = []
            if label.is_file():
                for line in label.read_text().splitlines():
                    values = line.split()
                    if len(values) < 7:
                        continue
                    polygons += 1
                    box = polygon_to_box([float(v) for v in values[1:]])
                    if box is None or box[2] < args.min_box_frac or box[3] < args.min_box_frac:
                        dropped += 1
                        continue
                    rows.append(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                    boxes += 1

            link = images_out / image.name
            if not link.exists():
                link.symlink_to(image.resolve())
            (labels_out / f"{image.stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""))
            by_sequence[match.group("seq")].append(str(link.resolve()))
            tiles += 1

    if unparsed:
        raise SystemExit(f"{len(unparsed)} tile names did not parse, e.g. {unparsed[:3]}")

    known = set(PIAZOLO_TRAIN) | set(PIAZOLO_TEST)
    missing = known - set(by_sequence)
    extra = set(by_sequence) - known
    if missing:
        raise SystemExit(f"sequences named by Piazolo et al. have no tiles: {sorted(missing)}")
    if extra:
        print(f"note: {len(extra)} sequence(s) present but not in their split: {sorted(extra)}")

    if args.split_spec:
        assignment = json.loads(args.split_spec.read_text())
        source = str(args.split_spec)
    else:
        assignment = {"train": [s for s in PIAZOLO_TRAIN if s not in VALIDATION],
                      "val": VALIDATION, "test": PIAZOLO_TEST}
        source = "Piazolo et al., Comput. Electron. Agric. 245 (2026) 111529, Table 2"
    missing_split = {v for group in assignment.values() for v in group} - set(by_sequence)
    if missing_split:
        raise SystemExit(f"split names sequences with no tiles: {sorted(missing_split)}")
    counts = {}
    for split, sequences in assignment.items():
        paths = [path for seq in sequences for path in sorted(by_sequence[seq])]
        (args.dst / f"{split}.txt").write_text("\n".join(paths) + "\n")
        counts[split] = len(paths)

    (args.dst / "bodegas_det.yaml").write_text(
        f"path: {args.dst.resolve()}\n"
        "train: train.txt\nval: val.txt\ntest: test.txt\n"
        "nc: 1\nnames: ['grape']\n"
    )
    (args.dst / "split_spec.json").write_text(json.dumps({
        "source": source,
        "assignment": assignment,
        "validation_taken_from": "their training rows, so their test rows stay unopened",
        "tiles": counts,
        "tile_size": 1280, "tile_stride": 960,
    }, indent=2) + "\n")

    print(f"tiles {tiles}, polygons {polygons} -> boxes {boxes} (dropped {dropped})")
    print(f"train {counts['train']} / val {counts['val']} / test {counts['test']} tiles")
    print(f"  steps/epoch at batch 8: {-(-counts['train'] // 8)}")
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
