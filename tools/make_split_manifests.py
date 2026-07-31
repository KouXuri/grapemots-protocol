#!/usr/bin/env python3
"""Build video-level split manifests over the existing tile pool.

The tiles under datasets/grapemots_det_tiles1280_step1_721/images/{train,val,test}
cover all 11 videos.  Which directory a tile sits in only records the 721 split
that was active when the tiles were cut; Ultralytics reads the manifest and
derives the label path by swapping /images/ for /labels/, so a manifest is free
to place any video in any split without moving a single file.

That makes an alternative video-level split cost nothing but three text files,
which is what the split-sensitivity study needs.

Usage:
  python tools/make_split_manifests.py --spec splits/split_B.json \
      --dst datasets/grapemots_split_B --step 3
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

POOL = Path("datasets/grapemots_det_tiles1280_step1_721")
TILE_RE = re.compile(r"^(?P<video>[A-Za-z]+_\d+)__frame_(?P<frame>\d+)__")


def load_pool(pool: Path) -> dict[str, dict[int, list[str]]]:
    """video -> frame number -> tile paths, read from the pool's three manifests."""
    by_video: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name in ("train", "val", "test"):
        manifest = pool / f"{name}.txt"
        if not manifest.exists():
            raise SystemExit(f"pool manifest missing: {manifest}")
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            m = TILE_RE.match(Path(line).name)
            if not m:
                raise SystemExit(f"unparsable tile name: {line}")
            by_video[m["video"]][int(m["frame"])].append(line)
    return by_video


def select(by_video, videos: list[str], step: int) -> list[str]:
    """Every `step`-th annotated frame of each video, all of its tiles."""
    out: list[str] = []
    for video in videos:
        if video not in by_video:
            raise SystemExit(f"video not in pool: {video}")
        frames = sorted(by_video[video])
        for frame in frames[::step]:
            out.extend(sorted(by_video[video][frame]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True,
                    help="JSON with train/val/test video lists")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--step", type=int, default=3,
                    help="temporal subsampling for train/val")
    ap.add_argument("--test-step", type=int, default=1,
                    help="test is never subsampled: tracking needs consecutive frames")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="videos to drop entirely, e.g. the two 1080p ones")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    if args.exclude:
        dropped = [v for s in spec for v in spec[s] if v in args.exclude]
        if any(v in spec["val"] or v in spec["test"] for v in args.exclude):
            raise SystemExit(f"refusing to drop an evaluation video: {dropped}")
        spec = {s: [v for v in spec[s] if v not in args.exclude] for s in spec}
        print(f"excluded from train: {dropped}")
    by_video = load_pool(args.pool)

    assigned = [v for s in ("train", "val", "test") for v in spec[s]]
    if len(assigned) != len(set(assigned)):
        raise SystemExit("a video appears in more than one split")
    missing = set(by_video) - set(assigned)
    if missing:
        print(f"note: {len(missing)} video(s) unused: {sorted(missing)}")

    args.dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        step = args.test_step if split == "test" else args.step
        tiles = select(by_video, spec[split], step)
        (args.dst / f"{split}.txt").write_text("\n".join(tiles) + "\n")
        counts[split] = len(tiles)

    (args.dst / "grapemots_tiles.yaml").write_text(
        f"path: {args.dst.resolve()}\n"
        "train: train.txt\nval: val.txt\ntest: test.txt\n"
        "nc: 1\nnames: ['grape']\n"
    )
    (args.dst / "split_spec.json").write_text(
        json.dumps({**spec, "step": args.step, "test_step": args.test_step,
                    "tiles": counts}, indent=2) + "\n"
    )

    steps_per_epoch = -(-counts["train"] // 8)  # batch 8
    print(f"{args.dst.name}: train {counts['train']} / val {counts['val']} / "
          f"test {counts['test']} tiles   ({steps_per_epoch} steps/epoch at batch 8)")


if __name__ == "__main__":
    main()
