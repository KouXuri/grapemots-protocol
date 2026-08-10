#!/usr/bin/env python3
"""Thin a corpus's annotation in time, to measure what temporal density buys.

The Bodegas release labels one frame in 12 to 75, and an IoU-gated associator
cannot work at that spacing. GrapeMOTS labels every first or second source
frame, and the same associator works. Those are two points, on two different
corpora, confounded with everything else that differs between them.

This builds the axis properly: take one corpus and remove annotated frames,
keeping the imagery, the objects, the flight and the annotator fixed, so the
only thing that changes is how densely the sequence is labelled. Running the
same oracle sweep at each density gives the exchange rate between annotation
budget and what the annotation can support -- which is the question anyone
planning a capture actually faces, and which cannot be answered by comparing two
finished datasets.

Frames are renumbered densely at every density, because oracle_master.py treats
consecutive sidecars as consecutive tracker updates. That is the intended
semantics here: at density k the tracker genuinely sees only the retained frames,
exactly as it would if the release had been built that way.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

FRAME_RE = re.compile(r"__frame_(\d+)\.txt$")


def read_manifest(root: Path) -> dict[str, dict]:
    with (root / "manifest.csv").open() as handle:
        return {row["video"]: row for row in csv.DictReader(handle)}


def video_sidecars(root: Path, video: str) -> list[Path]:
    found: list[Path] = []
    for split in ("train", "val", "test"):
        directory = root / "tracks" / split
        if directory.is_dir():
            found.extend(directory.glob(f"{video}__frame_*.txt"))
    # the pooled 'all' directory duplicates the split ones; keep one copy
    unique: dict[int, Path] = {}
    for path in found:
        unique.setdefault(int(FRAME_RE.search(path.name).group(1)), path)
    return [unique[key] for key in sorted(unique)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("datasets/grapemots_det_721"))
    ap.add_argument("--dst", type=Path, default=Path("datasets/density_ladder"))
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--min-frames", type=int, default=4,
                    help="a video with fewer retained frames than this is dropped at "
                         "that density rather than left as a degenerate one-frame case")
    args = ap.parse_args()

    manifest = read_manifest(args.root)
    videos = args.videos or sorted(manifest)

    for factor in args.factors:
        out = args.dst / f"k{factor:02d}"
        if out.exists():
            shutil.rmtree(out)
        (out / "tracks" / "train").mkdir(parents=True, exist_ok=True)
        rows = []
        for video in videos:
            sidecars = video_sidecars(args.root, video)
            kept = sidecars[::factor]
            if len(kept) < args.min_frames:
                print(f"  k={factor} {video}: only {len(kept)} frames retained, dropped")
                continue
            instances = 0
            tracks: set[str] = set()
            for index, path in enumerate(kept):
                text = path.read_text()
                for line in text.splitlines():
                    parts = line.split()
                    if len(parts) == 6:
                        instances += 1
                        tracks.add(parts[1])
                (out / "tracks" / "train" / f"{video}__frame_{index:06d}.txt").write_text(text)

            source = manifest[video]
            # Cadence is per SOURCE frame, so the interval this density represents
            # is the release's own cadence multiplied by the thinning factor.
            base_cadence = int(source.get("cadence", 1) or 1) if "cadence" in source else 1
            rows.append({
                "video": video, "split": "train", "mode": source.get("mode", "unknown"),
                "frames": len(kept), "instances": instances, "gt_tracks": len(tracks),
                "resolution": source["resolution"],
                "thinning_factor": factor,
                "source_frame_interval": base_cadence * factor,
            })
        fields = ["video", "split", "mode", "frames", "instances", "gt_tracks",
                  "resolution", "thinning_factor", "source_frame_interval"]
        with (out / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        total_frames = sum(row["frames"] for row in rows)
        total_tracks = sum(row["gt_tracks"] for row in rows)
        print(f"k={factor:2d}: {len(rows):2d} videos, {total_frames:5d} annotated frames, "
              f"{total_tracks:4d} trajectories -> {out}")


if __name__ == "__main__":
    main()
