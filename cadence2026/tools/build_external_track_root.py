#!/usr/bin/env python3
"""Convert an external tracking dataset into the GrapeMOTS track-sidecar layout.

The point is not convenience. The protocol claims in the manuscript are produced
by tools/oracle_master.py, and a reviewer is entitled to ask whether a
replication on another dataset used the same analysis or a re-implementation of
it. If the external data is written into the layout oracle_master.py already
reads, the replication runs the identical code path and the only thing that
changed is the sequences.

Two sources are supported:

  motchallenge  MOT17 / MOT20 style: <src>/<seq>/gt/gt.txt plus seqinfo.ini.
                Only rows the benchmark actually scores are kept (class 1,
                conf 1). MOT17 ships each sequence three times, once per public
                detector, with byte-identical ground truth; the duplicates are
                collapsed so a sequence is not counted three times.

  mots          Bodegas Terras Gauda style: <src>/<seq>/instances/*.png, 16-bit,
                pixel value = class * 1000 + track. Same encoding and the same
                20-pixel minimum component area as the GrapeMOTS converter, so
                the reference trajectories are built by the same rule.

  csv           A previously extracted trajectory table with the columns
                sequence, frame_index, gt_track_id, x1, y1, x2, y2. This is how
                the Bodegas 2023 trajectories are reached in practice: the raw
                release directory on this host holds only zero-byte placeholders,
                while datasets/bodegas_grape_bunch_seg/gt_tracks_from_mots.csv
                still carries the boxes that were decoded from the MOTS masks by
                the same class*1000+track rule.

Output layout, matching datasets/grapemots_det_721:

  <dst>/tracks/train/<seq>__frame_<NNNNNN>.txt   "cls track_key xc yc w h", normalised
  <dst>/manifest.csv                             video,split,mode,frames,instances,gt_tracks,resolution

Frame indices are renumbered to a dense 0..N-1 sequence in source order. That
matters: oracle_master.py sorts sidecars by the number in the filename and then
treats consecutive files as consecutive tracker updates, so a gap in the source
numbering would otherwise silently become a gap in the motion model's notion of
time.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MOT_SUFFIX = re.compile(r"-(DPM|FRCNN|SDP)$")
# MOTChallenge gt.txt: frame,id,left,top,width,height,conf,class,visibility
MOT_SCORED_CLASS = 1


def norm_line(cls: int, key: int, x1: float, y1: float, x2: float, y2: float,
              width: int, height: int) -> str:
    xc = (x1 + x2) / 2 / width
    yc = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {key} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def read_motchallenge(seq_dir: Path, min_visibility: float):
    """frame -> [(track_id, x1, y1, x2, y2)], plus (width, height, frame_rate)."""
    info = configparser.ConfigParser()
    info.read(seq_dir / "seqinfo.ini")
    width = int(info["Sequence"]["imWidth"])
    height = int(info["Sequence"]["imHeight"])
    frame_rate = int(float(info["Sequence"]["frameRate"]))

    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with (seq_dir / "gt" / "gt.txt").open() as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            frame, track = int(row[0]), int(row[1])
            left, top, bw, bh = (float(value) for value in row[2:6])
            conf = float(row[6]) if len(row) > 6 else 1.0
            klass = int(float(row[7])) if len(row) > 7 else MOT_SCORED_CLASS
            visibility = float(row[8]) if len(row) > 8 else 1.0
            if conf < 1 or klass != MOT_SCORED_CLASS or visibility < min_visibility:
                continue
            if bw <= 0 or bh <= 0:
                continue
            x1 = max(0.0, left)
            y1 = max(0.0, top)
            x2 = min(float(width), left + bw)
            y2 = min(float(height), top + bh)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            by_frame[frame].append((track, x1, y1, x2, y2))
    return by_frame, width, height, frame_rate


def read_mots(seq_dir: Path, min_area: int, keep_class: int):
    """Same MOTS decoding rule as tools/create_grapemots_detection_dataset.py."""
    import cv2

    masks = sorted((seq_dir / "instances").glob("*.png"))
    if not masks:
        raise SystemExit(f"no instance masks under {seq_dir}")
    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    width = height = 0
    for index, path in enumerate(masks):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise SystemExit(f"could not read {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        height, width = mask.shape[:2]
        rows: list[tuple[int, float, float, float, float]] = []
        for value in np.unique(mask):
            if value == 0:
                continue
            class_id = int(value) // 1000
            track_id = int(value) % 1000
            if class_id != keep_class:
                continue
            ys, xs = np.nonzero(mask == value)
            if xs.size < min_area:
                continue
            rows.append((class_id * 1000 + track_id,
                         float(xs.min()), float(ys.min()),
                         float(xs.max()) + 1.0, float(ys.max()) + 1.0))
        by_frame[index] = rows
    return by_frame, width, height


def read_trajectory_csv(path: Path, width: int, height: int):
    """sequence -> ({frame_index: [(track, x1, y1, x2, y2)]}, {frame_index: image_path})."""
    per_sequence: dict[str, dict[int, list[tuple[int, float, float, float, float]]]] = \
        defaultdict(lambda: defaultdict(list))
    images: dict[str, dict[int, str]] = defaultdict(dict)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            sequence, index = row["sequence"], int(row["frame_index"])
            if row.get("image_path"):
                images[sequence][index] = row["image_path"]
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            per_sequence[sequence][index].append(
                (int(row["gt_track_id"]),
                 max(0.0, x1), max(0.0, y1),
                 min(float(width), x2), min(float(height), y2))
            )
    return per_sequence, images


def collapse_mot17_duplicates(seq_dirs: list[Path]) -> list[Path]:
    """MOT17 ships MOT17-02-DPM/-FRCNN/-SDP with identical gt.txt."""
    chosen: dict[str, Path] = {}
    for path in sorted(seq_dirs):
        base = MOT_SUFFIX.sub("", path.name)
        # deterministic pick, and FRCNN is the conventional single-copy choice
        if base not in chosen or path.name.endswith("-FRCNN"):
            chosen[base] = path
    return [chosen[key] for key in sorted(chosen)]


def emit_sequence(dst: Path, name: str, by_frame, width: int, height: int,
                  frame_rate: int, mode: str) -> dict:
    """Write one sequence's sidecars and return its manifest row.

    Frames are renumbered densely. Where the source numbering has holes -- a
    missing annotation, not a missing object -- the hole is reported rather than
    interpolated, which is the convention the GrapeMOTS ledger already uses for
    its seventeen missing masks.
    """
    track_dir = dst / "tracks" / "train"
    track_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(by_frame)
    gaps = sum(1 for a, b in zip(ordered, ordered[1:]) if b - a > 1)
    missing = (ordered[-1] - ordered[0] + 1 - len(ordered)) if ordered else 0

    instances = 0
    tracks: set[int] = set()
    for index, source_frame in enumerate(ordered):
        lines = []
        for key, x1, y1, x2, y2 in by_frame[source_frame]:
            lines.append(norm_line(0, key, x1, y1, x2, y2, width, height))
            tracks.add(key)
        instances += len(lines)
        (track_dir / f"{name}__frame_{index:06d}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))

    print(f"{name}: {len(ordered)} frames, {instances} boxes, {len(tracks)} trajectories, "
          f"{width}x{height} @ {frame_rate}Hz, source gaps={gaps} ({missing} frames)",
          flush=True)
    return {
        "video": name,
        "split": "train",
        "mode": mode,
        "frames": len(ordered),
        "instances": instances,
        "gt_tracks": len(tracks),
        "resolution": f"{width}x{height}:{len(ordered)}",
        "frame_rate": frame_rate,
        "source_gaps": gaps,
        "source_frames_missing": missing,
    }


MANIFEST_FIELDS = ["video", "split", "mode", "frames", "instances", "gt_tracks",
                   "resolution", "frame_rate", "source_gaps", "source_frames_missing",
                   "images_linked"]


def write_manifest(dst: Path, rows: list[dict]) -> None:
    with (dst / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {dst}: {len(rows)} sequences, "
          f"{sum(r['frames'] for r in rows)} frames, "
          f"{sum(r['gt_tracks'] for r in rows)} trajectories")


def link_images(dst: Path, name: str, ordered: list[int],
                paths: dict[int, str], image_root: Path) -> int:
    """Symlink the source frames under the sidecar naming, in lockstep.

    The tracking tool groups frames by the text before '__' and orders them by
    the number in the filename, so an image tree that keeps the release's own
    numbering would silently disagree with the sidecars wherever the source
    numbering has a hole. Linking here, from the same `ordered` list that wrote
    the sidecars, makes the two indices identical by construction.
    """
    out = dst / "images" / "train"
    out.mkdir(parents=True, exist_ok=True)
    linked = 0
    for index, source_frame in enumerate(ordered):
        relative = paths.get(source_frame)
        if not relative:
            continue
        target = (image_root / relative).resolve()
        if not target.is_file():
            continue
        link = out / f"{name}__frame_{index:06d}{target.suffix}"
        if not link.exists():
            link.symlink_to(target)
        linked += 1
    return linked


def run_csv(args) -> None:
    width, height = args.csv_size
    per_sequence, images = read_trajectory_csv(args.src, width, height)
    names = sorted(per_sequence)
    if args.sequences:
        wanted = set(args.sequences)
        names = [name for name in names if name in wanted]
    if args.max_sequences:
        names = names[: args.max_sequences]
    if not names:
        raise SystemExit(f"no sequences selected from {args.src}")

    rows = []
    for name in names:
        row = emit_sequence(args.dst, name, per_sequence[name], width, height,
                            args.csv_frame_rate, args.mode)
        if args.image_root:
            ordered = sorted(per_sequence[name])
            linked = link_images(args.dst, name, ordered, images.get(name, {}),
                                 args.image_root)
            row["images_linked"] = linked
            if linked != len(ordered):
                raise SystemExit(
                    f"{name}: linked {linked} images for {len(ordered)} annotated frames; "
                    "the image tree and the sidecars would not line up")
        rows.append(row)
    write_manifest(args.dst, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["motchallenge", "mots", "csv"], required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", help="default: every sequence found")
    ap.add_argument("--mode", default="external", help="value written to manifest 'mode'")
    ap.add_argument("--min-area", type=int, default=20, help="MOTS: minimum component pixels")
    ap.add_argument("--keep-class", type=int, default=1, help="MOTS: class id to keep")
    ap.add_argument("--min-visibility", type=float, default=0.0,
                    help="MOTChallenge: drop reference boxes below this visibility. "
                         "The benchmark's own scoring keeps every conf=1 class=1 row, "
                         "so the default keeps them all")
    ap.add_argument("--csv-size", type=int, nargs=2, metavar=("W", "H"),
                    help="csv source: pixel size the boxes are expressed in")
    ap.add_argument("--csv-frame-rate", type=int, default=30, help="csv source: source Hz")
    ap.add_argument("--image-root", type=Path,
                    help="csv source: symlink each annotated frame's image under the "
                         "sidecar naming, resolving the table's image_path against this root")
    ap.add_argument("--max-sequences", type=int)
    args = ap.parse_args()

    if args.source == "csv":
        if not args.csv_size:
            raise SystemExit("--source csv requires --csv-size W H")
        return run_csv(args)

    if args.source == "motchallenge":
        candidates = [p for p in sorted(args.src.iterdir())
                      if p.is_dir() and (p / "gt" / "gt.txt").is_file()]
        candidates = collapse_mot17_duplicates(candidates)
    else:
        candidates = [p for p in sorted(args.src.iterdir())
                      if p.is_dir() and (p / "instances").is_dir()]
    if args.sequences:
        wanted = set(args.sequences)
        candidates = [p for p in candidates if p.name in wanted or MOT_SUFFIX.sub("", p.name) in wanted]
    if args.max_sequences:
        candidates = candidates[: args.max_sequences]
    if not candidates:
        raise SystemExit(f"no usable sequences under {args.src}")

    manifest_rows = []
    for seq_dir in candidates:
        name = MOT_SUFFIX.sub("", seq_dir.name)
        if args.source == "motchallenge":
            by_frame, width, height, frame_rate = read_motchallenge(seq_dir, args.min_visibility)
        else:
            by_frame, width, height = read_mots(seq_dir, args.min_area, args.keep_class)
            frame_rate = 30
        manifest_rows.append(
            emit_sequence(args.dst, name, by_frame, width, height, frame_rate, args.mode))
    write_manifest(args.dst, manifest_rows)


if __name__ == "__main__":
    main()
