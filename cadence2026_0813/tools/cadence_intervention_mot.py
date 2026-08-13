#!/usr/bin/env python3
"""The processing-cadence contrast, repeated on two densely annotated corpora.

The vineyard experiment holds footage, reference and scoring instants fixed and
changes only which frames reach the tracker. It could only be run where a source
video existed beside a sparse annotation, which in the vineyard corpora meant one
release from one campaign. MOT17 and MOT20 label every frame, so the same
contrast can be built there by choosing the scoring instants rather than
inheriting them: keep one annotated frame in k, let one arm see only those and
the other see all k, and score both on the same retained frames.

Two things differ from the vineyard run and both are deliberate. Detection is
replaced by the annotated boxes, so U is small by construction and the question
is whether the direction survives without a fragmenting detector. And the
reference is thinned with the processing, which makes the released arm here the
structural counterpart of the thinning ladder rather than of the released
cadence: G is the trajectories that survive in the retained frames, identical in
both arms at every k.

Output mirrors tools/fullrate_decompose.py: P, G, U, D, M and the identity check
per sequence, per k, per arm, per tau.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Boxes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_grapemots_mot import build_tracker, load_gt_tracks  # noqa: E402
from decompose_count_error import decompose  # noqa: E402

FRAME_RE = re.compile(r"__frame_(\d+)\.txt$")


def video_frames(root: Path, video: str) -> list[Path]:
    found: list[Path] = []
    for split in ("train", "val", "test"):
        found.extend((root / "tracks" / split).glob(f"{video}__frame_*.txt"))
    if not found:
        raise SystemExit(f"no track sidecars for {video} under {root}")
    return sorted(found, key=lambda p: int(FRAME_RE.search(p.name).group(1)))


def read_manifest(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with (root / "manifest.csv").open() as handle:
        for row in csv.DictReader(handle):
            first = row["resolution"].split(";")[0].split(":")[0]
            width, height = (int(value) for value in first.split("x"))
            rows[row["video"]] = {
                "size": (width, height),
                "frame_rate": float(row.get("frame_rate") or 30.0),
                "frames": int(row["frames"]),
            }
    return rows


def load_sequence(frames: list[Path], size: tuple[int, int]):
    """Every annotated frame of one sequence, as ids and pixel boxes."""
    width, height = size
    ids: list[list[int]] = []
    boxes: list[np.ndarray] = []
    for path in frames:
        frame_ids, frame_boxes = load_gt_tracks(path, width, height)
        ids.append([int(t) for t in frame_ids])
        boxes.append(np.asarray(frame_boxes, dtype=np.float32).reshape(-1, 4))
    return ids, boxes


def run_arm(gt_ids, gt_boxes, size, tracker_cfg, frame_rate, processed, scored):
    """Track the frames in `processed`; report what is live at each frame in `scored`.

    `processed` and `scored` are sets of frame indices. The released arm processes
    exactly what it scores; the source-rate arm processes everything and is read
    at the same instants, so the two differ in observations and in nothing else.
    """
    width, height = size
    tracker = build_tracker(tracker_cfg, frame_rate=frame_rate)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    predicted_ids: list[list[int]] = []
    predicted_boxes: list[list[list[float]]] = []
    updates = 0
    for index in range(len(gt_ids)):
        if index not in processed:
            continue
        boxes = gt_boxes[index]
        scores = np.full((len(boxes), 1), 0.9, dtype=np.float32)
        classes = np.zeros((len(boxes), 1), dtype=np.float32)
        data = (
            np.concatenate([boxes.astype(np.float32), scores, classes], axis=1)
            if len(boxes)
            else np.empty((0, 6), dtype=np.float32)
        )
        tracks = tracker.update(
            Boxes(torch.as_tensor(data, dtype=torch.float32), (height, width)), blank
        )
        updates += 1
        if index in scored:
            if len(tracks):
                predicted_ids.append([int(t) for t in tracks[:, 4]])
                predicted_boxes.append(np.asarray(tracks[:, :4], dtype=float).tolist())
            else:
                predicted_ids.append([])
                predicted_boxes.append([])
    return predicted_ids, predicted_boxes, updates


def pool(records: list[dict], taus: list[int]) -> dict:
    pooled: dict[str, dict] = {}
    for key in sorted({(r["k"], r["arm"], r["tracker_cfg"]) for r in records}):
        k, arm, cfg = key
        label = f"k={k}|{arm}|{Path(cfg).stem}"
        pooled[label] = {}
        for tau in taus:
            terms = Counter()
            videos = 0
            for record in records:
                if (record["k"], record["arm"], record["tracker_cfg"]) != key:
                    continue
                videos += 1
                one = record["decomposition"][str(tau)]
                for name in ("P", "G", "U", "D", "M"):
                    terms[name] += one[name]
            G = terms["G"]
            pooled[label][str(tau)] = {
                "videos": videos,
                "P": terms["P"], "G": G,
                "U": terms["U"], "D": terms["D"], "M": terms["M"],
                "assigned_fraction": 1 - terms["M"] / G if G else None,
                "signed_error": (terms["P"] - G) / G if G else None,
                "identity_holds": (terms["U"] + terms["D"] - terms["M"]) == (terms["P"] - G),
            }
    return pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 15, 30, 60])
    parser.add_argument(
        "--tracker", action="append", default=[],
        help="tracker yaml, repeatable; default bytetrack.yaml",
    )
    parser.add_argument("--frame-rate", type=int, default=30,
                        help="rate handed to the tracker, so the buffer counts processed frames")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--taus", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.tracker:
        args.tracker = ["bytetrack.yaml"]
    if 1 not in args.taus:
        parser.error("--taus must include 1")
    return args


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.root)
    videos = args.videos or sorted(manifest)
    records: list[dict] = []

    for video in videos:
        entry = manifest[video]
        frames = video_frames(args.root, video)
        gt_ids, gt_boxes = load_sequence(frames, entry["size"])
        total = len(gt_ids)
        started = time.time()

        for tracker_cfg in args.tracker:
            everything = set(range(total))
            source_ids, source_boxes, source_updates = run_arm(
                gt_ids, gt_boxes, entry["size"], tracker_cfg,
                args.frame_rate, everything, everything,
            )
            for k in args.steps:
                scored = set(range(0, total, k))
                order = sorted(scored)
                released_ids, released_boxes, released_updates = run_arm(
                    gt_ids, gt_boxes, entry["size"], tracker_cfg,
                    args.frame_rate, scored, scored,
                )
                arms = {
                    "released": (released_ids, released_boxes, released_updates),
                    "source": (
                        [source_ids[i] for i in order],
                        [source_boxes[i] for i in order],
                        source_updates,
                    ),
                }
                truth_ids = [gt_ids[i] for i in order]
                truth_boxes = [gt_boxes[i].tolist() for i in order]
                for arm, (pred_ids, pred_boxes, updates) in arms.items():
                    stub = {
                        "video": video,
                        "frame_predicted_ids": pred_ids,
                        "frame_predicted_boxes": pred_boxes,
                        "frame_gt_ids": truth_ids,
                        "frame_gt_boxes": truth_boxes,
                    }
                    record = {
                        "video": video,
                        "corpus": args.root.name,
                        "k": k,
                        "arm": arm,
                        "tracker_cfg": tracker_cfg,
                        "source_frames": total,
                        "scored_frames": len(order),
                        "tracker_frames": updates,
                        "decomposition": {
                            str(tau): decompose(stub, args.match_iou, tau) for tau in args.taus
                        },
                    }
                    records.append(record)
                    one = record["decomposition"]["1"]
                    print(
                        f"{video} k={k:<3} {arm:<9} {Path(tracker_cfg).stem}: "
                        f"tracker={updates:<5} scored={len(order):<5} "
                        f"P={one['P']:<5} G={one['G']:<5} U={one['U']:<5} D={one['D']:<5} "
                        f"M={one['M']:<5} e={one['signed_error']:+.4f} "
                        f"holds={one['identity_holds']}",
                        flush=True,
                    )
        print(f"  {video}: {time.time() - started:.0f}s", flush=True)

    payload = {
        "config": {
            "root": str(args.root),
            "videos": videos,
            "steps": args.steps,
            "trackers": args.tracker,
            "frame_rate": args.frame_rate,
            "match_iou": args.match_iou,
            "taus": args.taus,
            "out": str(args.out),
        },
        "runs": records,
        "pooled": pool(records, args.taus),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    for label, block in payload["pooled"].items():
        one = block["1"]
        print(
            f"POOLED {label}: P={one['P']} G={one['G']} U={one['U']} D={one['D']} "
            f"M={one['M']} assigned={one['assigned_fraction']:.4f} "
            f"e={one['signed_error']:+.4f} holds={one['identity_holds']}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
