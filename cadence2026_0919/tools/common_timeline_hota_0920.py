#!/usr/bin/env python3
"""HOTA on one common timeline: every annotated frame, whatever the interval.

The paper's HOTA is computed at the processed instants only, so its evaluation
frames change with Delta (review item R-13). Here each arm's tracks are carried
across the frames it skipped by linear interpolation between consecutive
observations of the same identity at consecutive processed frames (no
extrapolation, and no filling of gaps the tracker left in frames it did process) -- the tracklet interpolation ByteTrack uses (Zhang et al.,
ECCV 2022) -- and HOTA (Luiten et al., IJCV 2021; TrackEval) is computed on all
annotated frames against the full reference, per sequence, then averaged, as
tools/compute_hota.py does.

Self-check: at Delta = 1 nothing is interpolated, so the result must equal the
processed-frame HOTA of the same arm to the last digit.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools")
from compute_hota import iou_matrix  # noqa: E402
from trackeval.metrics import HOTA  # noqa: E402

ROOT = Path("datasets/grapemots_det_721")


def gt_for(stem, w=3840, h=2160):
    ids, boxes = [], []
    p = ROOT / "tracks" / "all" / f"{stem}.txt"
    if p.is_file():
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) != 6: continue
            _c, tid, xc, yc, bw, bh = map(float, q)
            xc, yc, bw, bh = xc * w, yc * h, bw * w, bh * h
            ids.append(int(tid)); boxes.append([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
    return ids, boxes


def record(video_entry, interpolate):
    names = video_entry["frame_names"]
    video = video_entry["video"]
    all_frames = sorted(p.name for p in (ROOT / "images" / "all").iterdir() if p.name.split("__")[0] == video)
    idx = {n: i for i, n in enumerate(all_frames)}
    T = len(all_frames) if interpolate else len(names)
    frame_of = [idx[n] for n in names]
    pred = [dict() for _ in range(T)]
    obs = defaultdict(list)
    for k, (ids, boxes) in enumerate(zip(video_entry["frame_predicted_ids"], video_entry["frame_predicted_boxes"])):
        t = frame_of[k] if interpolate else k
        for i, b in zip(ids, boxes):
            pred[t][i] = b; obs[i].append((t, np.asarray(b, float)))
    if interpolate:
        # Fill only frames the arm never processed: an identity seen at two
        # CONSECUTIVE processed frames is carried across the skipped frames between
        # them. Gaps the tracker left inside its own processed frames stay empty,
        # so at Delta = 1 nothing changes.
        nxt = {a: b for a, b in zip(frame_of, frame_of[1:])}
        for i, seq in obs.items():
            for (t0, b0), (t1, b1) in zip(seq, seq[1:]):
                if nxt.get(t0) != t1:
                    continue
                for t in range(t0 + 1, t1):
                    a = (t - t0) / (t1 - t0)
                    pred[t][i] = (b0 + a * (b1 - b0)).tolist()
    gi, gb = [], []
    for t in range(T):
        stem = Path(all_frames[t] if interpolate else names[t]).stem
        w, h = (1920, 1080) if video in ("PathPlanning_1", "PathPlanning_3") else (3840, 2160)
        ids, boxes = gt_for(stem, w, h); gi.append(ids); gb.append(boxes)
    # dense ids, as compute_hota.per_sequence expects
    gmap = {g: j for j, g in enumerate(sorted({g for f in gi for g in f}))}
    pmap = {p: j for j, p in enumerate(sorted({p for f in pred for p in f}))}
    g_ids = [np.array([gmap[g] for g in f], dtype=int) for f in gi]
    t_ids = [np.array([pmap[p] for p in f], dtype=int) for f in pred]
    sims = [iou_matrix(np.asarray(gb[t], float).reshape(-1, 4), np.asarray(list(pred[t].values()), float).reshape(-1, 4))
            for t in range(T)]
    return {"num_timesteps": T, "num_gt_ids": len(gmap), "num_tracker_ids": len(pmap),
            "num_gt_dets": int(sum(len(x) for x in g_ids)), "num_tracker_dets": int(sum(len(x) for x in t_ids)),
            "gt_ids": g_ids, "tracker_ids": t_ids, "similarity_scores": sims}


def main():
    arms = sorted(Path(sys.argv[1]).glob(sys.argv[2]))
    metric = HOTA(); out = {}
    for f in arms:
        d = json.loads(f.read_text())
        row = {}
        for mode, interp in (("processed", False), ("common", True)):
            per = [metric.eval_sequence(record(v, interp)) for v in d["videos"]]
            row[mode] = {k: float(np.mean([np.mean(r[k]) for r in per])) for k in ("HOTA", "DetA", "AssA")}
        out[f.stem] = row
        print(f"{f.stem:24s} processed {row['processed']['HOTA']:.4f}  common {row['common']['HOTA']:.4f}")
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
