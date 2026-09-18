#!/usr/bin/env python3
"""HOTA, DetA and AssA for the eight tracking arms, from the stored per-frame boxes.

Reviewers asked for the metric family designed for the association question this
paper is about. No re-inference is needed: every arm JSON already carries
frame_gt_boxes / frame_gt_ids and frame_predicted_boxes / frame_predicted_ids, so
the similarity matrices can be rebuilt offline and handed to TrackEval.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/kou/my_env/yolo26/.venv/lib/python3.12/site-packages")
from trackeval.metrics import HOTA  # noqa: E402

# The results directory and the arm list used to be constants here, so every new
# tracking arm silently scored the CBDCom submission's old eight arms instead of
# itself.  Both come from the command line now:
#
#     compute_hota.py                      the original eight arms, unchanged
#     compute_hota.py <dir>                every *.json in <dir>
#     compute_hota.py <dir> a.json b.json  just those, in that order
#
# Any JSON written by track_grapemots_mot.py works: the per-frame boxes and ids
# it already stores are all HOTA needs, so nothing is re-inferred.
DEFAULT_RES = Path("/home/kou/my_env/yolo26/runs/cbdcom2026_queue/results")
DEFAULT_ARMS = [
    ("Tiles, conf 0.55",   "arm_conf0.55.json"),
    ("Resize + BoT-SORT",  "arm_resize.json"),
    ("Tiles, conf 0.40",   "arm_conf0.40.json"),
    ("Tiles + IoS merge",  "arm_ios_tiled.json"),
    ("Tiles + BoT-SORT",   "arm_botsort_tiled.json"),
    ("Tiles + YOLO11s",    "arm_yolo11s_tiled.json"),
    ("Tiles + ByteTrack",  "arm_bytetrack_tiled.json"),
    ("Tiles + ReID",       "arm_reid.json"),
]
SKIP = {"hota_arms.json", "nwd_vs_iou.json"}


def resolve(argv):
    if not argv:
        return DEFAULT_RES, DEFAULT_ARMS
    res = Path(argv[0])
    names = argv[1:]
    if not names:
        names = [f.name for f in sorted(res.glob("*.json")) if f.name not in SKIP]
    return res, [(n[:-5] if n.endswith(".json") else n, n) for n in names]


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def build(entries):
    """Pool consecutive videos into one sequence-like record, ids offset per video.

    KEEP THIS, but do not use it for anything that will be compared against a
    published number.  Pooling and per-sequence averaging are different
    conventions and they do not give the same answer: on the AppleMOT ByteTrack
    reproduction, pooling gives HOTA 0.3407 / AssA 0.5534 while per-sequence
    averaging gives 0.2500 / 0.3688, and the second is what TrackEval's own
    MotChallenge2DBox pipeline returns to four decimals.  Pooling flatters the
    association term, because tracks in different videos can never be confused
    and the global alignment score improves accordingly.
    """
    gt_ids, tk_ids, sims = [], [], []
    gt_off = tk_off = 0
    for e in entries:
        g_ids, g_box = e["frame_gt_ids"], e["frame_gt_boxes"]
        p_ids, p_box = e["frame_predicted_ids"], e["frame_predicted_boxes"]
        gmax = max([max(f) for f in g_ids if f] + [-1]) + 1
        pmax = max([max(f) for f in p_ids if f] + [-1]) + 1
        for t in range(len(g_ids)):
            gt_ids.append(np.array([i + gt_off for i in g_ids[t]], dtype=int))
            tk_ids.append(np.array([i + tk_off for i in p_ids[t]], dtype=int))
            sims.append(iou_matrix(g_box[t], p_box[t]))
        gt_off += gmax; tk_off += pmax
    return {
        "num_timesteps": len(gt_ids),
        "num_gt_ids": gt_off, "num_tracker_ids": tk_off,
        "num_gt_dets": int(sum(len(x) for x in gt_ids)),
        "num_tracker_dets": int(sum(len(x) for x in tk_ids)),
        "gt_ids": gt_ids, "tracker_ids": tk_ids, "similarity_scores": sims,
    }


def per_sequence(entries):
    """One record per video; the caller averages. This is the reportable convention."""
    out = []
    for e in entries:
        g_ids, g_box = e["frame_gt_ids"], e["frame_gt_boxes"]
        p_ids, p_box = e["frame_predicted_ids"], e["frame_predicted_boxes"]
        gi = [np.array(x, dtype=int) for x in g_ids]
        ti = [np.array(x, dtype=int) for x in p_ids]
        sims = [iou_matrix(g_box[t], p_box[t]) for t in range(len(g_ids))]
        gmax = max([max(x) for x in gi if len(x)] + [-1]) + 1
        tmax = max([max(x) for x in ti if len(x)] + [-1]) + 1
        out.append({"num_timesteps": len(gi), "num_gt_ids": gmax, "num_tracker_ids": tmax,
                    "num_gt_dets": int(sum(len(x) for x in gi)),
                    "num_tracker_dets": int(sum(len(x) for x in ti)),
                    "gt_ids": gi, "tracker_ids": ti, "similarity_scores": sims})
    return out


def main():
    pooled = "--pooled" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--pooled"]
    RES, ARMS = resolve(argv)
    if not ARMS:
        print(f"no arm JSON found in {RES}")
        return
    metric = HOTA()
    out = {}
    print(f"reading {RES}  ({'POOLED - not comparable to published numbers' if pooled else 'per-sequence mean, TrackEval convention'})")
    print(f"{'arm':22s} {'HOTA':>7s} {'DetA':>7s} {'AssA':>7s} {'LocA':>7s}")
    for label, fname in ARMS:
        f = RES / fname
        if not f.exists():
            print(f"{label:22s} {'missing':>7s}")
            continue
        d = json.loads(f.read_text())
        if "videos" not in d:
            print(f"{label:22s} {'no boxes':>7s}")
            continue
        if pooled:
            res = metric.eval_sequence(build(d["videos"]))
            row = {k: float(np.mean(res[k])) for k in ("HOTA", "DetA", "AssA", "LocA")}
        else:
            per = [metric.eval_sequence(x) for x in per_sequence(d["videos"])]
            row = {k: float(np.mean([np.mean(r[k]) for r in per]))
                   for k in ("HOTA", "DetA", "AssA", "LocA")}
        out[label] = row
        print(f"{label:22s} {row['HOTA']:7.4f} {row['DetA']:7.4f} "
              f"{row['AssA']:7.4f} {row['LocA']:7.4f}")
    (RES / "hota_arms.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", RES / "hota_arms.json")


if __name__ == "__main__":
    main()
