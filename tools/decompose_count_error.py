#!/usr/bin/env python3
"""Split the signed count error into the three terms that produce it.

A signed aggregate error near zero can mean the system counted well, or it can
mean two large opposite errors cancelled. The paper argues the second happens, so
it cannot then rank configurations by the signed error alone. This script reports
the terms instead of their sum.

Assign every predicted track to at most one ground-truth trajectory: the one it
is matched to in the most frames, ties broken by the lower trajectory id so the
result does not depend on dictionary order. Then

    P - G = U + D - M
      U   predicted tracks assigned to no trajectory at all
      D   duplicates, sum over covered trajectories of (m_g - 1)
      M   trajectories that no predicted track was ever assigned to

U and D push the count up, M pushes it down, and the identity is checked on every
video rather than assumed. A configuration with a small net error and large U, D
and M is not counting well; it is cancelling.

The same per-frame dumps also answer a second question the manuscript raised
without measuring: the minimum-track-length filter tau is applied to predictions
but not to the reference. Applying it symmetrically changes what is being counted,
so both estimands are reported.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:  # greedy fallback, reported in the output so it is never silent
    HAVE_SCIPY = False

# The confidence arms were first run before per-frame dumps were kept, so prefer
# the re-run that has them and fall back to the older file, which will simply be
# reported as having no dump rather than silently dropping out of the table.
ARMS = [("Resize", "arm_resize"), ("conf 0.55", "arm_conf0.55"),
        ("ByteTrack", "arm_bytetrack_tiled"), ("conf 0.40", "arm_conf0.40"),
        ("YOLO11s", "arm_yolo11s_tiled"), ("IoS merge", "arm_ios_tiled"),
        ("BoT-SORT", "arm_botsort_tiled"), ("+ ReID", "arm_reid")]
FALLBACK = {"arm_conf0.55": "count_conf0.55", "arm_conf0.40": "count_conf0.40"}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def frame_matches(pred_boxes, pred_ids, gt_boxes, gt_ids, threshold):
    """(predicted id, trajectory id) pairs matched in one frame."""
    if not pred_ids or not gt_ids:
        return []
    scores = iou_matrix(np.asarray(pred_boxes, dtype=float),
                        np.asarray(gt_boxes, dtype=float))
    pairs = []
    if HAVE_SCIPY:
        rows, cols = linear_sum_assignment(-scores)
        for r, c in zip(rows, cols):
            if scores[r, c] >= threshold:
                pairs.append((pred_ids[r], gt_ids[c]))
    else:
        used_r, used_c = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-scores, axis=None),
                                           scores.shape))[0]
        for r, c in order:
            if scores[r, c] < threshold:
                break
            if r in used_r or c in used_c:
                continue
            used_r.add(r); used_c.add(c)
            pairs.append((pred_ids[r], gt_ids[c]))
    return pairs


def decompose(video, threshold=0.5, tau=1):
    """U, D, M and the identity check for one video at one tau."""
    pred_ids_by_frame = video["frame_predicted_ids"]
    gt_ids_by_frame = video["frame_gt_ids"]

    # tau filters predicted tracks by how many frames they are seen in, which is
    # exactly what the reported count does
    pred_life = Counter(t for frame in pred_ids_by_frame for t in frame)
    kept = {t for t, n in pred_life.items() if n >= tau}

    overlap: dict[int, Counter] = defaultdict(Counter)
    for pb, pi, gb, gi in zip(video["frame_predicted_boxes"], pred_ids_by_frame,
                              video["frame_gt_boxes"], gt_ids_by_frame):
        live = [(b, t) for b, t in zip(pb, pi) if t in kept]
        if not live:
            continue
        pairs = frame_matches([b for b, _ in live], [t for _, t in live],
                              gb, gi, threshold)
        for pred, truth in pairs:
            overlap[pred][truth] += 1

    all_pred = sorted(kept)
    all_gt = sorted({t for frame in gt_ids_by_frame for t in frame})
    # lower trajectory id wins a tie, so the assignment is reproducible
    owner = {p: min(overlap[p].items(), key=lambda kv: (-kv[1], kv[0]))[0]
             for p in all_pred if overlap[p]}

    per_gt = Counter(owner.values())
    U = sum(1 for p in all_pred if p not in owner)
    D = sum(n - 1 for n in per_gt.values())
    M = sum(1 for g in all_gt if per_gt[g] == 0)
    P, G = len(all_pred), len(all_gt)
    return {"video": video["video"], "tau": tau, "P": P, "G": G,
            "U": U, "D": D, "M": M, "identity_holds": (U + D - M) == (P - G),
            "signed_error": (P - G) / G if G else None,
            "covered_trajectories": len(per_gt)}


def symmetric_tau(video, taus):
    """Reference count with tau applied to predictions only, and to both sides.

    The manuscript filters short predicted tracks but keeps every annotated
    trajectory, so part of what looks like a tau effect is the asymmetry itself.
    """
    pred_life = Counter(t for frame in video["frame_predicted_ids"] for t in frame)
    gt_life = Counter(t for frame in video["frame_gt_ids"] for t in frame)
    rows = []
    for tau in taus:
        P = sum(1 for n in pred_life.values() if n >= tau)
        G_asym = len(gt_life)
        G_sym = sum(1 for n in gt_life.values() if n >= tau)
        rows.append({
            "tau": tau, "P": P, "G_asymmetric": G_asym, "G_symmetric": G_sym,
            "e_asymmetric": (P - G_asym) / G_asym if G_asym else None,
            "e_symmetric": (P - G_sym) / G_sym if G_sym else None,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--out", type=Path,
                    default=Path("results/count_decomposition.json"))
    args = ap.parse_args()

    print(f"assignment: {'Hungarian (scipy)' if HAVE_SCIPY else 'greedy fallback'}"
          f", match IoU {args.match_iou}")
    report = {"match_iou": args.match_iou,
              "assignment": "hungarian" if HAVE_SCIPY else "greedy",
              "arms": {}}

    print(f"\n{'arm':11s} {'video':6s} {'G':>4} {'P':>5} {'U':>5} {'D':>5} {'M':>4}"
          f" {'U+D-M':>6} {'P-G':>5} {'ok':>3}")
    for label, stem in ARMS:
        path = args.results / f"{stem}.json"
        if not path.exists() and stem in FALLBACK:
            path = args.results / f"{FALLBACK[stem]}.json"
        if not path.exists():
            print(f"{label:11s} missing {stem}.json"); continue
        data = json.loads(path.read_text())
        arm = {"videos": [], "symmetric_tau": {}}
        pooled = Counter()
        for video in data["videos"]:
            if "frame_predicted_boxes" not in video:
                print(f"{label:11s} {video['video']}: no frame dump, skipped")
                continue
            row = decompose(video, args.match_iou, tau=1)
            arm["videos"].append(row)
            arm["symmetric_tau"][video["video"]] = symmetric_tau(video, args.taus)
            for key in ("P", "G", "U", "D", "M"):
                pooled[key] += row[key]
            print(f"{label:11s} {video['video'][-5:]:6s} {row['G']:4d} {row['P']:5d}"
                  f" {row['U']:5d} {row['D']:5d} {row['M']:4d}"
                  f" {row['U'] + row['D'] - row['M']:6d} {row['P'] - row['G']:5d}"
                  f" {'yes' if row['identity_holds'] else 'NO':>3}")
        arm["pooled"] = dict(pooled)
        report["arms"][label] = arm

    print(f"\n{'arm':11s} {'U':>6} {'D':>6} {'M':>6} {'net e':>8}  interpretation")
    for label, arm in report["arms"].items():
        p = arm["pooled"]
        if not p:
            continue
        net = (p["P"] - p["G"]) / p["G"]
        gross = p["U"] + p["D"] + p["M"]
        print(f"{label:11s} {p['U']:6d} {p['D']:6d} {p['M']:6d} {net:+8.3f}"
              f"   gross errors {gross}, net {p['P'] - p['G']:+d}")

    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
