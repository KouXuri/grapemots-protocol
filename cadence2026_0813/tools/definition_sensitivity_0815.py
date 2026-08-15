#!/usr/bin/env python3
"""How much of the cadence contrast is the definition rather than the cadence?

A reviewer asked three questions the manuscript answered only by construction:
whether tau filters the tracks that enter U, D and M as well as the tracks that
enter P; whether the ownership IoU of 0.5 sets the result; and what the owner
rule hides when one predicted track drifts across two trajectories. All three
are answered from the frozen per-frame boxes of the cadence arms, so no GPU,
no imagery and no re-tracking is involved.

Reported per arm:

  tau, asymmetric and symmetric   the reference filtered like the prediction
  ownership IoU 0.3 / 0.5 / 0.7   U, D, M, e and the assigned fraction
  ties                            owners decided by the smaller identifier
  purity                          owner frames over matched frames of a track
  identity switches               a predicted track changing owner in time
  fragmentation                   a trajectory's coverage broken into runs
  frame recall                    annotated boxes matched in their own frame
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decompose_count_error import frame_matches  # noqa: E402


def per_video(record: dict, threshold: float, tau: int) -> dict:
    """U, D, M and the diagnostics the identity alone does not show."""
    pred_by_frame = record["frame_predicted_ids"]
    gt_by_frame = record["frame_gt_ids"]

    life = Counter(t for frame in pred_by_frame for t in frame)
    kept = {t for t, n in life.items() if n >= tau}

    overlap: dict[int, Counter] = defaultdict(Counter)
    owner_sequence: dict[int, list] = defaultdict(list)
    coverage_frames: dict[int, list] = defaultdict(list)
    matched_boxes = 0
    gt_boxes_total = 0

    for index, (pb, pi, gb, gi) in enumerate(
        zip(record["frame_predicted_boxes"], pred_by_frame,
            record["frame_gt_boxes"], gt_by_frame)
    ):
        gt_boxes_total += len(gi)
        live = [(b, t) for b, t in zip(pb, pi) if t in kept]
        if not live:
            continue
        pairs = frame_matches([b for b, _ in live], [t for _, t in live], gb, gi, threshold)
        matched_boxes += len(pairs)
        for pred, truth in pairs:
            overlap[pred][truth] += 1
            owner_sequence[pred].append((index, truth))
            coverage_frames[truth].append(index)

    all_pred = sorted(kept)
    all_gt = sorted({t for frame in gt_by_frame for t in frame})

    owner, ties = {}, 0
    for p in all_pred:
        if not overlap[p]:
            continue
        best = max(overlap[p].values())
        contenders = [g for g, n in overlap[p].items() if n == best]
        if len(contenders) > 1:
            ties += 1
        owner[p] = min(contenders)

    per_gt = Counter(owner.values())
    U = sum(1 for p in all_pred if p not in owner)
    D = sum(n - 1 for n in per_gt.values())
    M = sum(1 for g in all_gt if per_gt[g] == 0)
    P, G = len(all_pred), len(all_gt)

    # a track is pure when every frame it is matched in belongs to its owner
    purities, switches = [], 0
    for p, sequence in owner_sequence.items():
        if p not in owner:
            continue
        truths = [t for _, t in sequence]
        purities.append(truths.count(owner[p]) / len(truths))
        switches += sum(1 for a, b in zip(truths, truths[1:]) if a != b)

    # a trajectory is fragmented when its covered frames come in several runs
    fragments = 0
    for frames in coverage_frames.values():
        ordered = sorted(set(frames))
        fragments += 1 + sum(1 for a, b in zip(ordered, ordered[1:]) if b != a + 1)

    return {
        "video": record["video"], "arm": record["arm"], "tau": tau,
        "match_iou": threshold,
        "P": P, "G": G, "U": U, "D": D, "M": M,
        "identity_holds": (U + D - M) == (P - G),
        "signed_error": (P - G) / G if G else None,
        "assigned_fraction": 1 - M / G if G else None,
        "ties": ties,
        "impure_tracks": sum(1 for value in purities if value < 1.0),
        "mean_purity": sum(purities) / len(purities) if purities else None,
        "identity_switches": switches,
        "fragments": fragments,
        "matched_boxes": matched_boxes,
        "gt_boxes": gt_boxes_total,
        "frame_recall": matched_boxes / gt_boxes_total if gt_boxes_total else None,
    }


def symmetric_rows(record: dict, taus) -> list[dict]:
    """tau on predictions only, as reported, and on both sides."""
    pred_life = Counter(t for frame in record["frame_predicted_ids"] for t in frame)
    gt_life = Counter(t for frame in record["frame_gt_ids"] for t in frame)
    rows = []
    for tau in taus:
        P = sum(1 for n in pred_life.values() if n >= tau)
        G_asym = len(gt_life)
        G_sym = sum(1 for n in gt_life.values() if n >= tau)
        rows.append({
            "video": record["video"], "arm": record["arm"], "tau": tau,
            "P": P, "G_asymmetric": G_asym, "G_symmetric": G_sym,
            "e_asymmetric": (P - G_asym) / G_asym if G_asym else None,
            "e_symmetric": (P - G_sym) / G_sym if G_sym else None,
        })
    return rows


def median(values):
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=Path, nargs="+", required=True,
                    help="cadence_arms / cadence_timescale outputs with per-frame boxes")
    ap.add_argument("--labels", nargs="+", default=["rel", "src"])
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 3, 5, 8])
    ap.add_argument("--match-ious", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = []
    for path in args.arms:
        data = json.loads(path.read_text())
        for record in data["runs"]:
            if record["arm"] in args.labels and "frame_predicted_boxes" in record:
                records.append(record)
    videos = sorted({r["video"] for r in records})
    print(f"{len(records)} arm-runs over {len(videos)} videos: {', '.join(args.labels)}")

    report = {
        "videos": videos,
        "labels": list(args.labels),
        "gate": {}, "tau_symmetry": {}, "per_video": [],
    }

    for threshold in args.match_ious:
        for tau in args.taus:
            for label in args.labels:
                rows = [per_video(r, threshold, tau) for r in records if r["arm"] == label]
                if not rows:
                    continue
                report["per_video"].extend(rows)
                pooled = Counter()
                for row in rows:
                    for key in ("P", "G", "U", "D", "M", "ties", "identity_switches",
                                "fragments", "impure_tracks", "matched_boxes", "gt_boxes"):
                        pooled[key] += row[key]
                G = pooled["G"]
                key = f"iou{threshold}_tau{tau}_{label}"
                report["gate"][key] = {
                    "match_iou": threshold, "tau": tau, "arm": label,
                    "videos": len(rows),
                    "P": pooled["P"], "G": G, "U": pooled["U"],
                    "D": pooled["D"], "M": pooled["M"],
                    "signed_error": (pooled["P"] - G) / G if G else None,
                    "assigned_fraction": 1 - pooled["M"] / G if G else None,
                    "median_signed_error": median([r["signed_error"] for r in rows]),
                    "ties": pooled["ties"],
                    "impure_tracks": pooled["impure_tracks"],
                    "identity_switches": pooled["identity_switches"],
                    "fragments": pooled["fragments"],
                    "frame_recall": (pooled["matched_boxes"] / pooled["gt_boxes"]
                                     if pooled["gt_boxes"] else None),
                    "identity_holds": all(r["identity_holds"] for r in rows),
                }

    # the paired difference between the two arms, per video, at every gate and tau
    report["paired"] = {}
    for threshold in args.match_ious:
        for tau in args.taus:
            by_arm = {}
            for label in args.labels:
                by_arm[label] = {
                    r["video"]: r for r in report["per_video"]
                    if r["arm"] == label and r["tau"] == tau and r["match_iou"] == threshold
                }
            if len(by_arm) != 2:
                continue
            first, second = args.labels[0], args.labels[1]
            shared = sorted(set(by_arm[first]) & set(by_arm[second]))
            deltas = [by_arm[second][v]["signed_error"] - by_arm[first][v]["signed_error"]
                      for v in shared]
            report["paired"][f"iou{threshold}_tau{tau}"] = {
                "match_iou": threshold, "tau": tau, "videos": len(shared),
                "delta_median": median(deltas),
                "up": sum(1 for d in deltas if d > 0),
                "down": sum(1 for d in deltas if d < 0),
                "tie": sum(1 for d in deltas if d == 0),
            }

    for label in args.labels:
        rows = []
        for record in records:
            if record["arm"] == label:
                rows.extend(symmetric_rows(record, args.taus))
        for tau in args.taus:
            block = [r for r in rows if r["tau"] == tau]
            P = sum(r["P"] for r in block)
            G_asym = sum(r["G_asymmetric"] for r in block)
            G_sym = sum(r["G_symmetric"] for r in block)
            report["tau_symmetry"][f"tau{tau}_{label}"] = {
                "tau": tau, "arm": label, "videos": len(block),
                "P": P, "G_asymmetric": G_asym, "G_symmetric": G_sym,
                "e_asymmetric": (P - G_asym) / G_asym if G_asym else None,
                "e_symmetric": (P - G_sym) / G_sym if G_sym else None,
                "median_e_asymmetric": median([r["e_asymmetric"] for r in block]),
                "median_e_symmetric": median([r["e_symmetric"] for r in block]),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    print("\nownership gate, pooled over videos")
    print(f"{'gate':>5} {'tau':>4} {'arm':>4} {'P':>5} {'G':>5} {'U':>5} {'D':>5} {'M':>5}"
          f" {'e':>8} {'1-M/G':>7} {'ties':>5} {'IDSW':>5} {'recall':>7} {'ok':>3}")
    for key, cell in report["gate"].items():
        print(f"{cell['match_iou']:5.1f} {cell['tau']:4d} {cell['arm']:>4} {cell['P']:5d}"
              f" {cell['G']:5d} {cell['U']:5d} {cell['D']:5d} {cell['M']:5d}"
              f" {cell['signed_error']:+8.4f} {cell['assigned_fraction']:7.4f}"
              f" {cell['ties']:5d} {cell['identity_switches']:5d}"
              f" {cell['frame_recall']:7.4f} {'yes' if cell['identity_holds'] else 'NO':>3}")

    print("\npaired difference, second arm minus first")
    for key, cell in report["paired"].items():
        print(f"IoU {cell['match_iou']:.1f} tau {cell['tau']}: median "
              f"{cell['delta_median']:+.4f}, up/down/tie "
              f"{cell['up']}/{cell['down']}/{cell['tie']}")

    print("\ntau applied to the reference as well")
    for key, cell in report["tau_symmetry"].items():
        print(f"tau {cell['tau']} {cell['arm']:>4}: G {cell['G_asymmetric']} -> "
              f"{cell['G_symmetric']}, e {cell['e_asymmetric']:+.4f} -> "
              f"{cell['e_symmetric']:+.4f}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
