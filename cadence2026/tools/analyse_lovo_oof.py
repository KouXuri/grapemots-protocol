#!/usr/bin/env python3
"""Pool the leave-one-video-out tracking runs into the U/D/M account.

The manuscript's real-pipeline contrast rests on six model-unseen videos from
three bespoke assignments, which is why it has to be described as exploratory.
Under leave-one-video-out every one of the eleven videos is scored by a
checkpoint that never saw it, so the same six configurations can be compared on
the whole release rather than on a subset.

The decomposition itself is imported from tools/decompose_count_error.py rather
than rewritten. That matters for comparability: the ownership rule -- per-frame
one-to-one assignment by maximum total IoU, gated at 0.5, then the trajectory
that covers a track in the most frames owns it, ties broken by the smaller
trajectory identifier -- is the definition the manuscript states, and a second
implementation of it would be a second definition.

Every reported row is checked against P - G = U + D - M before it is used. A row
that fails the identity is a bug, not a finding, and is reported as a failure
rather than quietly averaged in.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose_count_error import decompose  # noqa: E402

# The same decomposition serves any run family named <prefix><video>_<arm>.json,
# so the cached-detection replay is analysed by the identical code path as the
# live leave-one-video-out runs rather than by a second implementation.
PREFIX = "lovo_track_"


def name_re():
    return re.compile(r"^" + re.escape(PREFIX) +
                      r"(?P<video>[A-Za-z]+_\d+)_(?P<arm>[a-z0-9_]+)\.json$")

ARM_LABELS = {
    "conf055": "Tiles, conf. 0.55",
    "conf040": "Tiles, conf. 0.40",
    "ios": "Tiles + IoS",
    "botsort": "Tiles + BoT-SORT",
    "bytetrack": "Tiles + ByteTrack",
    "reid": "Tiles + ReID",
}
FOCAL_PAIR = ("conf055", "reid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--prefix", default="lovo_track_",
                    help="filename prefix identifying the run family")
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    global PREFIX
    PREFIX = args.prefix

    per_arm: dict[str, list[dict]] = defaultdict(list)
    failures: list[str] = []
    pattern = name_re()
    for path in sorted(args.results.glob(f"{PREFIX}*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        arm = match.group("arm")
        payload = json.loads(path.read_text())
        for video in payload["videos"]:
            if "frame_predicted_boxes" not in video:
                failures.append(f"{path.name}: no per-frame boxes, cannot decompose")
                continue
            for tau in args.taus:
                row = decompose(video, threshold=args.match_iou, tau=tau)
                row["arm"] = arm
                row["source"] = path.name
                if not row["identity_holds"]:
                    failures.append(
                        f"{path.name} {row['video']} tau={tau}: "
                        f"P-G={row['P'] - row['G']} but U+D-M={row['U'] + row['D'] - row['M']}")
                per_arm[arm].append(row)

    videos = sorted({row["video"] for rows in per_arm.values() for row in rows})
    report = {
        "videos": videos,
        "video_count": len(videos),
        "arms": sorted(per_arm),
        "identity_failures": failures,
        "taus": args.taus,
        "match_iou": args.match_iou,
    }

    # Pooled over trajectories at tau = 1, the manuscript's primary endpoint.
    pooled = {}
    for arm, rows in per_arm.items():
        at_tau1 = [row for row in rows if row["tau"] == 1]
        if not at_tau1:
            continue
        U = sum(row["U"] for row in at_tau1)
        D = sum(row["D"] for row in at_tau1)
        M = sum(row["M"] for row in at_tau1)
        P = sum(row["P"] for row in at_tau1)
        G = sum(row["G"] for row in at_tau1)
        pooled[arm] = {
            "videos": len(at_tau1), "U": U, "D": D, "M": M, "P": P, "G": G,
            "assigned_fraction": 1 - M / G if G else None,
            "signed_error": (P - G) / G if G else None,
            "identity_holds": (U + D - M) == (P - G),
        }
    report["pooled_tau1"] = pooled

    # The focal contrast, checked per video rather than only in the pool.
    low, high = FOCAL_PAIR
    direction = []
    if low in per_arm and high in per_arm:
        by_video = {arm: {row["video"]: row for row in per_arm[arm] if row["tau"] == 1}
                    for arm in FOCAL_PAIR}
        for video in videos:
            a, b = by_video[low].get(video), by_video[high].get(video)
            if not a or not b:
                continue
            direction.append({
                "video": video,
                "abs_error_lower_for_conf055": abs(a["signed_error"]) < abs(b["signed_error"]),
                "assigned_higher_for_reid": (1 - a["M"] / a["G"]) < (1 - b["M"] / b["G"]),
                "error_conf055": a["signed_error"], "error_reid": b["signed_error"],
                "assigned_conf055": 1 - a["M"] / a["G"], "assigned_reid": 1 - b["M"] / b["G"],
            })
    holds = sum(1 for entry in direction
                if entry["abs_error_lower_for_conf055"] and entry["assigned_higher_for_reid"])
    report["focal_direction"] = {
        "pair": list(FOCAL_PAIR), "videos": len(direction), "both_hold": holds,
        "per_video": direction,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")

    lines = ["# Leave-one-video-out real-pipeline summary", "",
             f"{len(videos)} model-unseen videos, {len(per_arm)} configurations, "
             f"matching IoU {args.match_iou}.", ""]
    if failures:
        lines += ["**Identity check failed on the following rows; they are not "
                  "interpreted:**", ""] + [f"- {text}" for text in failures] + [""]
    else:
        lines += ["All decompositions satisfy `P - G = U + D - M`.", ""]

    lines += ["| Configuration | U | D | M | Assigned | e |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in sorted(pooled, key=lambda a: pooled[a]["signed_error"]):
        entry = pooled[arm]
        lines.append(f"| {ARM_LABELS.get(arm, arm)} | {entry['U']} | {entry['D']} | "
                     f"{entry['M']} | {entry['assigned_fraction']:.3f} | "
                     f"{entry['signed_error']:+.3f} |")

    lines += ["", f"Focal pair {FOCAL_PAIR[0]} vs {FOCAL_PAIR[1]}: both directions hold in "
                  f"{holds}/{len(direction)} videos.", ""]
    args.markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}\nwrote {args.markdown}")
    if failures:
        raise SystemExit(f"{len(failures)} decomposition rows failed the identity check")


if __name__ == "__main__":
    main()
