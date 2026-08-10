#!/usr/bin/env python3
"""The cadence contrast: released annotation rate against source rate.

Same footage, same checkpoint, same scoring instants. Only the rate at which
frames reach the tracker differs, so a difference between the two arms cannot be
attributed to the imagery, the detector, the objects or the reference.

Sequences are reported in two groups and never pooled. On a sequence the
detector trained on, the CONTRAST is still valid -- both arms share the
checkpoint -- but the absolute error is optimistic, so mixing it with the
model-unseen group would understate the error while appearing to add sample size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def released_errors(path: Path, tau: str = "1") -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {video["video"]: video["count_sensitivity"][tau]["signed_relative_error"]
            for video in payload["videos"]
            if video["count_sensitivity"][tau]["signed_relative_error"] is not None}


def sourcerate_errors(path: Path, step: int = 1, tau: int = 1) -> dict[str, float]:
    payload = json.loads(path.read_text())
    out = {}
    for record in payload.get("runs", payload.get("records", [])):
        if record.get("step") != step:
            continue
        rows = [cell for cell in record["count_error_surface"]
                if cell["min_track_len"] == tau and cell["signed_relative_error"] is not None]
        if rows:
            whole = max(rows, key=lambda cell: cell["window_frames"])
            out[record["video"]] = whole["signed_relative_error"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--released", type=Path, action="append", required=True)
    ap.add_argument("--sourcerate", type=Path, action="append", required=True)
    ap.add_argument("--model-unseen", nargs="+", required=True,
                    help="sequences whose checkpoint never trained on them")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    released, source = {}, {}
    for path in args.released:
        if path.is_file():
            released.update(released_errors(path))
    for path in args.sourcerate:
        if path.is_file():
            source.update(sourcerate_errors(path))

    unseen = set(args.model_unseen)
    rows = []
    for video in sorted(set(released) & set(source)):
        rows.append({"video": video,
                     "released": released[video], "source_rate": source[video],
                     "delta": source[video] - released[video],
                     "model_unseen": video in unseen})
    if not rows:
        raise SystemExit("no video appears in both arms")

    report = {"sequences": rows, "groups": {}}
    print(f"{'sequence':<14}{'released':>10}{'source rate':>13}{'delta':>10}  unseen")
    for row in rows:
        print(f"{row['video']:<14}{row['released']:>+10.3f}{row['source_rate']:>+13.3f}"
              f"{row['delta']:>+10.3f}  {'yes' if row['model_unseen'] else 'no'}")

    for label, subset in (("model-unseen", [r for r in rows if r["model_unseen"]]),
                          ("seen in training", [r for r in rows if not r["model_unseen"]])):
        if not subset:
            continue
        rel = np.array([r["released"] for r in subset])
        src = np.array([r["source_rate"] for r in subset])
        delta = src - rel
        entry = {"sequences": len(subset),
                 "released_median": float(np.median(rel)),
                 "source_rate_median": float(np.median(src)),
                 "delta_median": float(np.median(delta)),
                 "delta_positive": int((delta > 0).sum()),
                 "released_within_10pct": int((np.abs(rel) <= 0.10).sum())}
        report["groups"][label] = entry
        print(f"\n{label} ({len(subset)} sequences)")
        print(f"  released cadence median {entry['released_median']:+.3f}, "
              f"source rate median {entry['source_rate_median']:+.3f}")
        print(f"  the count moves by a median {entry['delta_median']:+.3f}; "
              f"it rises on {entry['delta_positive']}/{len(subset)}")
        print(f"  sequences whose released-cadence error is within 10%: "
              f"{entry['released_within_10pct']}/{len(subset)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
