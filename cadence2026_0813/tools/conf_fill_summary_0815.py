#!/usr/bin/env python3
"""Fill the gap in the confidence sweep between 0.70 and 0.85.

The operating-point path crosses zero somewhere between confidence 0.70, which
over-counts, and 0.85, which under-counts. Those two configurations reach 0.51
and 0.10 of the reference, so interpolating the crossing between them says
almost nothing about where it sits. Two further points, replayed from the same
detection cache at 0.75 and 0.80, close that gap with measurements.

The cohort and the checkpoints are the ones Panel A of the configuration table
uses: six out-of-fold 2024 videos, each read by the checkpoint that did not
train on it, and one cache per video built at confidence 0.10.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / "tools"))

from decompose_count_error import decompose  # noqa: E402
from hota_panelA import VIDEOS  # noqa: E402

FILL = ROOT / "runs/conf_fill_0815/results"
OUT = FILL / "conf_fill.json"


def pooled(token: str) -> dict:
    terms = Counter()
    for video in VIDEOS:
        path = FILL / f"cached_{video}_{token}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        for entry in json.loads(path.read_text())["videos"]:
            one = decompose(entry, 0.5, 1)
            if not one["identity_holds"]:
                raise SystemExit(f"{token}/{entry['video']}: P-G != U+D-M")
            for key in ("P", "G", "U", "D", "M"):
                terms[key] += one[key]
    G = terms["G"]
    return {
        "P": terms["P"], "G": G, "U": terms["U"], "D": terms["D"], "M": terms["M"],
        "signed_error": (terms["P"] - G) / G,
        "assigned_fraction": 1 - terms["M"] / G,
    }


def main() -> None:
    rows = {label: pooled(token) for label, token in
            (("Confidence 0.80", "conf080"), ("Confidence 0.75", "conf075"))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"videos": VIDEOS, "rows": rows}, indent=1) + "\n")
    for label, row in rows.items():
        print(f"{label}: P={row['P']} G={row['G']} U={row['U']} D={row['D']} "
              f"M={row['M']} e={row['signed_error']:+.4f} "
              f"1-M/G={row['assigned_fraction']:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
