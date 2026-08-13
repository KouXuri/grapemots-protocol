#!/usr/bin/env python3
"""HOTA, DetA and AssA for the cohort in which every sequence is model-unseen.

tools/hota_panelA.py covers the six out-of-fold videos of the main configuration
table. The stronger cohort is the leave-one-video-out one: eleven sequences, each
scored by a checkpoint trained without it, so no sequence contributed to the
model that reads it. That cohort existed only as four numbers in the running
text. This rebuilds it from the stored per-frame boxes, with U, D and M re-derived
so the rows are proved to be the same cohort rather than assumed to be.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/kou/my_env/yolo26")
LOVO = ROOT / "runs/grapemots_journal_0805/results"
OUT = ROOT / "runs/decomp_0812/results/hota_panelB.json"

VIDEOS = ["NoPathPlanning_1", "NoPathPlanning_2", "NoPathPlanning_3",
          "PathPlanning_1", "PathPlanning_2", "PathPlanning_3", "PathPlanning_4",
          "PathPlanning_5", "PathPlanning_6", "PathPlanning_7", "PathPlanning_8"]

ARMS = [
    ("Confidence 0.55", "conf055"),
    ("Confidence 0.40", "conf040"),
    ("IoS merge", "ios"),
    ("BoT-SORT, buffer 30", "botsort"),
    ("ByteTrack, buffer 30", "bytetrack"),
    ("BoT-SORT + ReID", "reid"),
]

sys.path.insert(0, str(ROOT / ".venv/lib/python3.12/site-packages"))
from trackeval.metrics import HOTA  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from decompose_count_error import decompose  # noqa: E402
from hota_panelA import build  # noqa: E402


def main() -> None:
    metric = HOTA()
    out = {}
    print(f"{'row':22s} {'P':>6s} {'G':>5s} {'U':>5s} {'D':>5s} {'M':>5s} "
          f"{'e':>8s} {'assigned':>9s} {'HOTA':>7s} {'DetA':>7s} {'AssA':>7s}")
    for label, token in ARMS:
        entries = []
        for video in VIDEOS:
            path = LOVO / f"lovo_track_{video}_{token}.json"
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            payload = json.loads(path.read_text())
            records = payload["videos"]
            if isinstance(records, dict):
                records = list(records.values())
            if len(records) != 1:
                raise SystemExit(f"{path}: {len(records)} records, expected 1")
            entries.append(records[0])

        terms = Counter()
        for entry in entries:
            one = decompose(entry, 0.5, 1)
            if not one["identity_holds"]:
                raise SystemExit(f"{label}/{entry['video']}: P-G != U+D-M")
            for key in ("P", "G", "U", "D", "M"):
                terms[key] += one[key]
        e = (terms["P"] - terms["G"]) / terms["G"]
        assigned = 1 - terms["M"] / terms["G"]

        res = metric.eval_sequence(build(entries))
        row = {k: float(np.mean(res[k])) for k in ("HOTA", "DetA", "AssA", "LocA")}
        row.update({k: int(terms[k]) for k in ("P", "G", "U", "D", "M")})
        row["signed_error"] = e
        row["assigned_fraction"] = assigned
        out[label] = row
        print(f"{label:22s} {terms['P']:6d} {terms['G']:5d} {terms['U']:5d} "
              f"{terms['D']:5d} {terms['M']:5d} {e:+8.4f} {assigned:9.4f} "
              f"{row['HOTA']:7.4f} {row['DetA']:7.4f} {row['AssA']:7.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"videos": VIDEOS, "rows": out}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
