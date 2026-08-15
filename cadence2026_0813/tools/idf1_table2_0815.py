#!/usr/bin/env python3
"""IDF1 for every row of the configuration table, on the cohort already scored.

The table reports U, D, M and HOTA. A reviewer asked whether the decomposition is
IDF1 under another name, which is answerable from the same per-frame boxes: if
IDF1 ordered the rows the way the signed error does, the decomposition would be
redundant. It does not, and this prints both orders from one pass.

Panel A is the six out-of-fold 2024 videos, Panel B the eleven leave-one-video-out
ones, exactly the cohorts tools/hota_panelA.py and tools/hota_panelB.py use.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / ".venv/lib/python3.12/site-packages"))
sys.path.insert(0, str(ROOT / "tools"))

from trackeval.metrics import CLEAR, Identity  # noqa: E402

from decompose_count_error import decompose  # noqa: E402
from hota_panelA import ARMS as ARMS_A, VIDEOS as VIDEOS_A, build  # noqa: E402
from hota_assoc_rows import ARMS as ARMS_ASSOC  # noqa: E402
from hota_panelB import ARMS as ARMS_B, VIDEOS as VIDEOS_B, LOVO  # noqa: E402

OUT = ROOT / "runs/definition_0815/results/idf1_table2.json"


def panel_a_entries(token, directory):
    entries = []
    for video in VIDEOS_A:
        path = directory / f"cached_{video}_{token}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        entries.extend(json.loads(path.read_text())["videos"])
    return entries


def panel_b_entries(token):
    entries = []
    for video in VIDEOS_B:
        path = LOVO / f"lovo_track_{video}_{token}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        records = json.loads(path.read_text())["videos"]
        if isinstance(records, dict):
            records = list(records.values())
        entries.extend(records)
    return entries


def row_for(entries):
    terms = Counter()
    for entry in entries:
        one = decompose(entry, 0.5, 1)
        if not one["identity_holds"]:
            raise SystemExit(f"{entry['video']}: P-G != U+D-M")
        for key in ("P", "G", "U", "D", "M"):
            terms[key] += one[key]
    data = build(entries)
    idr = Identity().eval_sequence(data)
    clr = CLEAR().eval_sequence(data)
    return {
        "P": int(terms["P"]), "G": int(terms["G"]), "U": int(terms["U"]),
        "D": int(terms["D"]), "M": int(terms["M"]),
        "signed_error": (terms["P"] - terms["G"]) / terms["G"],
        "assigned_fraction": 1 - terms["M"] / terms["G"],
        "IDF1": float(idr["IDF1"]), "IDP": float(idr["IDP"]), "IDR": float(idr["IDR"]),
        "MT": int(clr["MT"]), "PT": int(clr["PT"]), "ML": int(clr["ML"]),
        "IDSW": int(clr["IDSW"]), "Frag": int(clr["Frag"]),
    }


def main() -> None:
    out = {"panelA": {}, "panelB": {}}
    print(f"{'row':24s} {'e':>8s} {'1-M/G':>7s} {'IDF1':>7s} {'MT':>4s} {'ML':>4s} "
          f"{'IDSW':>5s} {'Frag':>5s} {'D':>5s} {'M':>5s}")
    seen = set()
    for label, token, directory in list(ARMS_A) + list(ARMS_ASSOC):
        if label in seen:
            continue
        seen.add(label)
        row = row_for(panel_a_entries(token, directory))
        out["panelA"][label] = row
        print(f"A {label:22s} {row['signed_error']:+8.4f} {row['assigned_fraction']:7.4f} "
              f"{row['IDF1']:7.4f} {row['MT']:4d} {row['ML']:4d} {row['IDSW']:5d} "
              f"{row['Frag']:5d} {row['D']:5d} {row['M']:5d}")
    for label, token in ARMS_B:
        row = row_for(panel_b_entries(token))
        out["panelB"][label] = row
        print(f"B {label:22s} {row['signed_error']:+8.4f} {row['assigned_fraction']:7.4f} "
              f"{row['IDF1']:7.4f} {row['MT']:4d} {row['ML']:4d} {row['IDSW']:5d} "
              f"{row['Frag']:5d} {row['D']:5d} {row['M']:5d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"videosA": VIDEOS_A, "videosB": VIDEOS_B, **out}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
