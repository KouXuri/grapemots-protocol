#!/usr/bin/env python3
"""What U, D and M say that IDF1 and the CLEAR family do not.

A reviewer asked the obvious question about the decomposition: is D another name
for fragmentation, M another name for mostly-lost, U another name for false
positive tracks, and if so why not report IDF1 and be done. The answer is in the
same frozen per-frame boxes the decomposition is read from, so it can be measured
instead of argued.

D and M are trajectory-level and defined by ownership: a duplicate is a second
predicted identity that owns a trajectory some other identity already owns, and
an unassigned trajectory is one no predicted track owns even if boxes overlapped
it. An identity switch is frame-level and needs the same track to change which
trajectory it covers, which a duplicate never has to do. This script prints them
side by side on the two cadence arms.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from decompose_count_error import decompose  # noqa: E402
from hota_panelA import build, iou_matrix  # noqa: E402  (iou_matrix used by build)

from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=["rel", "src"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = []
    for path in args.arms:
        payload = json.loads(path.read_text())
        records.extend(r for r in payload["runs"] if "frame_predicted_boxes" in r)

    identity, clear, hota = Identity(), CLEAR(), HOTA()
    out = {}
    print(f"{'arm':>8s} {'P':>5s} {'G':>5s} {'U':>5s} {'D':>5s} {'M':>5s} {'e':>8s} "
          f"{'1-M/G':>7s} {'IDF1':>7s} {'IDP':>7s} {'IDR':>7s} {'MT':>4s} {'PT':>4s} "
          f"{'ML':>4s} {'IDSW':>5s} {'Frag':>5s} {'HOTA':>7s}")
    for label in args.labels:
        entries = [r for r in records if r["arm"] == label]
        if not entries:
            continue
        terms = Counter()
        for entry in entries:
            one = decompose(entry, 0.5, 1)
            if not one["identity_holds"]:
                raise SystemExit(f"{label}/{entry['video']}: P-G != U+D-M")
            for key in ("P", "G", "U", "D", "M"):
                terms[key] += one[key]
        data = build(entries)
        idr = identity.eval_sequence(data)
        clr = clear.eval_sequence(data)
        hot = hota.eval_sequence(data)
        row = {
            "videos": len(entries),
            "P": int(terms["P"]), "G": int(terms["G"]), "U": int(terms["U"]),
            "D": int(terms["D"]), "M": int(terms["M"]),
            "signed_error": (terms["P"] - terms["G"]) / terms["G"],
            "assigned_fraction": 1 - terms["M"] / terms["G"],
            "IDF1": float(idr["IDF1"]), "IDP": float(idr["IDP"]),
            "IDR": float(idr["IDR"]),
            "IDTP": int(idr["IDTP"]), "IDFP": int(idr["IDFP"]), "IDFN": int(idr["IDFN"]),
            "MT": int(clr["MT"]), "PT": int(clr["PT"]), "ML": int(clr["ML"]),
            "IDSW": int(clr["IDSW"]), "Frag": int(clr["Frag"]),
            "MOTA": float(clr["MOTA"]),
            "HOTA": float(np.mean(hot["HOTA"])),
            "AssA": float(np.mean(hot["AssA"])),
        }
        out[label] = row
        print(f"{label:>8s} {row['P']:5d} {row['G']:5d} {row['U']:5d} {row['D']:5d} "
              f"{row['M']:5d} {row['signed_error']:+8.4f} {row['assigned_fraction']:7.4f} "
              f"{row['IDF1']:7.4f} {row['IDP']:7.4f} {row['IDR']:7.4f} {row['MT']:4d} "
              f"{row['PT']:4d} {row['ML']:4d} {row['IDSW']:5d} {row['Frag']:5d} "
              f"{row['HOTA']:7.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": out}, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
