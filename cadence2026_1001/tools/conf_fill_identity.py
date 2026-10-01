"""IDF1 and HOTA for the confidence 0.75 and 0.80 rows of Table III.

Those rows were added in August with U, D, M and e only. Their per-frame outputs
(cached_<video>_conf075/080.json, six out-of-fold GrapeMOTS videos, the cohort of
every other row) are archived here, so the two identity columns are computed with
the code that produced the others: cbdcom2026_r3/tools/hota_panelA.build and
TrackEval 1.3.0's HOTA (mean over alpha) and Identity (IDF1). The 0.70 and 0.85
rows are recomputed first from cbdcom2026_r3/results/cached_conf and must equal the
published values, which proves the pipeline before it is used on the new rows.

    PYTHONPATH=<trackeval 1.3.0> python3 cadence2026_1001/tools/conf_fill_identity.py OUT.json
"""
import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("GRAPEMOTS_ROOT", str(ROOT / "cbdcom2026_r3"))
sys.path[:0] = [str(ROOT / "cbdcom2026_r3/tools")]
from decompose_count_error import decompose  # noqa: E402
from hota_panelA import VIDEOS, build  # noqa: E402
from trackeval.metrics import HOTA, Identity  # noqa: E402

DIRS = {"conf070": ROOT / "cbdcom2026_r3/results/cached_conf",
        "conf085": ROOT / "cbdcom2026_r3/results/cached_conf",
        "conf075": ROOT / "cadence2026_1001/results/conf_fill_identity",
        "conf080": ROOT / "cadence2026_1001/results/conf_fill_identity"}
PUBLISHED = json.loads((ROOT / "cbdcom2026_r3/results/hota_panelA.json").read_text())["rows"]


def row(token):
    entries = []
    for v in VIDEOS:
        entries += json.loads((DIRS[token] / f"cached_{v}_{token}.json").read_text())["videos"]
    t = Counter()
    for e in entries:
        one = decompose(e, 0.5, 1)
        assert one["identity_holds"]
        for k in "PGUDM":
            t[k] += one[k]
    data = build(entries)
    h = HOTA().eval_sequence(data)
    i = Identity().eval_sequence(data)
    return {**{k: int(t[k]) for k in "PGUDM"}, "signed_error": (t["P"] - t["G"]) / t["G"],
            "assigned_fraction": 1 - t["M"] / t["G"], "HOTA": float(np.mean(h["HOTA"])),
            "IDF1": float(i["IDF1"])}


out = {}
for token, label in (("conf070", "Confidence 0.70"), ("conf085", "Confidence 0.85")):
    r = row(token)
    assert abs(r["HOTA"] - PUBLISHED[label]["HOTA"]) < 1e-9, (label, r["HOTA"])
    out[label] = r
for token, label in (("conf075", "Confidence 0.75"), ("conf080", "Confidence 0.80")):
    out[label] = row(token)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
for k, r in out.items():
    print(f"{k}: P={r['P']} U={r['U']} D={r['D']} M={r['M']} e={r['signed_error']:+.3f} "
          f"cov={r['assigned_fraction']:.3f} IDF1={r['IDF1']:.3f} HOTA={r['HOTA']:.3f}")
