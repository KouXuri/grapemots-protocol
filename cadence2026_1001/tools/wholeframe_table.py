"""Table I row "Whole frame": the 2021 cadence intervention with the detector reading
each 4096x2160 frame in one letterboxed pass instead of eight 1280 tiles.

Run from the archive root:
    python3 cadence2026_1001/tools/wholeframe_table.py RAW_DIR OUT.json

RAW_DIR holds the per-sequence outputs of tools/fullrate_decompose_1001.py
(raw/wholeframe_2021.tar.gz): whole_{unseen,seen}_<seq>.json from run_wholeframe.sh
and tiledmac_unseen_<seq>.json from run_tiledmac.sh, the frozen tiled read re-run on
the same machine so the two rows differ in read mode alone. Statistics as every
other Table I row (cadence2026_1001/tools/align_sensitivity.py): per-sequence e
from definition_sensitivity_0815.per_video over the archived decomposition,
medians over sequences, median paired difference, percentile bootstrap over
sequences with 10,000 resamples and seed 0.
"""
import json, random, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cadence2026_0919/tools"), str(ROOT / "cadence2026_0813/tools")]
from definition_sensitivity_0815 import per_video  # noqa: E402

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
UNSEEN = ["row_4.3_2", "row_4.4_2", "row_4.4_4", "row_6.1_3", "row_6.1_4", "row_6.2_1",
          "row_6.2_2", "row_7.1_3", "row_7.1_4", "row_7.2_1", "row_7.2_2",
          "row_4.2_1", "row_6.1_1", "row_6.1_2", "row_7.1_1", "row_7.1_2", "row_8_1"]
SEEN = ["row_7.2_3", "row_7.2_4", "row_7.3_1", "row_7.3_2", "row_7.3_3", "row_7.3_4",
        "row_7.4_1", "row_7.4_2", "row_8_2", "row_8_3", "row_8_4"]
ARMS = {"rel_buf30": "rel", "src_buf30": "src"}


def load(prefix, group, names):
    runs = []
    for v in names:
        path = RAW / f"{prefix}_{group}_{v}.json"
        if not path.is_file():
            return None
        for r in json.loads(path.read_text())["runs"]:
            if r["arm"] in ARMS:
                runs.append(dict(r, arm=ARMS[r["arm"]]))
    return runs


def summarise(runs, tau=1):
    per, pooled = {}, {a: dict(P=0, G=0, U=0, D=0, M=0) for a in ARMS.values()}
    for r in runs:
        v = per_video(r, 0.5, tau)
        for k in pooled[r["arm"]]:
            pooled[r["arm"]][k] += v[k]
        per.setdefault(r["video"], {})[r["arm"]] = (v["P"] - v["G"]) / v["G"]
    names = sorted(per)
    d = [per[n]["src"] - per[n]["rel"] for n in names]
    rng = random.Random(0)
    boots = sorted(statistics.median(rng.choices(d, k=len(d))) for _ in range(10000))
    up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
    out = {
        "sequences": len(names),
        "sparse_median": round(statistics.median(per[n]["rel"] for n in names), 3),
        "source_median": round(statistics.median(per[n]["src"] for n in names), 3),
        "delta_median": statistics.median(d),
        "ci95": [round(boots[249], 2), round(boots[9749], 2)],
        "up/down/tie": f"{up}/{down}/{len(d) - up - down}",
        "per_sequence_e": {n: [round(per[n]["rel"], 4), round(per[n]["src"], 4)] for n in names},
    }
    for a, t in pooled.items():
        out[f"pooled_{a}"] = dict(t, e=round((t["P"] - t["G"]) / t["G"], 4),
                                  coverage=round(1 - t["M"] / t["G"], 4),
                                  identity=t["P"] - t["G"] == t["U"] + t["D"] - t["M"])
    return out


result = {}
for prefix, label in (("whole", "whole frame, imgsz 4096"), ("tiledmac", "8 tiles, Mac re-run")):
    for group, names in (("unseen", UNSEEN), ("seen", SEEN)):
        runs = load(prefix, group, names)
        if runs is not None:
            result[f"{label} / {group}"] = {f"tau={t}": summarise(runs, t) for t in (1, 3)}
OUT.write_text(json.dumps(result, indent=1) + "\n")
for k, v in result.items():
    one = v["tau=1"]
    print(k, {x: one[x] for x in ("sequences", "sparse_median", "source_median",
                                   "delta_median", "ci95", "up/down/tie")})
