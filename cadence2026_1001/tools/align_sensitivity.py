"""Re-score the 2021 cadence intervention without the labelled frames whose source
alignment the 2026-09-22 full-resolution audit could not certify.

Run from the archive root:  python3 cadence2026_1001/tools/align_sensitivity.py OUT.json
Inputs: the audit summary (results/bodegas_alignment_audit/summary.json), the frozen
per-frame outputs of both arms (cadence2026_0813/results/adaptive_0813/), and the
archived decomposition (cadence2026_0919/tools/decompose_count_error.py) through
cadence2026_0813/tools/definition_sensitivity_0815.py."""
import json, random, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cadence2026_0919/tools"), str(ROOT / "cadence2026_0813/tools")]
from definition_sensitivity_0815 import per_video  # noqa: E402

audit = json.loads((ROOT / "cadence2026_1001/results/bodegas_alignment_audit/summary.json").read_text())["source_alignment"]
bad = {(f["sequence"], f["image"].replace(".png", "")) for f in audit["large_residual_frames"]}
runs = []
for f in ("arms_fold1_six.json", "arms_fold2_eleven.json"):
    runs += [r for r in json.loads((ROOT / "cadence2026_0813/results/adaptive_0813" / f).read_text())["runs"]
             if r["arm"] in ("rel", "src")]
PER_FRAME = ("frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes", "frame_names")

def drop(rec):
    keep = [i for i, n in enumerate(rec["frame_names"])
            if (rec["video"], n.split("__frame_")[1].replace(".txt", "")) not in bad]
    out = dict(rec)
    for k in PER_FRAME:
        out[k] = [rec[k][i] for i in keep]
    return out, len(rec["frame_names"]) - len(keep)

res = {}
for label, fn in (("all frames", lambda r: (r, 0)), ("suspect frames dropped", drop)):
    pooled = {a: dict(P=0, G=0, U=0, D=0, M=0) for a in ("rel", "src")}
    per, removed = {}, 0
    for r in runs:
        rec, n = fn(r); removed += n if r["arm"] == "rel" else 0
        v = per_video(rec, 0.5, 1)
        for k in pooled[r["arm"]]: pooled[r["arm"]][k] += v[k]
        per.setdefault(r["video"], {})[r["arm"]] = (v["P"] - v["G"]) / v["G"]
    delta = [per[v]["src"] - per[v]["rel"] for v in per]
    up = sum(d > 0 for d in delta); down = sum(d < 0 for d in delta)
    out = {}
    for a, t in pooled.items():
        out[a] = dict(t, e=round((t["P"] - t["G"]) / t["G"], 4), coverage=round(1 - t["M"] / t["G"], 4),
                      identity=t["P"] - t["G"] == t["U"] + t["D"] - t["M"])
    out["paired_median"] = round(statistics.median(delta), 4)
    out["up/down/tie"] = f"{up}/{down}/{len(delta)-up-down}"
    out["frames_removed_per_arm"] = removed
    out["sequences"] = len(per)
    res[label] = out

def table_row(fn):
    per = {}
    for r in runs:
        rec, _ = fn(r)
        v = per_video(rec, 0.5, 1)
        per.setdefault(r["video"], {})[r["arm"]] = (v["P"] - v["G"]) / v["G"]
    names = sorted(per)
    d = [per[n]["src"] - per[n]["rel"] for n in names]
    rng = random.Random(0)
    boots = sorted(statistics.median(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"sparse_median": round(statistics.median(per[n]["rel"] for n in names), 3),
            "source_median": round(statistics.median(per[n]["src"] for n in names), 3),
            "delta_median": statistics.median(d),
            "ci95": [round(boots[249], 2), round(boots[9749], 2)]}

res["table_ii_rows"] = {"all frames": table_row(lambda r: (r, 0)), "suspect frames dropped": table_row(drop)}
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print(json.dumps(res["table_ii_rows"]))
