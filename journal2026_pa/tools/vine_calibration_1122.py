#!/usr/bin/env python3
"""Vine-level calibration with detectors that saw neither the target vine nor its calibration vines.

Run: runs/pa_vine_calibration_1122 (runner/queue_vine_calibration_1122.py on raptor, 2026-11-22/23).  For each
of PP2-PP8 as the target, one whole-frame YOLO26s (seed 0, the recipe of the paper's whole-frame folds) was
trained on PP1 and three other test vines, its checkpoint selected on a fourth, and it counted the target and
the two remaining vines at sigma 3072 (BoT-SORT replay with cached GMC, intervals 1 and 8).  A ratio fitted on
the two calibration vines is applied to the target; the mean reference count of the two is the no-video
baseline.  Matched control: the same targets and the same two calibration vines with the cross-validated
seed-0 whole-frame counts of the paper (runs/fullframe_pponly_0922, sigma 3072), whose detectors had been
trained or validated on the target.  Frozen outputs only; every stored estimate is recomputed and checked.
    .venv/bin/python cadence_journal_1119/tools/vine_calibration_1122.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grapemots-protocol/cadence2026_0919/tools"))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

assert HAVE_SCIPY, "SciPy is required"
RUN = ROOT / "runs/pa_vine_calibration_1122"
PAPER = ROOT / "runs/fullframe_pponly_0922/surface_seed0"
OUT = ROOT / "cadence_journal_1119/results/vine_calibration_1122.json"
TEST = [f"PathPlanning_{i}" for i in range(2, 9)]
ALL = ["PathPlanning_1", *TEST]
DELTAS = (1, 8)


def q(xs, p):  # as calibration_transfer_1120.py
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summ(signed):
    a = [abs(x) for x in signed]
    return dict(n=len(a), median=st.median(a), p90=q(a, 0.9), max=max(a), signed=signed)


S = json.loads((RUN / "summary.json").read_text())
plan = json.loads((RUN / "plan.json").read_text())
assert S["plan"] == plan and plan["seed"] == 0 and plan["scale"] == 3072 and plan["intervals"] == list(DELTAS)
assert plan["confidence"] == 0.25 and plan["minimum_track_frames"] == 1
assert [f["test"] for f in plan["folds"]] == TEST and len(S["vines"]) == 7

# ------------------------------------------------------------------ paper counts (cross-validated), sigma 3072
P0, G = {}, {}
for v in TEST:
    for d in DELTAS:
        t = decompose(json.loads((PAPER / f"{v}_s3072_d{d}.json").read_text())["videos"][0], 0.5, 1)
        assert t["identity_holds"]
        P0[(v, d)] = t["P"]
        if d == 1:
            G[v] = t["G"]

folds, rows = [], {k: [] for k in (
    "raw_d1", "raw_d8", "ratio_d1", "ratio_d8", "transfer_d1_to_d8", "mean_two",
    "xval_raw_d1", "xval_raw_d8", "xval_ratio_d1", "xval_ratio_d8", "xval_transfer_d1_to_d8",
    "xval_ratio_six_d1", "xval_ratio_six_d8", "mean_six")}
for fold, r in zip(plan["folds"], S["vines"]):
    tv, cal, val, tr = fold["test"], fold["calibration"], fold["validation"], fold["training"]
    roles = [*tr, val, *cal, tv]
    assert r["split"] == fold and len(tr) == 4 and tr[0] == "PathPlanning_1" and len(cal) == 2
    assert len(set(roles)) == 8 and set(roles) == set(ALL), fold
    trained = json.loads((RUN / "folds" / tv / "trained.json").read_text())
    assert trained["weights"] == r["checkpoint"] and trained["args"]["seed"] == 0 and trained["args"]["imgsz"] == 3072
    # counts: reference fixed over intervals, decomposition identity, same G as the paper's records
    recs = [r["test"], *r["donors"]]
    assert [x["vine"] for x in recs] == [tv, *cal]
    for x in recs:
        assert x["G"] == G[x["vine"]], (x["vine"], x["G"], G[x["vine"]])
        stored = json.loads((RUN / "folds" / tv / f"{x['vine']}_counts.json").read_text())
        assert stored == x
        for d in DELTAS:
            c = x["intervals"][str(d)]
            assert c["P"] - x["G"] == c["U"] + c["D"] - c["M"]
    # recompute the stored calibration exactly
    gt = r["test"]["G"]
    Pt = {d: r["test"]["intervals"][str(d)]["P"] for d in DELTAS}
    Pc = {d: sum(x["intervals"][str(d)]["P"] for x in r["donors"]) for d in DELTAS}
    Gc = sum(x["G"] for x in r["donors"])
    k = {d: Gc / Pc[d] for d in DELTAS}
    mine = dict(raw_d1=Pt[1] / gt - 1, raw_d8=Pt[8] / gt - 1, ratio_d1=k[1] * Pt[1] / gt - 1,
                ratio_d8=k[8] * Pt[8] / gt - 1, transfer_d1_to_d8=k[1] * Pt[8] / gt - 1,
                mean_two=Gc / 2 / gt - 1)
    e = r["signed_relative_errors"]
    stored = dict(raw_d1=e["raw_d1"], raw_d8=e["raw_d8"], ratio_d1=e["calibrated_d1"],
                  ratio_d8=e["calibrated_d8"], transfer_d1_to_d8=e["transfer_d1_to_d8"], mean_two=e["no_video_mean"])
    assert all(abs(mine[n] - stored[n]) < 1e-12 for n in mine), (tv, mine, stored)
    # matched control: the paper's cross-validated counts, same target, same two calibration vines
    kx = {d: Gc / sum(P0[(u, d)] for u in cal) for d in DELTAS}
    others = [u for u in TEST if u != tv]
    k6 = {d: sum(G[u] for u in others) / sum(P0[(u, d)] for u in others) for d in DELTAS}
    mine.update(xval_raw_d1=P0[(tv, 1)] / gt - 1, xval_raw_d8=P0[(tv, 8)] / gt - 1,
                xval_ratio_d1=kx[1] * P0[(tv, 1)] / gt - 1, xval_ratio_d8=kx[8] * P0[(tv, 8)] / gt - 1,
                xval_transfer_d1_to_d8=kx[1] * P0[(tv, 8)] / gt - 1,
                xval_ratio_six_d1=k6[1] * P0[(tv, 1)] / gt - 1, xval_ratio_six_d8=k6[8] * P0[(tv, 8)] / gt - 1,
                mean_six=st.fmean(G[u] for u in others) / gt - 1)
    for n in rows:
        rows[n].append(mine[n])
    folds.append(dict(target=tv, calibration=cal, validation=val, training=tr, G=gt,
                      train_frames=fold["train_frames"], validation_frames=fold["validation_frames"],
                      epochs=trained["epochs"], validation_map50_95=trained["validation_map50_95"],
                      P_independent={str(d): Pt[d] for d in DELTAS},
                      P_cross_validated={str(d): P0[(tv, d)] for d in DELTAS},
                      ratio_independent={str(d): k[d] for d in DELTAS},
                      ratio_cross_validated={str(d): kx[d] for d in DELTAS},
                      attempts=sorted(p.stem for p in (RUN / "folds" / tv).glob("attempt_*.json"))))

out = dict(design=dict(targets=len(folds), seed=0, sigma=3072, intervals=list(DELTAS), training_vines=4,
                       validation_vines=1, calibration_vines=2, confidence=0.25, tracker=plan["tracker"],
                       source_manifest_sha256=plan["source_manifest_sha256"]),
           folds=folds, metrics={n: summ(v) for n, v in rows.items()})
OUT.write_text(json.dumps(out, indent=1) + "\n")
for n, m in out["metrics"].items():
    print(f"{n:24s} median {m['median']:.3f}  p90 {m['p90']:.3f}  max {m['max']:.3f}")
