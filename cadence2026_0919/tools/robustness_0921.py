#!/usr/bin/env python3
"""Recompute the robustness table of the camera-ready paper from the frozen stubs.

Reads raw/{lovo_surface,phase,ret,cu,rt,bytetrack,strongsort,retrain,apple_val0000}.tar.gz
and rebuilds, with the paper's conventions (tau = 1, matching IoU 0.5, e computed
against the reference trajectories over all annotated frames):

  * one row per arm of Table I: steps that lowered the error (ties in brackets),
    sign-change rows and sequences, sequences whose sign change never moves to a
    denser interval, and how many of the 50 brackets differ from the published map;
  * the seed spread of the retrained detectors and how many cells change sign with
    the seed;
  * the AppleMOT grid read with the detector validated on sequence 0000.

    python3 tools/robustness_0921.py            # recompute and print
    python3 tools/robustness_0921.py --check    # and compare with results/expected_robustness.json

Requires numpy and scipy (scipy for the one-to-one matching). No imagery or GPU.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from decompose_count_error import HAVE_SCIPY, decompose  # noqa: E402

MV = [f"PathPlanning_{i}" for i in range(2, 9)]
FR = [f"NoPathPlanning_{i}" for i in range(1, 4)]
V = MV + FR
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
ORDER = {"<1": 0, "(1,2)": 1, "(2,4)": 2, "(4,8)": 3, ">8": 4}


def load(family: str) -> dict[str, dict]:
    out = {}
    with tarfile.open(ROOT / "raw" / f"{family}.tar.gz") as tar:
        for m in tar.getmembers():
            stub = json.load(io.TextIOWrapper(tar.extractfile(m)))
            terms = [decompose(v, 0.5, 1) for v in stub["videos"]]
            out[Path(m.name).name] = {k: sum(t[k] for t in terms) for k in ("P", "G", "U", "D", "M")}
    return out


def bracket(es):
    if es[0] < 0:
        return "<1"
    for i in range(3):
        if es[i] >= 0 > es[i + 1]:
            return f"({DELTAS[i]},{DELTAS[i + 1]})"
    return ">8"


def row(P, GF, base_map=None):
    """P(video, sigma, delta) -> predicted count."""
    strict = ties = rev = 0
    cmap, cov = {}, []
    terms = {}
    for v in V:
        for s in SIGMAS:
            es = []
            for d in DELTAS:
                t = P(v, s, d)
                terms[(v, s, d)] = t
                es.append((t["P"] - GF[v]) / GF[v])
            for a, b in zip(es, es[1:]):
                strict += b < a; ties += b == a; rev += b > a
            cmap[(v, s)] = bracket(es)
            if cmap[(v, s)].startswith("("):
                lo, hi = (int(x) for x in cmap[(v, s)][1:-1].split(","))
                for d in (lo, hi):
                    t = terms[(v, s, d)]
                    cov.append(1 - (t["M"] + GF[v] - t["G"]) / GF[v])
    ordered = sum(all(ORDER[cmap[(v, a)]] <= ORDER[cmap[(v, b)]] for a, b in zip(SIGMAS, SIGMAS[1:])) for v in V)
    out = dict(steps_down=strict, ties=ties, reversals=rev,
               sign_change_rows=sum(1 for x in cmap.values() if x.startswith("(")),
               sign_change_sequences=len({v for (v, s), x in cmap.items() if x.startswith("(")}),
               ordered_sequences=ordered,
               coverage_max=round(max(cov), 4) if cov else None,
               coverage_median=round(statistics.median(cov), 3) if cov else None)
    if base_map is not None:
        out["brackets_differing"] = sum(cmap[k] != base_map[k] for k in cmap)
    return out, cmap


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not HAVE_SCIPY:
        sys.exit("scipy is required for the one-to-one matching the paper uses")

    base = load("lovo_surface")
    GF = {v: base[f"{v}_s3072_d1.json"]["G"] for v in V}
    N = {}
    N["surface"], base_map = row(lambda v, s, d: base[f"{v}_s{s}_d{d}.json"], GF)
    for fam in ("ret", "cu", "rt"):
        data = load(fam)
        N[fam], _ = row(lambda v, s, d, data=data: data[f"{v}_s{s}_d{d}.json"] if d > 1 else base[f"{v}_s{s}_d1.json"],
                        GF, base_map)
    phase = load("phase")
    cells = {(v, s, d): [base[f"{v}_s{s}_d{d}.json"]["P"]] +
                        [phase[f"{v}_s{s}_d{d}_o{k}.json"]["P"] for k in range(1, d)]
             for v in V for s in SIGMAS for d in DELTAS[1:]}
    for tag, pick in (("phase_low", min), ("phase_high", max), ("phase_mean", statistics.mean)):
        N[tag], _ = row(lambda v, s, d, pick=pick: {"P": pick(cells[(v, s, d)]), "G": base[f"{v}_s{s}_d{d}.json"]["G"],
                                                    "M": base[f"{v}_s{s}_d{d}.json"]["M"],
                                                    "U": 0, "D": 0} if d > 1 else base[f"{v}_s{s}_d1.json"],
                        GF, base_map)
    for fam in ("bytetrack", "strongsort"):
        data = load(fam)
        N[fam], _ = row(lambda v, s, d, data=data: data[f"{v}_s{s}_d{d}.json"], GF)
    retr = load("retrain")
    def rp(v, s, d, seed):
        key = f"{v}_s{seed}__{v}_s{s}_d{d}.json"
        return retr[key] if key in retr else base[f"{v}_s{s}_d{d}.json"]
    for seed in (0, 1, 2):
        N[f"retrain_seed{seed}"], _ = row(lambda v, s, d, seed=seed: rp(v, s, d, seed), GF, base_map)
    e = lambda v, s, d, seed: (rp(v, s, d, seed)["P"] - GF[v]) / GF[v]
    sds = [statistics.pstdev([e(v, s, d, k) for k in (0, 1, 2)]) for v in V for s in SIGMAS for d in DELTAS]
    N["retrain_seed_spread"] = dict(
        median_sd=round(statistics.median(sds), 3), max_sd=round(max(sds), 3),
        sign_flip_cells=sum(1 for v in V for s in SIGMAS for d in DELTAS
                            if len({e(v, s, d, k) >= 0 for k in (0, 1, 2)}) > 1),
        frontal_median_sd=round(statistics.median(
            [statistics.pstdev([e(v, s, d, k) for k in (0, 1, 2)]) for v in FR for s in SIGMAS for d in DELTAS]), 3))
    apple = load("apple_val0000")
    a_e = {}
    for s in (640, 960, 1280):
        g1 = apple[f"apple_s{s}_d1.json"]["G"]
        for d in DELTAS:
            a_e[f"{s}/{d}"] = round((apple[f"apple_s{s}_d{d}.json"]["P"] - g1) / g1, 3)
    N["apple_val0000"] = a_e

    (ROOT / "results" / "robustness_numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True))
    for k, v in N.items():
        print(f"{k:20s} {v}")
    if args.check:
        exp = json.loads((ROOT / "results" / "expected_robustness.json").read_text())
        bad = [k for k, v in exp.items() if json.loads(json.dumps(N.get(k))) != v]
        if bad:
            sys.exit(f"DRIFT in {bad}")
        print(f"all {len(exp)} expected entries match")


if __name__ == "__main__":
    main()
