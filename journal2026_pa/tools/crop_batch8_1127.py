#!/usr/bin/env python3
"""The batch-8 retrain of crop checkpoint PathPlanning_4_s2 in place of the published one (ledger 10.28.17, R rule).

The published checkpoint was trained at batch 4: Ultralytics 8.4.46 halved the batch after a first-epoch CUDA OOM
and only warned (runs/rev_0920/logs/train_PathPlanning_4_s2.log), while the manuscript says batch 8 for every crop
detector. Retrained on 27 November at batch 8 on an idle card (same data, seed 2 and recipe; the two args.yaml
differ only in paths, device and batch; same Ultralytics 8.4.46 and torch 2.11), it changed e on PP4 by a median
|delta e| of 0.339 over the 20 (scale x interval) cells (results/pa_followup_readout_1127.json). The rule written
before the run (median below the seed-pair median 0.30 -> one sentence, numbers unchanged; otherwise replace the
checkpoint and recompute every result that uses crop seed 2) therefore replaces it.

  records       per-frame tracks of the 20 arms: runs/pa_followup_1124/r/records/r_s2__PathPlanning_4_s{s}_d{d}_none.json,
                written by tools/error_sources_1120.py replay-config --keep with the main-grid tracker
                (cfg/trackers/botsort_gmc.yaml) from the GPU dumps of the retrained checkpoint. The same replay of the
                published checkpoint's dump at 3072 reproduces its four archived live arms identity for identity
                (frame by frame, boxes to 0.0; checked 27 November).
  substitution  the 32-subset error substitution at sigma 3072: runs/pa_followup_1124/r/substitution/r_s2__PathPlanning_4_s3072*.json
                in place of runs/error_sources_1120/substitution/other/crop_s2__PathPlanning_4_s3072*.json.

Loaders call apply(REC) right after reading retrain.tar.gz, or items() when they keep their own dictionary, and read
substitution files through read_substitution(path), which keeps the published name for every other file.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/pa_followup_1124/r"
VIDEO, SEED = "PathPlanning_4", 2
SIGMAS, DELTAS = (1536, 2048, 2560, 3072, 3840), (1, 2, 4, 8)


def record(s, d):
    rec = json.loads((R / "records" / f"r_s{SEED}__{VIDEO}_s{s}_d{d}_none.json").read_text())["videos"][0]
    assert rec["video"] == VIDEO, rec["video"]
    return rec


def items(family="crop"):
    """((family, seed, video, sigma, delta), per-frame record) for the 20 replaced arms."""
    for s in SIGMAS:
        for d in DELTAS:
            yield (family, SEED, VIDEO, s, d), record(s, d)


def apply(REC, family="crop"):
    n = 0
    for k, rec in items(family):
        assert k in REC, k
        REC[k] = rec
        n += 1
    assert n == 20
    return REC


def substitution_path(path):
    p = Path(path)
    if p.name.startswith(f"crop_s{SEED}__{VIDEO}_s3072"):
        q = R / "substitution" / p.name.replace(f"crop_s{SEED}__", f"r_s{SEED}__", 1)
        assert q.exists(), q
        return q
    return p


def read_substitution(path):
    return json.loads(substitution_path(path).read_text())
