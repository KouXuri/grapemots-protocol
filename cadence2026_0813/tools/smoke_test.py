#!/usr/bin/env python3
"""Check this bundle from a fresh checkout, without imagery, weights or a GPU.

Three things are verified:

1. every file in SHA256SUMS is present and hashes to what it says;
2. every table value the manuscript reports from this round is *recomputed* from
   the frozen results here and compared with `results/expected_tables.json`,
   which holds those values as the paper typesets them. A drift in either
   direction fails, so a table cannot quietly stop matching its source;
3. what this bundle does not carry is named, so the boundary between
   "auditable from frozen outputs" and "rebuildable from released inputs" is
   visible rather than implied.

    python3 tools/smoke_test.py

Requires Python 3.9+ and nothing else. The expected-value manifest is the same
set of numbers `tools/verify_tables_0813.py` matches against the manuscript
source, so the two checks close the loop from result file to printed table.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CADENCE0813_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"
SUMS = ROOT / "SHA256SUMS"

EXT = RESULTS / "ext_cadence_0813"
ARMS = RESULTS / "adaptive_0813"
DEC = RESULTS / "decomp_0812"

NOT_CARRIED = [
    "the source videos and 4K imagery (GrapeMOTS is CC-BY from its own archive)",
    "the MOT17/MOT20 image sequences (redistributed by their own benchmarks)",
    "detector checkpoints, and the per-frame detection caches they produce",
    "the GPU the on-board cost table was timed on (RTX 2000 Ada, desktop)",
]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def check_sums() -> int:
    if not SUMS.is_file():
        print(f"FAIL  no SHA256SUMS at {SUMS}")
        return 1
    bad = missing = checked = 0
    for line in SUMS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        want, _, name = line.partition("  ")
        path = ROOT / name.strip()
        if not path.is_file():
            print(f"MISSING  {name}")
            missing += 1
            continue
        checked += 1
        if digest(path) != want:
            print(f"MISMATCH {name}")
            bad += 1
    print(f"{checked} files hashed, {bad} mismatched, {missing} missing")
    return 1 if (bad or missing) else 0


def rebuild() -> dict:
    """Recompute every table value this round contributes, from frozen results."""
    out: dict[str, dict] = {}

    # Table IV: the cadence contrast on densely annotated corpora.
    for corpus in ("mot17", "mot20"):
        cadence = load(EXT / f"cadence_{corpus}.json")
        geometry = load(EXT / f"geometry_{corpus}.json")
        if not cadence or not geometry:
            continue
        for k in (1, 4, 15, 60):
            rel = cadence["pooled"][f"k={k}|released|bytetrack"]["1"]
            src = cadence["pooled"][f"k={k}|source|bytetrack"]["1"]
            out[f"tableIV/{corpus}/k={k}"] = {
                "r": round(geometry["by_step"][str(k)]["sequence_median_r"], 3),
                "e_released": round(rel["signed_error"], 3),
                "e_source": round(src["signed_error"], 3),
                "assigned_released": round(rel["assigned_fraction"], 3),
                "D": rel["D"],
                "M": rel["M"],
            }

    # Section III-C: one frame budget spent uniformly or by frame difference.
    pooled: dict[str, dict[str, int]] = {}
    for name in ("arms_fold1_six.json", "arms_fold2_eleven.json"):
        payload = load(ARMS / name)
        if not payload:
            continue
        for arm, cells in payload["pooled"].items():
            one = cells["1"]
            entry = pooled.setdefault(arm, dict.fromkeys(("P", "G", "M", "tracker_frames"), 0))
            for key in entry:
                entry[key] += one[key]
    for arm in ("rel", "uni2", "ada2", "uni4", "ada4", "uni8", "ada8", "src"):
        if arm not in pooled:
            continue
        block = pooled[arm]
        out[f"sec3C-adaptive/{arm}"] = {
            "frames": block["tracker_frames"],
            "e": round((block["P"] - block["G"]) / block["G"], 3),
            "assigned": round(1 - block["M"] / block["G"], 3),
        }

    # Table VI: the two cuts priced, on board and on the link.
    for name, label in (("edge_yolo26s_tiled.json", "yolo26s_8tiles"),
                        ("edge_yolo26n_tiled.json", "yolo26n_8tiles"),
                        ("edge_yolo26n_full.json", "yolo26n_full")):
        payload = load(ARMS / name)
        if not payload:
            continue
        compute = payload["compute"]
        out[f"tableVI/{label}"] = {
            "fps": round(compute["fps"], 1),
            "detect_ms": round(compute["stage_ms"]["detect_ms_median"]),
            "associate_ms": round(compute["stage_ms"]["track_ms_median"]),
            "joules_per_frame": round(compute["joules_per_frame"], 1),
        }
    # The link row is measured through one x264 encoder at CRF 23, so the two
    # frame architectures differ in the frames sent and in nothing else. The
    # delivered bitrate of the release is kept beside them because it is a
    # different encode, not a third architecture.
    link = load(RESULTS / "link_allintra_0814" / "link_allintra.json")
    if link:
        out["tableVI/link"] = {
            "delivered_mbit_s": round(link["source_mbit_s"], 1),
            "every_frame_mbit_s": round(link["fullrate_same_codec_mbit_s"], 1),
            "sparse_mbit_s": round(link["allintra_mbit_s_at_sparse_rate"], 1),
            "sparse_interframe_mbit_s": round(link["interframe_mbit_s_at_sparse_rate"], 1),
            "frame_saving": round(link["frame_saving"], 3),
            "byte_saving_same_codec": round(link["byte_saving_same_codec"], 2),
        }

    # Table II: IDF1 replaces AssA, recomputed from the same per-frame boxes.
    idf1 = load(RESULTS / "definition_0815" / "idf1_table2.json")
    # the manuscript tabulates nine of the eleven association rows in panel A
    PANEL_A_ROWS = ("Confidence 0.85", "Confidence 0.70", "Confidence 0.55",
                    "Confidence 0.40", "IoS merge", "BoT-SORT, buffer 30",
                    "BoT-SORT, GMC off", "BoT-SORT + ReID", "ByteTrack, buffer 60")
    if idf1:
        for label, row in idf1["panelA"].items():
            if label in PANEL_A_ROWS:
                out[f"tableII-A-idf1/{label}"] = {"IDF1": round(row["IDF1"], 3)}
        for label, row in idf1["panelB"].items():
            out[f"tableII-B-idf1/{label}"] = {"IDF1": round(row["IDF1"], 3)}

    # The decomposition read at three ownership gates: the count does not move
    # with the gate, the coverage does, and the ordering of the arms does not.
    sensitivity = load(RESULTS / "definition_0815" / "definition_sensitivity.json")
    if sensitivity:
        for gate in ("0.3", "0.5", "0.7"):
            for arm in ("rel", "src"):
                cell = sensitivity["gate"].get(f"iou{gate}_tau1_{arm}")
                if cell:
                    out[f"ownership-gate/iou{gate}/{arm}"] = {
                        "e": round(cell["signed_error"], 3),
                        "assigned": round(cell["assigned_fraction"], 2),
                    }
        for tau in ("3",):
            for arm in ("rel", "src"):
                cell = sensitivity["tau_symmetry"].get(f"tau{tau}_{arm}")
                if cell:
                    out[f"symmetric-tau/tau{tau}/{arm}"] = {
                        "G_symmetric": cell["G_symmetric"],
                        "e_symmetric": round(cell["e_symmetric"], 3),
                    }

    # U, D and M beside the identity family on the same two arms.
    identity = load(RESULTS / "definition_0815" / "identity_metrics.json")
    if identity:
        for arm, row in identity["rows"].items():
            out[f"identity-family/{arm}"] = {
                "D": row["D"], "M": row["M"], "IDSW": row["IDSW"],
                "ML": row["ML"], "IDF1": round(row["IDF1"], 3),
            }

    # The timescale and gate controls on the cadence arms.
    timescale = load(RESULTS / "timescale_0815" / "timescale_summary.json")
    if timescale:
        for key, cell in timescale["pairs"].items():
            if not key.endswith("/tau1"):
                continue
            out[f"timescale/{key}"] = {
                "delta_median": round(cell["delta_median"], 3),
                "up": cell["up"], "down": cell["down"], "tie": cell["tie"],
            }

    # Table II panel B: eleven sequences, each read by a checkpoint blind to it.
    panel_b = load(DEC / "hota_panelB.json")
    if panel_b:
        for label, row in panel_b["rows"].items():
            out[f"tableII-B/{label}"] = {
                key: (round(row[key], 3) if isinstance(row[key], float) else row[key])
                for key in ("U", "D", "M", "HOTA", "AssA")
            }

    # Flight-clustered bootstrap: the interval the manuscript quotes at tau=1.
    clusters = load(DEC / "cluster_bootstrap.json")
    if clusters:
        tau1 = clusters["by_tau"]["1"]
        out["flights/tau=1"] = {
            "sequence_level_median_delta": round(tau1["sequence_level_median_delta"], 3),
            "ci_low": round(tau1["cluster_bootstrap_ci95"][0], 2),
            "ci_high": round(tau1["cluster_bootstrap_ci95"][1], 2),
            "flights": tau1["flight_count"],
            "flights_all_positive": tau1["flights_all_positive"],
        }

    # The low-score second stage, at the floor that actually produces low-score
    # candidates: what it was offered, and what it accepted.
    offered = accepted = 0
    for path in sorted(ARMS.glob("second_stage_*.json")):
        payload = load(path)
        for name, arm in payload["arms"].items():
            if not name.endswith("0.1"):
                continue
            second = arm["stages"]["second"]
            offered += second["candidates_offered"]
            accepted += second["assignments_returned"]
    out["second_stage"] = {"candidates_offered": offered, "accepted": accepted}
    return out


def check_tables() -> int:
    expected_path = RESULTS / "expected_tables.json"
    if not expected_path.is_file():
        print(f"FAIL  no expected_tables.json at {expected_path}")
        return 1
    expected = json.loads(expected_path.read_text())
    got = rebuild()
    bad = 0
    for key, want in expected.items():
        if key not in got:
            print(f"MISSING  {key} could not be rebuilt from the frozen results")
            bad += 1
            continue
        if got[key] != want:
            print(f"DRIFT    {key}\n           paper: {want}\n           rebuilt: {got[key]}")
            bad += 1
    for key in got:
        if key not in expected:
            print(f"EXTRA    {key} rebuilt but not claimed by the paper")
            bad += 1
    print(f"{len(expected)} table entries rebuilt from frozen results, {bad} disagreeing")
    return 1 if bad else 0


def main() -> int:
    status = 0
    print("== SHA-256 ==")
    status |= check_sums()
    print("\n== tables rebuilt from the frozen results ==")
    status |= check_tables()
    print("\n== not carried here ==")
    for item in NOT_CARRIED:
        print(f"  - {item}")
    print("\nOK" if status == 0 else "\nFAILED")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
