#!/usr/bin/env python3
"""Check every table cell added in the 08-13 round against the file it came from.

tools/verify_cbdcom_paper.py checks structure and the two tables that predate
this round. This checks the rest, cell by cell, so a table cannot drift from its
source without the check failing: the external cadence contrast, the adaptive
sampling arms, the on-board and link costs, and Panel B of the configuration
table. Numbers are matched as they are typeset, so a changed rounding shows up.

    python3 tools/verify_tables_0813.py [paper.tex]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cbdcom2026_paper3_EN_2026-08-14.tex"
EXT = ROOT / "runs/ext_cadence_0813/results"
ARMS = ROOT / "runs/adaptive_0813/results"
DEC = ROOT / "runs/decomp_0812/results"


def load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> int:
    tex = PAPER.read_text()
    bad: list[str] = []
    checked = 0

    # ---- external cadence contrast -----------------------------------------
    for corpus, rows in (("mot17", [1, 4, 15, 60]), ("mot20", [1, 4, 15, 60])):
        cadence = load(EXT / f"cadence_{corpus}.json")
        geometry = load(EXT / f"geometry_{corpus}.json")
        if not cadence or not geometry:
            bad.append(f"missing results for {corpus}")
            continue
        for k in rows:
            rel = cadence["pooled"][f"k={k}|released|bytetrack"]["1"]
            src = cadence["pooled"][f"k={k}|source|bytetrack"]["1"]
            r = geometry["by_step"][str(k)]["sequence_median_r"]
            cells = [f"${r:.3f}$", f"${rel['signed_error']:+.3f}$",
                     f"${src['signed_error']:+.3f}$",
                     f"${rel['assigned_fraction']:.3f}$"]
            for cell in cells:
                checked += 1
                if cell not in tex:
                    bad.append(f"{corpus} k={k}: {cell} not in the paper")
            for term in ("D", "M"):
                checked += 1
                value = rel[term]
                shown = f"{value:,}".replace(",", "{,}") if value >= 1000 else str(value)
                if shown not in tex:
                    bad.append(f"{corpus} k={k}: {term}={shown} not in the paper")

    # ---- adaptive vs uniform arms ------------------------------------------
    pooled: dict[str, dict[str, int]] = {}
    for name in ("arms_fold1_six.json", "arms_fold2_eleven.json"):
        payload = load(ARMS / name)
        if not payload:
            bad.append(f"missing {name}")
            continue
        for arm, cells in payload["pooled"].items():
            one = cells["1"]
            entry = pooled.setdefault(arm, {k: 0 for k in ("P", "G", "M", "tracker_frames")})
            for key in entry:
                entry[key] += one[key]
    for arm in ("rel", "uni2", "ada2", "uni4", "ada4", "uni8", "ada8", "src"):
        if arm not in pooled:
            bad.append(f"arm {arm} missing from the pooled results")
            continue
        block = pooled[arm]
        e = (block["P"] - block["G"]) / block["G"]
        assigned = 1 - block["M"] / block["G"]
        for cell in (f"${e:+.3f}$", f"${assigned:.3f}$"):
            checked += 1
            if cell not in tex:
                bad.append(f"adaptive table, arm {arm}: {cell} not in the paper")
        frames = block["tracker_frames"]
        shown = f"{frames:,}".replace(",", "{,}") if frames >= 1000 else str(frames)
        checked += 1
        if shown not in tex:
            bad.append(f"adaptive table, arm {arm}: frames {shown} not in the paper")

    # ---- on-board cost ------------------------------------------------------
    for name, label in (("edge_yolo26s_tiled.json", "YOLO26s tiled"),
                        ("edge_yolo26n_tiled.json", "YOLO26n tiled"),
                        ("edge_yolo26n_full.json", "YOLO26n full")):
        payload = load(ARMS / name)
        if not payload:
            bad.append(f"missing {name}")
            continue
        compute = payload["compute"]
        cells = [f"${compute['fps']:.1f}$",
                 f"${compute['stage_ms']['detect_ms_median']:.0f}$",
                 f"${compute['stage_ms']['track_ms_median']:.0f}$",
                 f"${compute['joules_per_frame']:.1f}$"]
        for cell in cells:
            checked += 1
            if cell not in tex:
                bad.append(f"edge table, {label}: {cell} not in the paper")

    link = load(ARMS / "link_row_6.1_1.json")
    if link:
        source_mbit = link["link"]["source_mbit_s"]
        sparse = link["link"]["jpeg_mean_bytes"] * 1.67 * 8 / 1e6
        for cell in (f"${source_mbit:.1f}$", f"${sparse:.1f}$"):
            checked += 1
            if cell not in tex:
                bad.append(f"link table: {cell} not in the paper")

    # ---- Panel B of the configuration table ---------------------------------
    panel_b = load(DEC / "hota_panelB.json")
    if panel_b:
        for label, row in panel_b["rows"].items():
            cell = f"${row['HOTA']:.3f}$ & ${row['AssA']:.3f}$"
            checked += 1
            if cell not in tex:
                bad.append(f"Panel B row absent or stale: {label}")
            for term in ("U", "D", "M"):
                value = row[term]
                shown = f"{value:,}".replace(",", "{,}") if value >= 1000 else str(value)
                checked += 1
                if shown not in tex:
                    bad.append(f"Panel B {label}: {term}={shown} not in the paper")

    # ---- the low-score second stage -----------------------------------------
    # prose rather than a table cell, which is how it drifted once: the audit
    # covers every second_stage_*.json here, not a subset of them
    offered = accepted = sequences = 0
    for path in sorted(ARMS.glob("second_stage_*.json")):
        payload = load(path)
        if not payload:
            continue
        sequences += 1
        for name, arm in payload["arms"].items():
            if not name.endswith("0.1"):
                continue
            offered += arm["stages"]["second"]["candidates_offered"]
            accepted += arm["stages"]["second"]["assignments_returned"]
    shown = f"{offered:,}".replace(",", "{,}") if offered >= 1000 else str(offered)
    for cell, what in ((shown, "candidates offered"),
                       (f"on {['', 'one', 'two', 'three', 'four', 'five'][sequences]} sequences",
                        "sequence count")):
        checked += 1
        if cell not in tex:
            bad.append(f"second stage: {what} ({cell}) not in the paper")
    if accepted:
        bad.append(f"second stage accepted {accepted}, the paper claims none")

    for line in bad:
        print("FAIL:", line)
    print(f"{checked} cells checked against their result files")
    print("all table cells match" if not bad else f"{len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
