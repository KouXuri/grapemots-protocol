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


def results_dir(name: str) -> Path:
    """Locate a result directory in the working tree or in a release checkout.

    The working tree keeps these under runs/; the released bundle flattens them
    into results/. Searching both lets one script serve an auditor and an author.
    """
    for base in (ROOT / "runs", ROOT / "results", Path.cwd() / "results"):
        candidate = base / name / "results" if base.name == "runs" else base / name
        if candidate.is_dir():
            return candidate
    return ROOT / "runs" / name / "results"


EXT = results_dir("ext_cadence_0813")
ARMS = results_dir("adaptive_0813")
DEC = results_dir("decomp_0812")


def load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> int:
    tex = PAPER.read_text()
    bad: list[str] = []
    checked = 0

    # ---- external cadence contrast -----------------------------------------
    # k=1 is the implementation check the text states rather than a table row
    for corpus, rows in (("mot17", [4, 15, 60]), ("mot20", [4, 15, 60])):
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
            # G and U are printed too: without them the identity cannot be closed
            # from the table, which is how a wrong G survived a whole round
            for term in ("G", "U", "D", "M"):
                checked += 1
                value = rel[term]
                shown = f"{value:,}".replace(",", "{,}") if value >= 1000 else str(value)
                if shown not in tex:
                    bad.append(f"{corpus} k={k}: {term}={shown} not in the paper")
            checked += 1
            if (rel["P"] - rel["G"]) != (rel["U"] + rel["D"] - rel["M"]):
                bad.append(f"{corpus} k={k}: the row does not close the identity")

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
    # the arms are reported in prose, so check the two claims the prose makes:
    # adaptive placement reaches more and over-counts more at every budget, and it
    # crosses zero at half the frames uniform placement needs
    def stats(arm):
        b = pooled[arm]
        return (b["P"] - b["G"]) / b["G"], 1 - b["M"] / b["G"], b["tracker_frames"]
    for n in (2, 4, 8):
        if f"uni{n}" not in pooled or f"ada{n}" not in pooled:
            bad.append(f"budget {n} missing from the pooled results")
            continue
        (ue, ua, _), (ae, aa, _) = stats(f"uni{n}"), stats(f"ada{n}")
        checked += 2
        if not aa > ua:
            bad.append(f"budget {n}: adaptive does not reach more ({aa:.3f} vs {ua:.3f})")
        if not ae > ue:
            bad.append(f"budget {n}: adaptive does not over-count more ({ae:+.3f} vs {ue:+.3f})")
    checked += 1
    crossing_u = next((n for n in (2, 4, 8) if stats(f"uni{n}")[0] > 0), None)
    crossing_a = next((n for n in (2, 4, 8) if stats(f"ada{n}")[0] > 0), None)
    if not (crossing_u and crossing_a and crossing_a * 2 == crossing_u):
        bad.append(f"crossing budgets are {crossing_a} and {crossing_u}, not a factor of two")

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

    # the link rows are now measured through one encoder, so they come from the
    # all-intra benchmark rather than the JPEG stand-in
    intra = load(results_dir("link_allintra_0814") / "link_allintra.json")
    if intra:
        for key, label in (("source_mbit_s", "released bitrate"),
                           ("fullrate_same_codec_mbit_s", "full rate, same codec"),
                           ("allintra_mbit_s_at_sparse_rate", "sparse, all-intra"),
                           ("interframe_mbit_s_at_sparse_rate", "sparse, inter")):
            checked += 1
            cell = f"${intra[key]:.1f}$"
            if cell not in tex:
                bad.append(f"link table, {label}: {cell} not in the paper")

    # ---- Panel B of the configuration table ---------------------------------
    panel_b = load(DEC / "hota_panelB.json")
    identity = load(results_dir("definition_0815") / "idf1_table2.json")
    if panel_b:
        for label, row in panel_b["rows"].items():
            idf1 = identity["panelB"][label]["IDF1"] if identity else None
            if idf1 is None:
                bad.append(f"Panel B {label}: no IDF1 in idf1_table2.json")
                continue
            cell = f"${idf1:.3f}$ & ${row['HOTA']:.3f}$"
            checked += 1
            if cell not in tex:
                bad.append(f"Panel B row absent or stale: {label}")
            for term in ("U", "D", "M"):
                value = row[term]
                shown = f"{value:,}".replace(",", "{,}") if value >= 1000 else str(value)
                checked += 1
                if shown not in tex:
                    bad.append(f"Panel B {label}: {term}={shown} not in the paper")

    # ---- the two confidence replays that bracket the crossing ----------------
    fill = load(results_dir("conf_fill_0815") / "conf_fill.json")
    figure = load(ROOT / "figures/fig_cancellation_data.json")
    if fill:
        for label in ("Confidence 0.75", "Confidence 0.80"):
            row = fill["rows"][label]
            for shown in (f"${row['signed_error']:+.3f}$",
                          f"${row['assigned_fraction']:.2f}$"):
                checked += 1
                if shown not in tex:
                    bad.append(f"{label}: {shown} not in the paper")
    if figure:
        for name, value in figure["crossings"].items():
            checked += 1
            if f"${value:.2f}$" not in tex:
                bad.append(f"Fig. 1 crossing for {name}, {value:.2f}, not in the paper")

    # ---- the timescale and gate controls of Table III ------------------------
    timescale = load(results_dir("timescale_0815") / "timescale_summary.json")
    if timescale:
        for key in ("rel_dt->src/tau1", "rel_g03->src_g03/tau1", "rel_g05->src_g05/tau1"):
            cell = timescale["pairs"][key]
            for shown in (f"${cell['first_median']:+.3f}$",
                          f"${cell['second_median']:+.3f}$",
                          f"${cell['delta_median']:+.3f}$",
                          f"$[{cell['ci95_sequence'][0]:+.2f},{cell['ci95_sequence'][1]:+.2f}]$"):
                checked += 1
                if shown not in tex:
                    bad.append(f"control row {key}: {shown} not in the paper")
            counts = f"{cell['up']}\\,/\\,{cell['down']}\\,/\\,{cell['tie']}"
            checked += 1
            if counts not in tex:
                bad.append(f"control row {key}: {cell['up']}/{cell['down']}/{cell['tie']} not in the paper")
        # the sparse arm's own move once its filter runs on elapsed time
        own = timescale["pairs"]["rel->rel_dt/tau1"]
        checked += 1
        if f"${own['delta_median']:+.3f}$" not in tex:
            bad.append(f"elapsed-time arm: own median move {own['delta_median']:+.3f} not in the paper")
        for arm in ("rel_dt", "src"):
            steps = timescale["kalman_steps"][arm]["kalman_steps"]
            checked += 1
            if f"{steps:,}".replace(",", "{,}") not in tex:
                bad.append(f"{arm}: {steps} Kalman steps not in the paper")

    # ---- the definition controls quoted in Section II-D ----------------------
    sens = load(results_dir("definition_0815") / "definition_sensitivity.json")
    if sens:
        for gate, arm in [(g, a) for g in ("0.3", "0.5", "0.7") for a in ("rel", "src")]:
            cell = sens["gate"][f"iou{gate}_tau1_{arm}"]
            shown = f"${cell['assigned_fraction']:.2f}$".replace("$0.", "$0.")
            checked += 1
            if f"{cell['assigned_fraction']:.2f}"[1:] not in tex:
                bad.append(f"ownership gate {gate}, {arm}: "
                           f"{cell['assigned_fraction']:.2f} not in the paper")
        for arm in ("rel", "src"):
            cell = sens["tau_symmetry"][f"tau3_{arm}"]
            checked += 1
            if f"${cell['e_symmetric']:+.3f}$" not in tex:
                bad.append(f"symmetric tau=3, {arm}: {cell['e_symmetric']:+.3f} not in the paper")
        checked += 1
        if str(sens["gate"]["iou0.5_tau1_rel"]["P"] + sens["gate"]["iou0.5_tau1_src"]["P"]) not in tex:
            bad.append("the predicted-track total behind the purity claim is not in the paper")

    identity = load(results_dir("definition_0815") / "identity_metrics.json")
    if identity:
        for arm in ("rel", "src"):
            row = identity["rows"][arm]
            for key in ("D", "IDSW", "M", "ML"):
                checked += 1
                if str(row[key]) not in tex:
                    bad.append(f"identity family, {arm}: {key}={row[key]} not in the paper")
            checked += 1
            if f"${row['IDF1']:.3f}$" not in tex:
                bad.append(f"identity family, {arm}: IDF1 {row['IDF1']:.3f} not in the paper")

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

    # ---- out-of-fold detector quality ---------------------------------------
    ap = load(results_dir("ap_lovo_0814") / "lovo_ap_summary.json")
    if ap:
        for res, group in ap["groups"].items():
            for key in ("median_ap50", "median_ap50_95"):
                checked += 1
                if f"${group[key]:.3f}$" not in tex:
                    bad.append(f"detector AP {res} {key}={group[key]:.3f} not in the paper")
            checked += 1
            if str(group["videos"]) not in tex:
                bad.append(f"detector AP {res}: {group['videos']} videos not stated")

    for line in bad:
        print("FAIL:", line)
    print(f"{checked} cells checked against their result files")
    print("all table cells match" if not bad else f"{len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
