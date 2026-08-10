#!/usr/bin/env python3
"""The countability preflight: what a corpus permits, before any model is trained.

A distinct-track count is only meaningful if the reference itself supports
association at the cadence it is released at. That is checkable from the
annotation alone, and it separates three regimes that the manuscript's four
corpora fall into cleanly:

  associable      consecutive annotated boxes of one trajectory still overlap,
                  and trajectories live far longer than the tracker's
                  confirmation delay. Association is feasible; what remains is
                  fragmentation, which drives the count UP.

  marginal        overlap is present but frequently below the association gate.
                  Both fragmentation and loss operate, and the protocol can move
                  the reported number by a large factor.

  degenerate      a trajectory typically does not overlap itself between
                  consecutive annotated frames. An IoU-gated associator is
                  operating blind, and the reported count is governed by the
                  confirmation and eligibility rules rather than by tracking.

The thresholds below are declared, not fitted: 0.3 is a common association gate
and the manuscript's ownership threshold is 0.5, so 'usually below 0.3' is the
point at which an IoU-gated cost matrix stops carrying information. They are
reported alongside the raw distributions so a reader can apply their own.

Named splits are included because the published results this analysis explains
were computed on particular subsets, and a statement about a paper's numbers
should be computed on that paper's data rather than on a corpus average.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Association gates in common use. Piazolo et al. (CEA 2026) evaluate at 0.2 on
# the Bodegas release; the manuscript owns tracks at 0.5.
GATE = 0.3

# Subsets on which published numbers were computed, so the preflight can be
# reported for those exact numbers.
NAMED_SPLITS = {
    "bodegas2023": {
        "piazolo2026_test": ["row_4.2_1", "row_6.1_1", "row_6.1_2",
                             "row_7.1_1", "row_7.1_2", "row_8_1"],
        "piazolo2026_train": None,          # complement, filled in below
    },
}


def classify(entry: dict) -> str:
    below = entry["consecutive_iou_below_0p3"]
    overlap = entry["consecutive_iou_median"]
    if below is None or overlap is None:
        return "unknown"
    if overlap <= 0.05 or below >= 0.8:
        return "degenerate"
    if overlap < 0.5 or below >= 0.25:
        return "marginal"
    return "associable"


def summarise(entries: list[dict], label: str) -> dict:
    def med(key):
        values = [entry[key] for entry in entries if entry[key] is not None]
        return float(np.median(values)) if values else None

    verdicts = [classify(entry) for entry in entries]
    return {
        "label": label,
        "sequences": len(entries),
        "frames": int(sum(entry["frames"] for entry in entries)),
        "trajectories": int(sum(entry["trajectories"] for entry in entries)),
        "consecutive_iou_median": med("consecutive_iou_median"),
        "consecutive_iou_below_gate_median": med("consecutive_iou_below_0p3"),
        "consecutive_iou_below_gate_min": float(min(
            entry["consecutive_iou_below_0p3"] for entry in entries)),
        "consecutive_iou_below_gate_max": float(max(
            entry["consecutive_iou_below_0p3"] for entry in entries)),
        "step_over_size_median": med("step_over_size_median"),
        "lifetime_median": med("lifetime_median"),
        "lifetime_min": float(min(entry["lifetime_median"] for entry in entries)),
        "visible_mean_median": med("visible_mean"),
        "gap_frac_median": med("gap_frac"),
        "verdict_counts": {name: verdicts.count(name)
                           for name in ("associable", "marginal", "degenerate", "unknown")
                           if verdicts.count(name)},
        "verdict": max(set(verdicts), key=verdicts.count),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", type=Path, required=True)
    ap.add_argument("--regime", type=Path, help="optional; adds the measured count error")
    ap.add_argument("--confirmation-frames", type=int, default=5,
                    help="tracker confirmation delay to compare lifetimes against; "
                         "5 is the value Piazolo et al. (CEA 2026) use")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-markdown", type=Path, required=True)
    args = ap.parse_args()

    sequences = json.loads(args.structure.read_text())["sequences"]
    by_corpus: dict[str, list[dict]] = {}
    for entry in sequences:
        by_corpus.setdefault(entry["corpus"], []).append(entry)

    measured = {}
    if args.regime and args.regime.is_file():
        regime = json.loads(args.regime.read_text())
        measured = regime.get("per_corpus", {})

    report = {"gate": GATE, "confirmation_frames": args.confirmation_frames,
              "corpora": {}, "named_splits": {}}

    for corpus, entries in by_corpus.items():
        entry = summarise(entries, corpus)
        entry["lifetime_over_confirmation"] = (
            entry["lifetime_median"] / args.confirmation_frames
            if entry["lifetime_median"] is not None else None)
        if corpus in measured:
            entry["measured"] = {
                "whole_sequence_error_median": measured[corpus]["whole_sequence_error_median"],
                "over_counting": measured[corpus]["over_counting"],
                "under_counting": measured[corpus]["under_counting"],
                "span_ratio_median": measured[corpus]["span_ratio_median"],
            }
        report["corpora"][corpus] = entry

    for corpus, splits in NAMED_SPLITS.items():
        if corpus not in by_corpus:
            continue
        named = {name: members for name, members in splits.items() if members}
        for name, members in splits.items():
            if members is None:
                covered = {video for group in named.values() for video in group}
                members = [entry["video"] for entry in by_corpus[corpus]
                           if entry["video"] not in covered]
            subset = [entry for entry in by_corpus[corpus] if entry["video"] in members]
            if not subset:
                continue
            summary = summarise(subset, name)
            summary["corpus"] = corpus
            summary["lifetime_over_confirmation"] = (
                summary["lifetime_median"] / args.confirmation_frames)
            summary["videos"] = sorted(entry["video"] for entry in subset)
            report["named_splits"][name] = summary

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, default=float) + "\n")

    lines = [
        "# Countability preflight",
        "",
        "Computed from reference annotations only. No detector, no tracker, no training.",
        "",
        f"Association gate for the 'below gate' column: IoU < {GATE}. "
        f"Lifetimes are compared against a {args.confirmation_frames}-frame track "
        "confirmation delay.",
        "",
        "## Corpora",
        "",
        "| corpus | seq | frames | traj | consec IoU | frac below gate | step/size | "
        "lifetime | life/confirm | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for corpus, entry in report["corpora"].items():
        lines.append(
            f"| {corpus} | {entry['sequences']} | {entry['frames']} | {entry['trajectories']} "
            f"| {entry['consecutive_iou_median']:.3f} "
            f"| {entry['consecutive_iou_below_gate_median']:.3f} "
            f"| {entry['step_over_size_median']:.3f} "
            f"| {entry['lifetime_median']:.1f} "
            f"| {entry['lifetime_over_confirmation']:.1f} "
            f"| {entry['verdict']} |")

    if report["named_splits"]:
        lines += ["", "## Splits on which published numbers were computed", "",
                  "| split | seq | frames | traj | consec IoU | frac below gate | "
                  "lifetime | life/confirm | verdict |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for name, entry in report["named_splits"].items():
            lines.append(
                f"| {name} | {entry['sequences']} | {entry['frames']} | {entry['trajectories']} "
                f"| {entry['consecutive_iou_median']:.3f} "
                f"| {entry['consecutive_iou_below_gate_median']:.3f} "
                f"| {entry['lifetime_median']:.1f} "
                f"| {entry['lifetime_over_confirmation']:.1f} "
                f"| {entry['verdict']} |")

    lines += ["", "## Measured count error, for comparison", ""]
    if measured:
        lines += ["| corpus | oracle e (median) | over | under | protocol span |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for corpus, entry in report["corpora"].items():
            if "measured" not in entry:
                continue
            span = entry["measured"]["span_ratio_median"]
            lines.append(
                f"| {corpus} | {entry['measured']['whole_sequence_error_median']:+.3f} "
                f"| {entry['measured']['over_counting']} "
                f"| {entry['measured']['under_counting']} "
                f"| {span:.2f}x |" if span else
                f"| {corpus} | {entry['measured']['whole_sequence_error_median']:+.3f} "
                f"| {entry['measured']['over_counting']} "
                f"| {entry['measured']['under_counting']} | - |")
    else:
        lines.append("(no regime file supplied)")

    args.out_markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out_json}\nwrote {args.out_markdown}")


if __name__ == "__main__":
    main()
