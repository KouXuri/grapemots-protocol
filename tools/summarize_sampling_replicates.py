#!/usr/bin/env python3
"""Summarise paired seed replicates for sparse- and all-frame training."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_RUNS = {
    "step3": {
        0: ("gm_ctrl_newsplit_oldcfg", "v1_traintiled_infertiled_val.json"),
        1: ("gm_step3_A_s1_0802", "sampling_step3_s1_val.json"),
        2: ("gm_step3_A_s2_0802", "sampling_step3_s2_val.json"),
    },
    "step1": {
        0: ("gm_step1_matched", "fair_gm_step1_matched.json"),
        1: ("gm_step1_A_s1_0802", "sampling_step1_s1_val.json"),
        2: ("gm_step1_A_s2_0802", "sampling_step1_s2_val.json"),
    },
}


def training_curve(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No training rows in {path}")
    metric = "metrics/mAP50(B)"
    best = max(rows, key=lambda row: float(row[metric]))
    return {
        "epochs_completed": int(float(rows[-1]["epoch"])),
        "best_epoch": int(float(best["epoch"])),
        "best_tile_ap50": float(best[metric]),
        "last_tile_ap50": float(rows[-1][metric]),
    }


def evaluation(path: Path) -> dict:
    data = json.loads(path.read_text())
    overall = data["overall"]
    return {
        "ap50": overall["ap50"],
        "ap50_95": overall["ap50_95"],
        "ar100": overall["ar100"],
        "frames": overall["frames"],
        "merge_iou": data["merge_iou"],
    }


def summarise(runs_root: Path, current_results: Path, legacy_results: Path) -> dict:
    report = {"representations": {}, "paired_ap50_differences": {}}
    for representation, seeds in DEFAULT_RUNS.items():
        records = []
        for seed, (run_name, result_name) in seeds.items():
            result_root = legacy_results if seed == 0 else current_results
            run_path = runs_root / run_name
            record = {
                "seed": seed,
                "run": run_name,
                **training_curve(run_path / "results.csv"),
                **evaluation(result_root / result_name),
            }
            records.append(record)
        ap50 = [record["ap50"] for record in records]
        report["representations"][representation] = {
            "runs": records,
            "mean_ap50": statistics.mean(ap50),
            "population_sd_ap50": statistics.pstdev(ap50),
            "mean_ap50_95": statistics.mean(record["ap50_95"] for record in records),
        }

    step3 = {record["seed"]: record for record in report["representations"]["step3"]["runs"]}
    step1 = {record["seed"]: record for record in report["representations"]["step1"]["runs"]}
    differences = {seed: step1[seed]["ap50"] - step3[seed]["ap50"] for seed in step3}
    report["paired_ap50_differences"] = {
        "step1_minus_step3_by_seed": differences,
        "mean": statistics.mean(differences.values()),
        "population_sd": statistics.pstdev(differences.values()),
    }
    return report


def markdown(report: dict) -> str:
    lines = [
        "# Sampling replicate summary",
        "",
        "| representation | seed | best epoch | val AP50 | val AP50--95 | AR@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for representation, group in report["representations"].items():
        for record in group["runs"]:
            lines.append(
                f"| {representation} | {record['seed']} | {record['best_epoch']} | "
                f"{record['ap50']:.4f} | {record['ap50_95']:.4f} | {record['ar100']:.4f} |"
            )
        lines.append(
            f"| {representation} mean (population SD) | | | "
            f"{group['mean_ap50']:.4f} ({group['population_sd_ap50']:.4f}) | "
            f"{group['mean_ap50_95']:.4f} | |"
        )
    paired = report["paired_ap50_differences"]
    lines.extend(
        [
            "",
            "Paired AP50 differences (`step1 - step3`): "
            + ", ".join(
                f"seed {seed}: {difference:+.4f}"
                for seed, difference in paired["step1_minus_step3_by_seed"].items()
            )
            + f"; mean {paired['mean']:+.4f}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--legacy-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = summarise(args.runs_root, args.results, args.legacy_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(markdown(report))
    print(markdown(report))
    print(f"wrote {args.out} and {markdown_path}")


if __name__ == "__main__":
    main()
