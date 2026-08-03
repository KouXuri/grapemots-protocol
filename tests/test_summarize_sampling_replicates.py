import json
from pathlib import Path

import pytest

from tools.summarize_sampling_replicates import DEFAULT_RUNS, summarise


def write_curve(path: Path, best: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "epoch,metrics/mAP50(B)\n"
        f"1,{best}\n"
        f"2,{best - 0.1}\n"
    )


def write_evaluation(path: Path, ap50: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "merge_iou": 0.6,
                "overall": {
                    "ap50": ap50,
                    "ap50_95": ap50 / 2,
                    "ar100": ap50 / 3,
                    "frames": 1160,
                },
            }
        )
    )


def test_summarise_uses_paired_seed_differences(tmp_path: Path):
    runs = tmp_path / "runs"
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    for representation, seeds in DEFAULT_RUNS.items():
        for seed, (run_name, result_name) in seeds.items():
            write_curve(runs / run_name / "results.csv", 0.4 + seed / 100)
            root = legacy if seed == 0 else current
            base = 0.2 + seed / 100
            if representation == "step1":
                base += 0.03
            write_evaluation(root / result_name, base)

    report = summarise(runs, current, legacy)

    differences = report["paired_ap50_differences"]
    assert list(differences["step1_minus_step3_by_seed"]) == [0, 1, 2]
    assert differences["mean"] == pytest.approx(0.03)
    assert differences["population_sd"] == pytest.approx(0.0)
