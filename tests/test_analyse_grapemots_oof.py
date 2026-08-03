import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from tools.analyse_grapemots_oof import (
    ARMS,
    SPLITS,
    analyse_results,
    build_trackeval_data,
    load_oof_results,
    main,
    render_markdown,
)


GT_IDS = [[101, 909], [101, 909]]
GT_BOXES = [
    [[0, 0, 10, 10], [20, 0, 30, 10]],
    [[0, 0, 10, 10], [20, 0, 30, 10]],
]


def video_payload(name: str, arm: str) -> dict:
    if arm == "conf055":
        pred_ids = [[71, 83], [71, 83]]
        pred_boxes = [
            [[40, 0, 50, 10], [60, 0, 70, 10]],
            [[40, 0, 50, 10], [60, 0, 70, 10]],
        ]
    elif arm == "reid":
        pred_ids = [[71, 83, 95], [71, 83, 95]]
        pred_boxes = [
            [[0, 0, 10, 10], [20, 0, 30, 10], [60, 0, 70, 10]],
            [[0, 0, 10, 10], [20, 0, 30, 10], [60, 0, 70, 10]],
        ]
    elif arm == "botsort":
        pred_ids = [[71, 83], [95, 83]]
        pred_boxes = [GT_BOXES[0], GT_BOXES[1]]
    else:
        pred_ids = [[71, 83], [71, 83]]
        pred_boxes = [GT_BOXES[0], GT_BOXES[1]]
    return {
        "video": name,
        "frames": 2,
        "frame_predicted_ids": pred_ids,
        "frame_predicted_boxes": pred_boxes,
        "frame_gt_ids": GT_IDS,
        "frame_gt_boxes": GT_BOXES,
    }


def write_complete_results(directory: Path) -> None:
    for split in SPLITS:
        for arm in ARMS:
            payload = {
                "config": {"split": split},
                "videos": [
                    video_payload(f"video_{split}_1", arm),
                    video_payload(f"video_{split}_2", arm),
                ],
            }
            (directory / f"oof_{split}_{arm}.json").write_text(json.dumps(payload))


def test_load_and_aggregate_both_cohorts(tmp_path: Path):
    write_complete_results(tmp_path)
    loaded = load_oof_results(tmp_path)
    report = analyse_results(loaded, hota_factory=None)

    primary = report["cohorts"]["primary"]
    sensitivity = report["cohorts"]["sensitivity"]
    assert primary["split_video_cells"] == 6
    assert sensitivity["split_video_cells"] == 10
    assert primary["identity_checks"] == {
        "all_hold": True,
        "passed": 36,
        "total": 36,
        "failures": [],
    }

    conf = primary["arms"]["conf055"]
    assert conf["pooled"] == {
        "P": 12,
        "G": 12,
        "U": 12,
        "D": 0,
        "M": 12,
        "net_signed_error": 0.0,
        "recovered_trajectories": 0,
        "recovered_fraction": 0.0,
        "identity_holds": True,
    }
    assert conf["macro_medians"]["per_cell_net_signed_error"] == 0.0
    assert conf["macro_medians"]["per_cell_recovered_fraction"] == 0.0

    botsort = primary["arms"]["botsort"]["pooled"]
    assert (botsort["P"], botsort["G"], botsort["D"]) == (18, 12, 6)
    assert botsort["identity_holds"] is True

    pair = primary["conf055_vs_reid"]
    assert pair["strict_inversion_count"] == 6
    assert pair["conf055_lower_absolute_error_reid_higher_recovery_count"] == 6
    assert pair["conf055_lower_signed_error_reid_higher_recovery_count"] == 6


def test_loader_accepts_gzip_results(tmp_path: Path):
    write_complete_results(tmp_path)
    for path in tmp_path.glob("*.json"):
        compressed = path.with_suffix(".json.gz")
        with compressed.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                handle.write(path.read_bytes())
        path.unlink()

    loaded = load_oof_results(tmp_path)
    assert analyse_results(loaded, hota_factory=None)["cohorts"]["primary"][
        "split_video_cells"
    ] == 6


def test_trackeval_pool_remaps_sparse_ids_between_video_cells():
    first = video_payload("first", "reid")
    second = video_payload("second", "reid")
    data = build_trackeval_data([first, second])

    assert data["num_timesteps"] == 4
    assert data["num_gt_ids"] == 4
    assert data["num_tracker_ids"] == 6
    assert data["num_gt_dets"] == 8
    assert data["num_tracker_dets"] == 12
    np.testing.assert_array_equal(data["gt_ids"][0], [0, 1])
    np.testing.assert_array_equal(data["gt_ids"][2], [2, 3])
    np.testing.assert_array_equal(data["tracker_ids"][0], [0, 1, 2])
    np.testing.assert_array_equal(data["tracker_ids"][2], [3, 4, 5])
    np.testing.assert_allclose(data["similarity_scores"][0][0], [1.0, 0.0, 0.0])


def test_fake_trackeval_metrics_are_pooled_and_serialised(tmp_path: Path):
    write_complete_results(tmp_path)
    loaded = load_oof_results(tmp_path)

    class FakeHOTA:
        def eval_sequence(self, data):
            assert data["num_timesteps"] in {2, 12, 20}
            return {
                "HOTA": np.asarray([0.2, 0.4]),
                "DetA": np.asarray([0.3, 0.5]),
                "AssA": np.asarray([0.1, 0.3]),
                "LocA": np.asarray([0.7, 0.9]),
            }

    report = analyse_results(loaded, hota_factory=FakeHOTA)
    metrics = report["cohorts"]["primary"]["arms"]["conf055"]["pooled_tracking_metrics"]
    assert metrics == {
        "available": True,
        "reason": None,
        "HOTA": pytest.approx(0.3),
        "DetA": pytest.approx(0.4),
        "AssA": pytest.approx(0.2),
        "LocA": pytest.approx(0.8),
    }


def test_missing_trackeval_is_reported_without_failure(tmp_path: Path):
    write_complete_results(tmp_path)
    report = analyse_results(load_oof_results(tmp_path), hota_factory=None)
    metrics = report["cohorts"]["primary"]["arms"]["conf055"]["pooled_tracking_metrics"]

    assert metrics["available"] is False
    assert all(metrics[name] is None for name in ("HOTA", "DetA", "AssA", "LocA"))
    markdown = render_markdown(report)
    assert "TrackEval was not supplied" in markdown
    assert "6 of 6 paired cells were strict inversions" in markdown
    assert "summarized" not in markdown


def test_loader_rejects_non_unique_primary_test_videos(tmp_path: Path):
    write_complete_results(tmp_path)
    for arm in ARMS:
        path = tmp_path / f"oof_B_{arm}.json"
        payload = json.loads(path.read_text())
        payload["videos"][0]["video"] = "video_A_1"
        path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="6 unique held-out videos"):
        load_oof_results(tmp_path)


def test_loader_rejects_misaligned_frame_dump(tmp_path: Path):
    write_complete_results(tmp_path)
    path = tmp_path / "oof_A_reid.json"
    payload = json.loads(path.read_text())
    payload["videos"][0]["frame_predicted_boxes"].pop()
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="inconsistent frame-dump lengths"):
        load_oof_results(tmp_path)


def test_cli_writes_json_and_markdown_outputs(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    write_complete_results(results)
    json_out = tmp_path / "reports" / "oof.json"
    markdown_out = tmp_path / "reports" / "oof.md"

    main(
        [
            "--results",
            str(results),
            "--splits",
            "A",
            "B",
            "C",
            "D",
            "E",
            "--primary-splits",
            "A",
            "B",
            "C",
            "--out",
            str(json_out),
        ]
    )

    payload = json.loads(json_out.read_text())
    assert payload["cohorts"]["primary"]["split_video_cells"] == 6
    text = markdown_out.read_text()
    assert "# GrapeMOTS out-of-fold configuration analysis" in text
    assert "Count identity checks: 36/36 per-cell checks passed." in text
