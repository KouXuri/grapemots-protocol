import gzip
import json
from pathlib import Path

import pytest

from tools.analyse_oof_protocol_surface import ARMS, analyse, main


SPLITS = ("A", "B", "C")
PROTOCOLS = ((2, 1), (4, 1), (4, 2))
GROUND_TRUTH_IDS = [
    [101, 102, 103, 104],
    [101, 102, 103, 104],
    [105],
    [],
]


def _boxes(frame_ids):
    return [[[index, 0, index + 1, 1] for index, _ in enumerate(ids)] for ids in frame_ids]


def _predicted_ids(arm):
    if arm == "conf055":
        return [list(range(1, 6)), list(range(1, 6)), [], []]
    if arm == "reid":
        return [list(range(1, 5)), list(range(1, 5)), [5, 6], []]
    return [list(range(1, 7)), list(range(1, 7)), [], []]


def _surface(predicted_ids):
    rows = []
    for length, tau in PROTOCOLS:
        predicted_counts = {}
        ground_truth = set()
        for frame_predicted, frame_truth in zip(
            predicted_ids[:length], GROUND_TRUTH_IDS[:length]
        ):
            for track_id in frame_predicted:
                predicted_counts[track_id] = predicted_counts.get(track_id, 0) + 1
            ground_truth.update(frame_truth)
        predicted = sum(count >= tau for count in predicted_counts.values())
        truth = len(ground_truth)
        rows.append(
            {
                "window_frames": length,
                "min_track_len": tau,
                "prefix_predicted_tracks": predicted,
                "prefix_gt_tracks": truth,
                "prefix_signed_relative_error": (predicted - truth) / truth,
            }
        )
    return rows


def _video(name, arm):
    predicted_ids = _predicted_ids(arm)
    return {
        "video": name,
        "frames": 4,
        "frame_predicted_ids": predicted_ids,
        "frame_predicted_boxes": _boxes(predicted_ids),
        "frame_gt_ids": GROUND_TRUTH_IDS,
        "frame_gt_boxes": _boxes(GROUND_TRUTH_IDS),
        "count_error_surface": _surface(predicted_ids),
    }


def _write_results(directory: Path):
    directory.mkdir()
    for split in SPLITS:
        for arm in ARMS:
            payload = {
                "videos": [
                    _video(f"video_{split}_1", arm),
                    _video(f"video_{split}_2", arm),
                ]
            }
            (directory / f"oof_{split}_{arm}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )


def test_analysis_recomputes_prefixes_and_finds_pairwise_reversals(tmp_path):
    results = tmp_path / "results"
    _write_results(results)

    report = analyse(results, SPLITS)

    assert report["validation"]["input_files"] == 18
    assert report["validation"]["distinct_videos"] == 6
    assert report["validation"]["surface_cells_recomputed"] == 108
    retained = report["retained_surface"]
    assert retained["video_protocol_cells"] == 18
    assert retained["winner_set_changes_vs_full_tau1"] == 6
    conf_reid = next(
        row
        for row in retained["per_pair"]
        if row["first"] == "conf055" and row["second"] == "reid"
    )
    assert conf_reid["reversal_video_count"] == 6

    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    main(
        [
            "--results",
            str(results),
            "--splits",
            *SPLITS,
            "--out",
            str(json_out),
            "--markdown",
            str(markdown_out),
        ]
    )
    assert json.loads(json_out.read_text())["schema_version"] == 1
    assert "Recomputed 108 stored prefix cells" in markdown_out.read_text()


def test_analysis_accepts_gzip_and_preserves_json_hashes(tmp_path):
    results = tmp_path / "results"
    _write_results(results)
    plain_report = analyse(results, SPLITS)
    for path in results.glob("*.json"):
        compressed = path.with_suffix(".json.gz")
        with compressed.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                handle.write(path.read_bytes())
        path.unlink()

    compressed_report = analyse(results, SPLITS)
    assert compressed_report == plain_report


def test_analysis_keeps_a_relative_results_path_portable(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_results(results)
    monkeypatch.chdir(tmp_path)

    report = analyse(Path("results"), SPLITS)

    assert report["inputs"]["results_root"] == "results"


def test_analysis_rejects_a_stale_stored_surface(tmp_path):
    results = tmp_path / "results"
    _write_results(results)
    path = results / "oof_A_conf055.json"
    payload = json.loads(path.read_text())
    payload["videos"][0]["count_error_surface"][0][
        "prefix_signed_relative_error"
    ] = 99
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="e mismatch"):
        analyse(results, SPLITS)
