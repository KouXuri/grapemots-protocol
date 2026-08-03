import gzip
import json

import pytest

from tools.summarize_fullrate_controls import read_json, summarise


def record(video, step, predicted, ground_truth=10):
    return {
        "video": video,
        "step": step,
        "tracker_frames": 40 if step == 1 else 20,
        "scored_frames": 20,
        "count_error_surface": [
            {
                "window_frames": 20,
                "min_track_len": 1,
                "predicted_tracks": predicted,
                "gt_tracks": ground_truth,
                "signed_relative_error": (predicted - ground_truth) / ground_truth,
            }
        ],
    }


def test_summarise_pairs_rates_and_buffers_by_video():
    default = {
        "runs": [
            record("video_a", 1, 16),
            record("video_a", 2, 12),
            record("video_b", 1, 14),
            record("video_b", 2, 10),
        ]
    }
    buffer60 = {"runs": [record("video_a", 1, 15), record("video_b", 1, 13)]}

    report = summarise(default, buffer60)

    assert report["pooled"]["15hz_buffer30"]["signed_relative_error"] == pytest.approx(0.1)
    assert report["pooled"]["30hz_buffer30"]["signed_relative_error"] == pytest.approx(0.5)
    assert report["pooled"]["30hz_buffer60"]["signed_relative_error"] == pytest.approx(0.4)
    assert report["buffer_share_of_30hz_difference"] == pytest.approx(0.25)
    assert report["scope"] == {
        "videos": 2,
        "processed_source_frames_30hz": 80,
        "scored_frames": 40,
    }


def png_record(video, predicted, ground_truth=10):
    return {
        "video": video,
        "frames": 20,
        "count_error_surface": [
            {
                "window_frames": 20,
                "min_track_len": 1,
                "prefix_predicted_tracks": predicted,
                "prefix_gt_tracks": ground_truth,
                "prefix_signed_relative_error": (predicted - ground_truth) / ground_truth,
            }
        ],
    }


def test_summarise_freezes_decoded_mp4_vs_released_png_difference():
    default = {
        "runs": [
            record("video_a", 1, 16),
            record("video_a", 2, 12),
            record("video_b", 1, 14),
            record("video_b", 2, 11),
        ]
    }
    buffer60 = {"runs": [record("video_a", 1, 15), record("video_b", 1, 13)]}
    png = {"videos": [png_record("video_a", 11), png_record("video_b", 13)]}

    report = summarise(default, buffer60, [png])

    comparison = report["decoded15hz_vs_released_png"]
    assert comparison["minimum_absolute_error_difference"] == pytest.approx(0.1)
    assert comparison["maximum_absolute_error_difference"] == pytest.approx(0.2)
    assert comparison["median_absolute_error_difference"] == pytest.approx(0.15)
    assert comparison["maximum_difference_video"] == "video_b"


def test_summarise_rejects_incomplete_png_baselines():
    default = {
        "runs": [
            record("video_a", 1, 16),
            record("video_a", 2, 12),
            record("video_b", 1, 14),
            record("video_b", 2, 10),
        ]
    }
    buffer60 = {"runs": [record("video_a", 1, 15), record("video_b", 1, 13)]}

    with pytest.raises(ValueError, match="Missing released-PNG baseline for video_b"):
        summarise(default, buffer60, [{"videos": [png_record("video_a", 11)]}])


def test_read_json_accepts_gzip(tmp_path):
    payload = {"videos": [png_record("video_a", 11)]}
    path = tmp_path / "baseline.json.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(json.dumps(payload).encode())
    assert read_json(path) == payload
