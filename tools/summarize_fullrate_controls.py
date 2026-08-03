#!/usr/bin/env python3
"""Combine 15/30 Hz and time-matched-buffer full-rate controls."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def whole_cell(record: dict) -> dict:
    scored_frames = record.get("scored_frames", record.get("frames"))
    if scored_frames is None:
        raise ValueError(f"Record has no scored-frame count: {record.get('video', '<unknown>')}")
    return next(
        cell
        for cell in record["count_error_surface"]
        if cell["window_frames"] == scored_frames and cell["min_track_len"] == 1
    )


def cell_counts(cell: dict) -> dict:
    """Normalise full-rate and OOF prefix cells to one count schema."""
    if "predicted_tracks" in cell:
        return {
            "predicted": cell["predicted_tracks"],
            "gt": cell["gt_tracks"],
            "signed_relative_error": cell["signed_relative_error"],
        }
    return {
        "predicted": cell["prefix_predicted_tracks"],
        "gt": cell["prefix_gt_tracks"],
        "signed_relative_error": cell["prefix_signed_relative_error"],
    }


def index_runs(data: dict) -> dict[tuple[str, int], dict]:
    indexed = {}
    for record in data["runs"]:
        key = (record["video"], record["step"])
        if key in indexed:
            raise ValueError(f"Duplicate full-rate record: {key}")
        indexed[key] = record
    return indexed


def index_png_baselines(payloads: list[dict] | tuple[dict, ...]) -> dict[str, dict]:
    indexed = {}
    for payload in payloads:
        for record in payload["videos"]:
            video = record["video"]
            if video in indexed:
                raise ValueError(f"Duplicate released-PNG baseline: {video}")
            indexed[video] = record
    return indexed


def summarise(
    default: dict,
    buffer60: dict,
    png_baselines: list[dict] | tuple[dict, ...] = (),
) -> dict:
    base = index_runs(default)
    matched = index_runs(buffer60)
    png = index_png_baselines(png_baselines)
    videos = sorted(video for video, step in base if step == 1)
    if not videos or {(video, 2) for video in videos} - set(base):
        raise ValueError("Default result must contain steps 1 and 2 for every video")
    if {(video, 1) for video in videos} - set(matched):
        raise ValueError("Buffer-60 result must contain step 1 for every video")

    rows = []
    pooled = {
        "15hz_buffer30": {"predicted": 0, "gt": 0},
        "30hz_buffer30": {"predicted": 0, "gt": 0},
        "30hz_buffer60": {"predicted": 0, "gt": 0},
    }
    for video in videos:
        conditions = {
            "15hz_buffer30": cell_counts(whole_cell(base[(video, 2)])),
            "30hz_buffer30": cell_counts(whole_cell(base[(video, 1)])),
            "30hz_buffer60": cell_counts(whole_cell(matched[(video, 1)])),
        }
        ground_truth = {cell["gt"] for cell in conditions.values()}
        if len(ground_truth) != 1:
            raise ValueError(f"Ground-truth count changed across conditions for {video}")
        row = {"video": video, "gt": ground_truth.pop(), "conditions": {}}
        for name, cell in conditions.items():
            row["conditions"][name] = {
                "predicted": cell["predicted"],
                "signed_relative_error": cell["signed_relative_error"],
            }
            pooled[name]["predicted"] += cell["predicted"]
            pooled[name]["gt"] += cell["gt"]
        row["delta_30hz_minus_15hz"] = (
            conditions["30hz_buffer30"]["signed_relative_error"]
            - conditions["15hz_buffer30"]["signed_relative_error"]
        )
        row["delta_buffer60_minus_buffer30_at_30hz"] = (
            conditions["30hz_buffer60"]["signed_relative_error"]
            - conditions["30hz_buffer30"]["signed_relative_error"]
        )
        if png:
            if video not in png:
                raise ValueError(f"Missing released-PNG baseline for {video}")
            png_counts = cell_counts(whole_cell(png[video]))
            if png_counts["gt"] != row["gt"]:
                raise ValueError(f"Ground-truth count changed in released-PNG baseline for {video}")
            row["released_png_baseline"] = png_counts
            row["absolute_error_difference_decoded15hz_vs_png"] = abs(
                conditions["15hz_buffer30"]["signed_relative_error"]
                - png_counts["signed_relative_error"]
            )
        rows.append(row)

    if png and set(png) != set(videos):
        extra = sorted(set(png) - set(videos))
        raise ValueError(f"Unexpected released-PNG baselines: {extra}")

    for values in pooled.values():
        values["signed_relative_error"] = (
            values["predicted"] - values["gt"]
        ) / values["gt"]
    cadence_delta = (
        pooled["30hz_buffer30"]["signed_relative_error"]
        - pooled["15hz_buffer30"]["signed_relative_error"]
    )
    buffer_reduction = (
        pooled["30hz_buffer30"]["signed_relative_error"]
        - pooled["30hz_buffer60"]["signed_relative_error"]
    )
    report = {
        "scope": {
            "videos": len(videos),
            "processed_source_frames_30hz": sum(
                base[(video, 1)]["tracker_frames"] for video in videos
            ),
            "scored_frames": sum(base[(video, 1)]["scored_frames"] for video in videos),
        },
        "videos": rows,
        "pooled": pooled,
        "pooled_30hz_minus_15hz": cadence_delta,
        "pooled_reduction_from_time_matched_buffer": buffer_reduction,
        "buffer_share_of_30hz_difference": (
            buffer_reduction / cadence_delta if cadence_delta else None
        ),
        "median_video_delta_30hz_minus_15hz": statistics.median(
            row["delta_30hz_minus_15hz"] for row in rows
        ),
    }
    if png:
        differences = [
            row["absolute_error_difference_decoded15hz_vs_png"] for row in rows
        ]
        max_index = max(range(len(rows)), key=lambda index: differences[index])
        report["decoded15hz_vs_released_png"] = {
            "minimum_absolute_error_difference": min(differences),
            "maximum_absolute_error_difference": max(differences),
            "median_absolute_error_difference": statistics.median(differences),
            "maximum_difference_video": rows[max_index]["video"],
        }
    return report


def markdown(report: dict) -> str:
    scope = report["scope"]
    lines = [
        "# Full-rate out-of-fold control",
        "",
        f"The 30 Hz arm processes {scope['processed_source_frames_30hz']:,} source frames; "
        f"all conditions are scored at the same {scope['scored_frames']:,} annotated times.",
        "",
        "| video | G | 15 Hz, buffer 30 | 30 Hz, buffer 30 | 30 Hz, buffer 60 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["videos"]:
        conditions = row["conditions"]
        lines.append(
            f"| {row['video']} | {row['gt']} | "
            f"{conditions['15hz_buffer30']['signed_relative_error']:+.3f} | "
            f"{conditions['30hz_buffer30']['signed_relative_error']:+.3f} | "
            f"{conditions['30hz_buffer60']['signed_relative_error']:+.3f} |"
        )
    pooled = report["pooled"]
    lines.append(
        "| pooled | "
        f"{pooled['15hz_buffer30']['gt']} | "
        f"{pooled['15hz_buffer30']['signed_relative_error']:+.3f} | "
        f"{pooled['30hz_buffer30']['signed_relative_error']:+.3f} | "
        f"{pooled['30hz_buffer60']['signed_relative_error']:+.3f} |"
    )
    share = report["buffer_share_of_30hz_difference"]
    lines.extend(
        [
            "",
            f"Pooled 30 Hz minus 15 Hz error: {report['pooled_30hz_minus_15hz']:+.3f}.",
            "Time-matching the buffer changes that difference by "
            + (f"{share:.1%}." if share is not None else "an undefined fraction (zero denominator)."),
            "",
        ]
    )
    comparison = report.get("decoded15hz_vs_released_png")
    if comparison:
        lines.extend(
            [
                "The decoded 15 Hz and released-PNG errors differ by "
                f"{comparison['minimum_absolute_error_difference']:.3f}--"
                f"{comparison['maximum_absolute_error_difference']:.3f} across videos "
                f"(median {comparison['median_absolute_error_difference']:.3f}); "
                f"{comparison['maximum_difference_video']} has the largest difference.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--buffer60", type=Path, required=True)
    parser.add_argument(
        "--png-baseline",
        type=Path,
        action="append",
        default=[],
        help="OOF BoT-SORT JSON on released PNG frames; repeat once per split",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = summarise(
        read_json(args.default),
        read_json(args.buffer60),
        [read_json(path) for path in args.png_baseline],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(markdown(report))
    print(markdown(report))
    print(f"wrote {args.out} and {markdown_path}")


if __name__ == "__main__":
    main()
