#!/usr/bin/env python3
"""Every oracle, dropout and split number the paper quotes, from one definition.

Why this exists. The dropout medians in the 2026-07-30 rev2 draft (+1.329 at
p=0, +1.192 at p=0.4 i.i.d., +1.977 under block dropout) could not be recovered
from the stored JSON under any obvious reading of "retained-cell median": pooling
cells gives +1.105/+1.151/+1.586, taking a per-video median first gives
+0.922/+1.333/+1.644, and restricting to tau=1 gives different values again.
A number in a paper about undeclared analysis choices cannot itself depend on an
undeclared analysis choice, so the definition is fixed here, in code, and the
manuscript quotes this file's output and nothing else.

THE DEFINITION, applied everywhere below:
  retained cell   tau <= L/2, G(L) > 0, and coverage = G(L)/G(full) >= 0.8, where
                  G(full) is the whole-sequence cell of the same run
  per video       median over that video's retained cells
  headline        median over the eleven per-video values, so a video with many
                  retained cells does not outweigh one with few
  seeds           pooled within a video before the per-video median is taken
Pooled-cell figures are printed alongside, labelled as such, because they are
what the earlier drafts used and the difference should be visible rather than
quietly corrected.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _find(results, stem):
    """Locate a frozen artefact under results/ or results/oracle/, plain or gzipped."""
    for base in (results, results / "oracle"):
        for name in (f"{stem}.json", f"{stem}.json.gz"):
            candidate = base / name
            if candidate.exists():
                return candidate
    return results / f"{stem}.json"


def _read_json(path):
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())

COVERAGE = 0.8
MODES = ["bernoulli", "block", "identity", "size"]

# Source frames between consecutive annotated frames. Half the release is
# annotated every other frame, so a window length in annotated frames is not a
# duration and phi in annotated-frame units is not comparable across videos.
CADENCE = {"NoPathPlanning_1": 1, "NoPathPlanning_2": 1, "NoPathPlanning_3": 1,
           "PathPlanning_1": 1, "PathPlanning_2": 2, "PathPlanning_3": 2,
           "PathPlanning_4": 2, "PathPlanning_5": 2, "PathPlanning_6": 2,
           "PathPlanning_7": 2, "PathPlanning_8": 2}


def retained(run, coverage=COVERAGE):
    cells = [c for c in run["count_error_surface"]
             if c["min_track_len"] <= c["window_frames"] / 2 and c["gt_tracks"]]
    if not cells:
        return []
    full = max(c["gt_tracks"] for c in cells)
    return [c for c in cells if c["gt_tracks"] / full >= coverage]


def error(cell):
    return (cell["predicted_tracks"] - cell["gt_tracks"]) / cell["gt_tracks"]


def dropout_table(path: Path, coverage=COVERAGE):
    """{miss rate: summary} under the frozen definition."""
    data = _read_json(path)
    per_video = defaultdict(lambda: defaultdict(list))
    for run in data["runs"]:
        for cell in retained(run, coverage):
            per_video[run["miss_rate"]][run["video"]].append(error(cell))
    out = {}
    for miss in sorted(per_video):
        videos = per_video[miss]
        medians = {v: statistics.median(e) for v, e in videos.items()}
        pooled = [x for e in videos.values() for x in e]
        negatives = sorted(v for v, e in videos.items() if min(e) < 0)
        out[miss] = {
            "videos": len(videos),
            "median_of_video_medians": statistics.median(medians.values()),
            "pooled_cell_median": statistics.median(pooled),
            "pooled_cells": len(pooled),
            "min_error": min(pooled),
            "max_error": max(pooled),
            "videos_with_a_negative_cell": negatives,
            "per_video_median": {v: round(m, 4) for v, m in sorted(medians.items())},
        }
    return out


def q_ratio(path: Path, coverage=COVERAGE, miss=0.0):
    """Largest within-sequence spread of the count ratio q = P/G = 1 + e.

    Reported as a ratio of ratios. It is not a ratio of relative errors, and the
    draft's '4.78x' has been read as one.
    """
    data = _read_json(path)
    best = None
    for run in data["runs"]:
        if run["miss_rate"] != miss:
            continue
        cells = retained(run, coverage)
        if len(cells) < 2:
            continue
        qs = [c["predicted_tracks"] / c["gt_tracks"] for c in cells]
        ratio = max(qs) / min(qs)
        if best is None or ratio > best["ratio"]:
            best = {"video": run["video"], "ratio": ratio,
                    "q_low": min(qs), "q_high": max(qs),
                    "e_low": min(qs) - 1, "e_high": max(qs) - 1,
                    "n_cells": len(cells)}
    ratios = []
    for run in data["runs"]:
        if run["miss_rate"] != miss:
            continue
        cells = retained(run, coverage)
        if len(cells) < 2:
            continue
        qs = [c["predicted_tracks"] / c["gt_tracks"] for c in cells]
        ratios.append(max(qs) / min(qs))
    if best:
        best["median_ratio_across_videos"] = statistics.median(ratios)
    return best


def phi_table(path: Path):
    """Fitted slope per source frame, and the spread across sequences."""
    data = _read_json(path)
    phi = {}
    for run in data["runs"]:
        if run["miss_rate"] != 0.0:
            continue
        fit = run.get("drift_fit", {}).get("1")
        if not fit:
            continue
        phi[run["video"]] = {
            "annotated": fit["phi_per_frame"],
            "source": fit["phi_per_frame"] / CADENCE[run["video"]],
            "in_sample_r2": fit["r2"],
        }
    positive = [v["source"] for v in phi.values() if v["source"] > 0]
    return {
        "per_video": {v: {k: round(x, 5) for k, x in d.items()} for v, d in sorted(phi.items())},
        "source_range": [min(positive), max(positive)] if positive else None,
        "source_fold": max(positive) / min(positive) if positive else None,
        "annotated_fold": (max(v["annotated"] for v in phi.values())
                           / min(v["annotated"] for v in phi.values() if v["annotated"] > 0)),
    }


def blocked_validation(path: Path):
    """Does the fitted line beat a no-drift baseline on INDEPENDENT windows?

    The earlier check fitted short prefix windows and predicted long prefix
    windows. Prefix windows are nested, so that tests extrapolation, not
    generalisation, and the manuscript should not have called it held-out. Here
    the samples are the non-overlapping blocks, which share no frames.
    """
    data = _read_json(path)
    wins, rows = 0, []
    for run in sorted(data["runs"], key=lambda r: r["video"]):
        if run["miss_rate"] != 0.0:
            continue
        points = [(row["window_frames"], row["mean_excess"])
                  for row in run.get("blocked_surface", [])
                  if row["min_track_len"] == 1 and row["n_windows"] >= 2]
        if len(points) < 6:
            rows.append((run["video"], None, None, None))
            continue
        points.sort()
        half = len(points) // 2
        train, test = points[:half], points[half:]
        xs = [p[0] for p in train]
        ys = [p[1] for p in train]
        n = len(xs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        var = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var if var else 0.0
        intercept = mean_y - slope * mean_x
        mae_model = statistics.mean(abs(y - (slope * x + intercept)) for x, y in test)
        mae_flat = statistics.mean(abs(y - mean_y) for _, y in test)
        wins += mae_model < mae_flat
        rows.append((run["video"], mae_model, mae_flat, mae_model < mae_flat))
    return {"wins": wins, "total": sum(1 for r in rows if r[1] is not None), "rows": rows}


def split_table(results: Path):
    """Pooled and per-video validation AP for every split, checkpoint and seed."""
    table = defaultdict(dict)
    for path in sorted(results.glob("pervideo_*.json")):
        stem = path.stem[len("pervideo_"):]
        split, checkpoint, target = stem.split("_", 2)
        value = json.loads(path.read_text())["overall"]["ap50"]
        table[f"{split}/{checkpoint}"][target] = round(value, 4)
    return dict(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/paper_numbers.json"))
    args = ap.parse_args()

    report = {"definition": {
        "coverage_threshold": COVERAGE,
        "tau_rule": "tau <= L/2",
        "headline_statistic": "median over videos of each video's median retained cell",
    }}

    print("=" * 78)
    print(f"dropout, coverage >= {COVERAGE}, median over videos of per-video medians")
    print("=" * 78)
    for mode in MODES:
        path = _find(args.results, f"oracle_master_{mode}")
        if not path.exists():
            print(f"  {mode}: missing"); continue
        table = dropout_table(path)
        report[f"dropout_{mode}"] = {str(k): v for k, v in table.items()}
        print(f"\n  {mode}")
        for miss, row in table.items():
            print(f"    p={miss:.1f}  videos={row['videos']:2d}  "
                  f"median-of-medians {row['median_of_video_medians']:+.3f}  "
                  f"(pooled cells {row['pooled_cell_median']:+.3f}, n={row['pooled_cells']})  "
                  f"min {row['min_error']:+.3f}  "
                  f"negatives on {len(row['videos_with_a_negative_cell'])} video(s)")

    base = _find(args.results, "oracle_master_bernoulli")
    if base.exists():
        for coverage in (0.6, 0.7, 0.8, 0.9):
            table = dropout_table(base, coverage)
            row = table.get(0.0)
            if row:
                report.setdefault("coverage_sensitivity", {})[str(coverage)] = {
                    "videos": row["videos"],
                    "median": row["median_of_video_medians"],
                    "videos_with_a_negative_cell": row["videos_with_a_negative_cell"],
                }
        print("\n" + "=" * 78)
        print("coverage sensitivity at p=0 (the filter is post hoc, so report it)")
        print("=" * 78)
        for coverage, row in report["coverage_sensitivity"].items():
            print(f"  coverage>={coverage}: {row['videos']} videos, median {row['median']:+.3f}, "
                  f"negative on {row['videos_with_a_negative_cell'] or 'none'}")

        report["q_ratio"] = q_ratio(base)
        q = report["q_ratio"]
        print("\n" + "=" * 78)
        print("count ratio spread within one sequence, from L and tau alone")
        print("=" * 78)
        print(f"  worst: {q['video']}  q from {q['q_low']:.2f} to {q['q_high']:.2f} "
              f"= {q['ratio']:.2f}-fold  (e from {q['e_low']:+.3f} to {q['e_high']:+.3f})")
        print(f"  median across sequences: {q['median_ratio_across_videos']:.2f}-fold")

        report["phi"] = phi_table(base)
        print("\n" + "=" * 78)
        print("drift rate phi, annotated frames vs source frames")
        print("=" * 78)
        print(f"  source-frame range {report['phi']['source_range'][0]:.4f} to "
              f"{report['phi']['source_range'][1]:.4f} = {report['phi']['source_fold']:.0f}-fold")
        print(f"  annotated-frame fold {report['phi']['annotated_fold']:.0f} "
              f"(the cadence artefact the draft quoted)")

        report["blocked_validation"] = blocked_validation(base)
        bv = report["blocked_validation"]
        print("\n" + "=" * 78)
        print("drift model on NON-OVERLAPPING blocked windows vs a no-drift baseline")
        print("=" * 78)
        for video, mae_model, mae_flat, won in bv["rows"]:
            if mae_model is None:
                print(f"  {video:20s} too few blocked windows")
            else:
                print(f"  {video:20s} model {mae_model:7.2f}  no-drift {mae_flat:7.2f}   "
                      f"{'model wins' if won else 'BASELINE WINS'}")
        print(f"  model wins on {bv['wins']}/{bv['total']}")

    report["splits"] = split_table(args.results)
    print("\n" + "=" * 78)
    print("validation AP50 by split, checkpoint and video")
    print("=" * 78)
    for key, row in sorted(report["splits"].items()):
        print(f"  {key:12s} " + "  ".join(f"{k}={v}" for k, v in sorted(row.items())))

    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
