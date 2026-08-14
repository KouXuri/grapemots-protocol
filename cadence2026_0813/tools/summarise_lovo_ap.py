#!/usr/bin/env python3
"""Group the leave-one-video-out detection AP by source resolution.

The manuscript states that no AP is pooled across resolutions, so the two groups
are reported separately. Each value is that video's own out-of-fold AP: the
checkpoint that scored it never trained on it, and detections are the full-frame
boxes the tracker actually consumes, after tiling and merge.

    python3 tools/summarise_lovo_ap.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs/ap_lovo_0814/results"
HD = {"PathPlanning_1", "PathPlanning_3"}   # 1920x1080; the other nine are 3840x2160


def main() -> None:
    rows: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("ap_*.json")):
        video = path.stem[3:]
        overall = json.loads(path.read_text())["overall"]
        rows[video] = {
            "ap50": round(overall["ap50"], 4),
            "ap50_95": round(overall["ap50_95"], 4),
            "frames": overall["frames"],
            "gt": overall["gt"],
        }
    groups = {"3840x2160": sorted(v for v in rows if v not in HD),
              "1920x1080": sorted(v for v in rows if v in HD)}
    out = {"note": "out-of-fold, full-frame after tile merge, every 8th annotated frame",
           "per_video": rows, "groups": {}}
    for name, vids in groups.items():
        if not vids:
            continue
        out["groups"][name] = {
            "videos": len(vids),
            "median_ap50": round(statistics.median(rows[v]["ap50"] for v in vids), 3),
            "median_ap50_95": round(statistics.median(rows[v]["ap50_95"] for v in vids), 3),
            "min_ap50": round(min(rows[v]["ap50"] for v in vids), 3),
            "max_ap50": round(max(rows[v]["ap50"] for v in vids), 3),
        }
    (RESULTS / "lovo_ap_summary.json").write_text(json.dumps(out, indent=1))
    for name, g in out["groups"].items():
        lo, hi = g["min_ap50"], g["max_ap50"]
        print(f"{name}: {g['videos']} videos, median AP50 {g['median_ap50']}, "
              f"median AP50:95 {g['median_ap50_95']}, AP50 range {lo}-{hi}")


if __name__ == "__main__":
    main()
