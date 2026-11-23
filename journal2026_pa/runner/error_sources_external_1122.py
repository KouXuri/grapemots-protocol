#!/usr/bin/env python3
"""Error substitution on APPLE MOTS: the GrapeMOTS harness, unchanged, pointed at another dataset.

External check of "supplying the missed bunches alone made the count worse" (ledger 10.28.16).
tools/error_sources_1120.py is imported as it is; only its dataset constants are replaced
(DATA, OUT, CACHE and the frame list), so the error typing after TIDE, the five fixes, the
BoT-SORT configuration (cfg/trackers/botsort_gmc.yaml with cached sparse-optical-flow GMC), the
decomposition and the exact Shapley attribution are the ones the GrapeMOTS tables come from.

Detector: the YOLO26s checkpoint trained for the CBDCom AppleMOT replication
(runs/detect/caout_stage2_20260825/applemot_1280, imgsz 1280, seed 0, trained on sequences
0001-0005, 0000 for validation), read whole-frame at 640, 960 and 1280 by tools/dump_dets_0920.py
(confidence 0.25, IoU-0.5 merge) on the three distinct test sequences 0006-0008.

    .venv/bin/python tools/error_sources_external_1122.py dumps        # GPU, minutes
    nice -n 19 .venv/bin/python tools/error_sources_external_1122.py substitute
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / "tools"))
import error_sources_1120 as es  # noqa: E402

WEIGHTS = ROOT / "runs/detect/caout_stage2_20260825/applemot_1280/weights/best.pt"
VIDEOS = ("0006", "0007", "0008")
SCALES = (640, 960, 1280)
es.DATA = ROOT / "datasets/applemots_det"
es.OUT = ROOT / "runs/external_1122/apple"
es.CACHE = es.OUT / "gmc_cache"


def video_frames(video):
    return sorted(p for p in (es.DATA / "images/test").iterdir()
                  if p.name.split("__")[0] == video and p.suffix.lower() in {".png", ".jpg", ".jpeg"})


es.video_frames = video_frames


def dumps():
    for v in VIDEOS:
        for s in SCALES:
            out = es.OUT / "dumps" / f"{v}_s{s}.json"
            if out.exists():
                continue
            subprocess.run([sys.executable, "tools/dump_dets_0920.py", "--weights", str(WEIGHTS),
                            "--root", str(es.DATA), "--split", "test", "--video", v, "--imgsz", str(s),
                            "--out", str(out)], cwd=ROOT, check=True)


def substitute():
    for dp in sorted((es.OUT / "dumps").glob("*_s*.json")):
        es.cmd_substitute(argparse.Namespace(dets=str(dp), deltas=[1, 2, 4, 8],
                                             out=str(es.OUT / "substitution"), tag=None))


if __name__ == "__main__":
    {"dumps": dumps, "substitute": substitute}[sys.argv[1]]()
