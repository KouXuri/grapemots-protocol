#!/usr/bin/env python3
"""Oracle OC-SORT and Deep OC-SORT: reference boxes as detections (raptor, CPU, nice 19).

Question (H4, written before the run, ledger 10.28.8): is the surplus of identities on
multi-view flights under perfect detection specific to BoT-SORT and StrongSORT, or does it
hold for an associator built to re-link after occlusion (OC-SORT) and for one that adds
appearance and camera-motion compensation (Deep OC-SORT)?
Inputs are the detection files oracle_strongsort_1120.py already wrote (reference boxes at
score 0.9, every annotated frame, all eleven videos); tools/boxmot_track_1120.py runs the
trackers at intervals 1, 2, 4, 8 (phase 0), one process at a time, waiting while memory is short
(the disease queue holds the GPUs and most of the RAM).

    nice -n 19 .venv/bin/python tools/oracle_boxmot_1120.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
DETS = ROOT / "runs/error_sources_1120/oracle_strongsort/dets"
OUT = ROOT / "runs/error_sources_1120"
BOX = "/home/kou/my_env/boxmot_venv/bin/python"
VIDEOS = [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]
TRACKERS = ("ocsort", "deepocsort")


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def status(state: str, **kw):
    (OUT / "oracle_boxmot_status.json").write_text(json.dumps(
        {"state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}))


def main():
    for trk in TRACKERS:
        for v in VIDEOS:
            dets = DETS / f"{v}_s0.json"
            done = all((OUT / f"oracle_{trk}/tracks/{v}_s0_d{d}.json").exists() for d in (1, 2, 4, 8))
            if done:
                continue
            while mem_available_gib() < 3.0:
                status("waiting_memory", tracker=trk, video=v, mem_gib=round(mem_available_gib(), 1))
                time.sleep(120)
            status("running", tracker=trk, video=v)
            subprocess.run(f"OMP_NUM_THREADS=2 {BOX} tools/boxmot_track_1120.py --tracker {trk} --dets {dets} "
                           f"--out-dir {OUT}/oracle_{trk}/tracks --device cpu", shell=True, cwd=ROOT, check=True)
    status("completed")


if __name__ == "__main__":
    main()
