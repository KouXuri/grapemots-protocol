#!/usr/bin/env python3
"""StrongSORT on the whole-frame, plant-disjoint seed-0 detectors (raptor).

Nothing new in the method: the September scripts run unchanged.
  1. tools/dump_dets_0920.py dumps every annotated frame's detections of one test
     video at one scale with the seed-0 whole-frame checkpoint of its fold, through
     the same resize_raw + merge_detections (IoU 0.5 NMS, confidence 0.25) functions
     track_grapemots_mot.py used for the main surface.
  2. tools/strongsort_track_0920.py (boxmot_venv, osnet_x0_25_msmt17, min_conf 0.25)
     replays them at intervals 1, 2, 4, 8, phase 0.
Grid: PathPlanning_2..8 x sigma 1536..3840 = 35 dumps, 140 StrongSORT arms.

The disease project has priority. The driver starts only when the seed-backfill queue
(reports/replay_seeds_20261119/status.json) has left the 'running' state, no process
has used either GPU for 15 continuous minutes, and MemAvailable >= 6 GiB; it re-checks
GPU idleness and memory before every job and waits again if anything else appears.

    setsid nohup nice -n 10 .venv/bin/python tools/queue_wholeframe_strongsort_1119.py &
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
Q = ROOT / "runs/wholeframe_strongsort_1119"
PY = ROOT / ".venv/bin/python"
BOX = "/home/kou/my_env/boxmot_venv/bin/python"
DISEASE = ROOT / "reports/replay_seeds_20261119/status.json"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)


def log(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (Q / "status.json").write_text(json.dumps(kw))
    with (Q / "driver.log").open("a") as f:
        f.write(json.dumps(kw) + "\n")


def gpu_busy() -> bool:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def mem_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def disease_running() -> bool:
    try:
        return json.loads(DISEASE.read_text()).get("state") == "running"
    except Exception:
        return True                      # unreadable status: assume it is still running


def wait_for_gpu(idle_minutes=15):
    while True:
        if disease_running():
            log(state="waiting_priority_disease")
            time.sleep(300)
            continue
        start = time.time()
        while time.time() - start < idle_minutes * 60:
            if gpu_busy() or mem_gib() < 6 or disease_running():
                break
            time.sleep(30)
        else:
            return
        log(state="waiting_idle_gpu", mem_gib=round(mem_gib(), 1))
        time.sleep(120)


def run(cmd: str, label: str):
    wait_for_gpu(idle_minutes=2)         # re-check before every job; long wait happens once at start
    log(state="running", job=label)
    with (Q / "logs" / f"{label}.log").open("w") as f:
        r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        log(state="failed", job=label, returncode=r.returncode)
        raise SystemExit(r.returncode)


def main():
    (Q / "logs").mkdir(parents=True, exist_ok=True)
    (Q / "dets").mkdir(exist_ok=True)
    wait_for_gpu(idle_minutes=15)
    for v in VIDEOS:
        weights = ROOT / f"runs/detect/fullframe_pponly_0922/{v}_s0/weights/best.pt"
        assert weights.is_file(), weights
        for s in SIGMAS:
            dets = Q / "dets" / f"{v}_s{s}.json"
            if not dets.exists():
                run(f"CUDA_VISIBLE_DEVICES=0 {PY} tools/dump_dets_0920.py --weights {weights} "
                    f"--root datasets/grapemots_det_721 --video {v} --imgsz {s} --out {dets}", f"dump_{v}_s{s}")
            if not all((Q / "strongsort" / f"{v}_s{s}_d{d}.json").exists() for d in (1, 2, 4, 8)):
                run(f"CUDA_VISIBLE_DEVICES=0 {BOX} tools/strongsort_track_0920.py --dets {dets} "
                    f"--out-dir {Q}/strongsort --device cuda:0", f"strongsort_{v}_s{s}")
    log(state="completed")


if __name__ == "__main__":
    main()
