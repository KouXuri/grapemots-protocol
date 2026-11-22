#!/usr/bin/env python3
"""GPU queue: detection dumps for the five detector sets not yet dumped (raptor).

Needed because a CPU dump is not the GPU dump: on PathPlanning_2 at 3072 px, 5 of 209
frames differ in detection count and scores differ by up to 0.012
(runs/error_sources_1120/cpu_check), so only GPU dumps replay the main grid exactly.

Code: tools/dump_dets_0920.py unchanged (resize_raw + merge_detections, the main grid's
detector path). Confidence 0.05 is dumped so that thresholds above it stay replayable;
error_sources_1120.py keeps only scores >= 0.25 for the main-grid input. Job 0 checks that
this filter reproduces the existing 0.25 dump of PathPlanning_2 (seed-0 whole-frame)
box for box; if it does not, every later dump is taken at 0.25 instead.

The disease project has priority: the queue starts only after no process has used either
GPU for 15 continuous minutes and MemAvailable >= 6 GiB, and re-checks before every dump.

    setsid nohup nice -n 10 .venv/bin/python tools/queue_dumps_1120.py &
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
Q = ROOT / "runs/error_sources_1120"
OUT = Q / "dets"
PY = ROOT / ".venv/bin/python"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)]
SETS = [("crop", s, "runs/detect/rev_lovo_0920") for s in (0, 1, 2)] + \
       [("whole", s, "runs/detect/fullframe_pponly_0922") for s in (1, 2)]
SIGMA = 3072


def log(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (Q / "dets_status.json").write_text(json.dumps(kw))
    with (Q / "dets_queue.log").open("a") as f:
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


def wait_for_gpu(idle_minutes):
    while True:
        start = time.time()
        while time.time() - start < idle_minutes * 60:
            if gpu_busy() or mem_gib() < 6:
                break
            time.sleep(30)
        else:
            return
        log(state="waiting_idle_gpu", mem_gib=round(mem_gib(), 1))
        time.sleep(120)


def dump(weights, video, conf, dest, label):
    wait_for_gpu(idle_minutes=2)
    log(state="running", job=label)
    with (Q / "logs" / f"{label}.log").open("w") as f:
        r = subprocess.run(f"CUDA_VISIBLE_DEVICES=0 {PY} tools/dump_dets_0920.py --weights {weights} "
                           f"--root datasets/grapemots_det_721 --video {video} --imgsz {SIGMA} "
                           f"--conf {conf} --out {dest}", shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        log(state="failed", job=label, returncode=r.returncode)
        raise SystemExit(r.returncode)


def same_after_filter(low, ref):
    a = json.loads(Path(low).read_text())["frames"]
    b = json.loads(Path(ref).read_text())["frames"]
    for fa, fb in zip(a, b):
        keep = [(bx, s) for bx, s in zip(fa["boxes"], fa["scores"]) if s >= 0.25]
        if [k[0] for k in keep] != fb["boxes"] or [k[1] for k in keep] != fb["scores"]:
            return False
    return len(a) == len(b)


def main():
    (Q / "logs").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (Q / "dets_tmp").mkdir(exist_ok=True)
    wait_for_gpu(idle_minutes=15)
    check = Q / "cpu_check" / "PathPlanning_2_s3072_gpu_conf005.json"
    if not check.exists():
        dump(ROOT / "runs/detect/fullframe_pponly_0922/PathPlanning_2_s0/weights/best.pt", "PathPlanning_2",
             0.05, check, "check_conf005")
    ok = same_after_filter(check, ROOT / "runs/wholeframe_strongsort_1119/dets/PathPlanning_2_s3072.json")
    conf = 0.05 if ok else 0.25
    log(state="checked", low_conf_filter_reproduces_025_dump=ok, conf_used=conf)
    for fam, seed, base in SETS:
        for v in VIDEOS:
            dest = OUT / f"{fam}_s{seed}__{v}_s{SIGMA}.json"
            if not dest.exists():
                tmp = Q / "dets_tmp" / dest.name          # outside dets/, which the replay queue globs
                dump(ROOT / base / f"{v}_s{seed}/weights/best.pt", v, conf, tmp, f"dump_{fam}_s{seed}_{v}")
                tmp.rename(dest)
    log(state="completed", conf_used=conf)


if __name__ == "__main__":
    main()
