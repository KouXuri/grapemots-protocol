#!/usr/bin/env python3
"""Queue for the error-source study (raptor, CPU only, lowest priority).

  A. wait until the GMC cache holds all 44 (video, interval) warp files
  B. verify: replay from the cache must reproduce the 140 seed-0 whole-frame main-grid
     arms and the 44 phase-0 oracle arms identity for identity; the queue stops otherwise
  C. substitution (32 fix subsets x 4 intervals) on the 35 seed-0 whole-frame dumps
     (runs/wholeframe_strongsort_1119/dets), 3 processes at a time
  D. substitution on any further dump that appears in runs/error_sources_1120/dets
     (other detector sets; filled by a separate dump queue), until that queue reports done

The disease training shares the machine and has priority: every worker is nice 19 /
ionice idle and waits while MemAvailable < 2.5 GiB (error_sources_1120.wait_for_memory).

    setsid nohup nice -n 19 .venv/bin/python tools/queue_error_sources_1120.py &
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
Q = ROOT / "runs/error_sources_1120"
PY = ROOT / ".venv/bin/python"
TOOL = "tools/error_sources_1120.py"
sys.path.insert(0, str(ROOT / "tools"))
from error_sources_1120 import wait_for_memory  # noqa: E402


def log(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (Q / "status.json").write_text(json.dumps(kw))
    with (Q / "queue.log").open("a") as f:
        f.write(json.dumps(kw) + "\n")


def run(args, label):
    wait_for_memory()
    with (Q / "logs" / f"{label}.log").open("w") as f:
        r = subprocess.run(f"OMP_NUM_THREADS=1 nice -n 19 ionice -c3 {PY} {TOOL} {args}",
                           shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        log(state="failed", job=label, returncode=r.returncode)
    return r.returncode


def substitute_all(dumps, out, tag_prefix, workers=3):
    jobs = []
    for d in dumps:
        tag = f"{tag_prefix}{d.stem}"
        if not (out / f"{tag}.json").exists():
            jobs.append((f"substitute --dets {d} --out {out} --tag {tag}", f"sub_{tag}"))
    with ThreadPoolExecutor(workers) as ex:
        codes = list(ex.map(lambda j: run(*j), jobs))
    return sum(1 for c in codes if c)


def main():
    (Q / "logs").mkdir(parents=True, exist_ok=True)
    log(state="waiting_gmc_cache")
    while len(list((Q / "gmc_cache").glob("*_o0.npy"))) < 44:
        time.sleep(60)
    if not (Q / "verify.json").exists():
        log(state="running", job="verify")
        if run("verify", "verify"):
            raise SystemExit(1)
    rep = json.loads((Q / "verify.json").read_text())
    bad_main = [k for k, x in rep["main_grid"].items() if not (x["ids"] and x["names"])]
    bad_orc = [k for k, x in rep["oracle"].items() if not x["ids"]]
    log(state="verified", main_grid_mismatch=bad_main, oracle_mismatch=bad_orc)
    if bad_main:
        raise SystemExit("replay does not reproduce the main grid; stopping")
    log(state="running", job="substitution_whole_s0")
    dumps = sorted((ROOT / "runs/wholeframe_strongsort_1119/dets").glob("PathPlanning_*_s*.json"))
    failed = substitute_all(dumps, Q / "substitution/whole_s0", "whole_s0__")
    log(state="running", job="substitution_other_sets", failed_so_far=failed)
    while True:
        extra = sorted((Q / "dets").glob("*.json")) if (Q / "dets").exists() else []
        failed += substitute_all(extra, Q / "substitution/other", "")
        done = (Q / "dets_status.json").exists() and \
            json.loads((Q / "dets_status.json").read_text()).get("state") == "completed"
        if done and all((Q / "substitution/other" / f"{d.stem}.json").exists() for d in extra):
            break
        time.sleep(300)
    log(state="completed", failed=failed)


if __name__ == "__main__":
    main()
