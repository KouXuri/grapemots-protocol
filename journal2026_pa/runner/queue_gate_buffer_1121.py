#!/usr/bin/env python3
"""H5: two configuration-level tests of the duplicate mechanisms (raptor, CPU, nice 19). 2026-11-21.

Diagnosis so far (ledger 10.28.2, 10.28.6): with perfect detections, two-thirds of BoT-SORT's duplicates on
multi-view flights arise while the bunch stays in view, consistent with residual (parallax) motion pushing the
compensated box below the overlap gate; one-third are bunches that leave view for >= 2 s and come back.
Interventions, each changing one value of cfg/trackers/botsort_gmc.yaml (Ultralytics 8.4.46 BoT-SORT, unchanged):
  gate095  match_thresh 0.8 -> 0.95: the first-stage gate IoU x score >= 0.2 becomes >= 0.05
  buf150   track_buffer 30 -> 150: a lost track is kept for 150 processed frames instead of 30
Each runs tools/error_sources_1120.py replay-config --keep on the 35 seed-0 whole-frame dumps (cached GMC, the
harness verified bitwise on 11-20), giving the real detections ('none') and the reference boxes ('all') per arm.
First a self-check: the default config must reproduce the stored substitution values of one arm exactly.
Pre-registered readings (ledger 10.28.10, written before the run):
  H5a  oracle, multi-view, interval 1: duplicates classed 'switch in view' fall by >= 50% under gate095 ->
       the in-view duplicates are gate failures; < 20% -> the gate value is not their cause.
  H5b  'reentry beyond buffer' duplicates (absence >= 30 processed frames, same class definition for both runs)
       fall by >= 50% under buf150 -> memory length re-links them; < 20% -> motion and overlap cannot re-link
       a bunch that returns from another viewpoint.
  Real detections: the change of U, D, M, pooled e and the vine-level calibration error is reported as found.
    nohup nice -n 19 .venv/bin/python tools/queue_gate_buffer_1121.py > runs/error_sources_1120/logs/gate_buffer.log 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
OUT = ROOT / "runs/error_sources_1120/gate_buffer_1121"
DUMPS = ROOT / "runs/wholeframe_strongsort_1119/dets"
SUBST = ROOT / "runs/error_sources_1120/substitution/whole_s0"
BASE = ROOT / "cfg/trackers/botsort_gmc.yaml"
PY = ".venv/bin/python"
CFGS = {"gate095": ("match_thresh: 0.8", "match_thresh: 0.95"),
        "buf150": ("track_buffer: 30", "track_buffer: 150")}
ALL = "bkg+jit+loc+miss+score"


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def status(state: str, **kw):
    (OUT / "status.json").write_text(json.dumps({"state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}))


def run(dump: Path, cfg: Path, out: Path, tag: str, deltas=(1, 2, 4, 8), keep=True):
    cmd = (f"OMP_NUM_THREADS=1 {PY} tools/error_sources_1120.py replay-config --dets {dump} --cfg {cfg} "
           f"--out {out} --tag {tag} --deltas {' '.join(map(str, deltas))}" + (" --keep" if keep else ""))
    return subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=open(OUT / "logs" / f"{tag}.log", "w"),
                            stderr=subprocess.STDOUT)


def main():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    base = BASE.read_text()
    cfgs = {}
    for name, (a, b) in CFGS.items():
        assert base.count(a) == 1, (name, a)
        p = ROOT / f"cfg/trackers/botsort_gmc_{name}.yaml"
        p.write_text(f"# H5 intervention {name} (2026-11-21): one value changed from botsort_gmc.yaml\n" + base.replace(a, b))
        cfgs[name] = p
    # self-check: default config reproduces stored substitution values (PP2, sigma 3072, intervals 1 and 8)
    status("self_check")
    chk = OUT / "selfcheck"
    run(DUMPS / "PathPlanning_2_s3072.json", BASE, chk, "default__PathPlanning_2_s3072", deltas=(1, 8), keep=False).wait()
    got = json.loads((chk / "default__PathPlanning_2_s3072.json").read_text())["arms"]
    ref = json.loads((SUBST / "whole_s0__PathPlanning_2_s3072.json").read_text())["arms"]
    bad = [(d, k) for d in ("1", "8") for k in ("none", ALL)
           if {q: got[d][k][q] for q in ("P", "U", "D", "M")} != {q: ref[d]["values"][k][q] for q in ("P", "U", "D", "M")}]
    if bad:
        status("self_check_failed", mismatches=bad)
        return
    jobs = [(name, cfg, p) for name, cfg in cfgs.items() for p in sorted(DUMPS.glob("PathPlanning_*_s*.json"))]
    assert len(jobs) == 70, len(jobs)
    running = []
    for i, (name, cfg, p) in enumerate(jobs):
        tag = f"{name}__{p.stem}"
        if (OUT / name / f"{tag}.json").exists():
            continue
        while len(running) >= 3 or mem_available_gib() < 3.0:
            running = [r for r in running if r.poll() is None]
            status("running", done=i, total=len(jobs), mem_gib=round(mem_available_gib(), 1))
            time.sleep(20)
        running.append(run(p, cfg, OUT / name, tag))
    for r in running:
        r.wait()
    status("completed", total=len(jobs))


if __name__ == "__main__":
    main()
