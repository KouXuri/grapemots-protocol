#!/usr/bin/env python3
"""OC-SORT and Deep OC-SORT on the real seed-0 whole-frame detections (raptor, CPU, nice 19).

H4b (ledger 10.28.8, written before the run): does the vine-level failure of the count (per-vine
error after leave-one-vine-out ratio calibration; Spearman rho over the seven vines) hold for other
associator families on the same detections? Inputs are the 35 dumps BoT-SORT and StrongSORT already
read (runs/wholeframe_strongsort_1119/dets, conf 0.25, seven vines x five scales), so only the
associator changes. tools/boxmot_track_1120.py runs each tracker at intervals 1, 2, 4, 8.
Starts after tools/oracle_boxmot_1120.py has completed; one process at a time; waits while memory
is short.
    nohup nice -n 19 .venv/bin/python tools/real_boxmot_1120.py > runs/error_sources_1120/logs/real_boxmot.log 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
DUMPS = ROOT / "runs/wholeframe_strongsort_1119/dets"
OUT = ROOT / "runs/error_sources_1120"
BOX = "/home/kou/my_env/boxmot_venv/bin/python"
TRACKERS = ("ocsort", "deepocsort")


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def status(state: str, **kw):
    (OUT / "real_boxmot_status.json").write_text(json.dumps(
        {"state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}))


def main():
    st = OUT / "oracle_boxmot_status.json"
    while not (st.exists() and json.loads(st.read_text()).get("state") == "completed"):
        status("waiting_oracle")
        time.sleep(300)
    dumps = sorted(DUMPS.glob("PathPlanning_*_s*.json"))
    assert len(dumps) == 35, len(dumps)
    for trk in TRACKERS:
        for p in dumps:
            if all((OUT / f"real_{trk}/tracks/{p.stem}_d{d}.json").exists() for d in (1, 2, 4, 8)):
                continue
            while mem_available_gib() < 3.0:
                status("waiting_memory", tracker=trk, dump=p.stem, mem_gib=round(mem_available_gib(), 1))
                time.sleep(120)
            status("running", tracker=trk, dump=p.stem)
            subprocess.run(f"OMP_NUM_THREADS=2 {BOX} tools/boxmot_track_1120.py --tracker {trk} --dets {p} "
                           f"--out-dir {OUT}/real_{trk}/tracks --device cpu", shell=True, cwd=ROOT, check=True)
    status("completed")


if __name__ == "__main__":
    main()
