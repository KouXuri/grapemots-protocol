#!/usr/bin/env python3
"""Error substitution on the LettuceMOTS return passes with a real detector (ledger 10.28.17, check L).

tools/error_sources_1120.py is imported as it is; only its dataset constants (DATA, OUT, CACHE) and the
frame list are replaced, exactly as tools/error_sources_external_1122.py does for APPLE MOTS. So the error
typing after TIDE, the five fixes, the BoT-SORT configuration (cfg/trackers/botsort_gmc.yaml with cached
sparse-optical-flow GMC), the decomposition and the exact Shapley attribution are the ones behind the
GrapeMOTS tables.

Data: datasets/lettucemots_det_official, the authors' split (train and test folders of the LettuceMOTS
release), built by tools/queue_pa_followup_1124.py --preflight with the GrapeMOTS converter. Test
sequences 0003, 0007 and 0011, all three return passes.

    .venv/bin/python tools/error_sources_lettuce_1124.py gmc
    nice -n 19 .venv/bin/python tools/error_sources_lettuce_1124.py substitute --dets DUMP.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / "tools"))
import error_sources_1120 as es  # noqa: E402

TEST = ("0003", "0007", "0011")
DELTAS = (1, 2, 4, 8)
es.DATA = ROOT / "datasets/lettucemots_det_official"
es.OUT = ROOT / "runs/pa_followup_1124/lettuce"
es.CACHE = es.OUT / "gmc_cache"


def video_frames(video):
    return sorted(p for p in (es.DATA / "images/test").iterdir()
                  if p.name.split("__")[0] == video and p.suffix.lower() in {".png", ".jpg", ".jpeg"})


es.video_frames = video_frames      # gmc_cache and size_of look the name up at call time


def gmc(_a):
    for v in TEST:
        for d in DELTAS:
            print(es.gmc_cache(v, d), flush=True)


def substitute(a):
    es.cmd_substitute(argparse.Namespace(dets=a.dets, deltas=list(DELTAS),
                                         out=str(es.OUT / "substitution"), tag=None))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("gmc")
    s = sp.add_parser("substitute")
    s.add_argument("--dets", required=True)
    args = ap.parse_args()
    {"gmc": gmc, "substitute": substitute}[args.cmd](args)
