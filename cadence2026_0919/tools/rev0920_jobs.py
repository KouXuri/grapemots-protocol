#!/usr/bin/env python3
"""Write the job lists for the 2026-09-20 experiments (runs/rev_0920/jobs_*.tsv).

DEVICE=__CARD__ in the training jobs: Ultralytics' select_device() overwrites
CUDA_VISIBLE_DEVICES with the index it is given, so the GPU has to be chosen by
the training argument, not by the environment. The worker substitutes its card.

Each line: job_id <TAB> shell command. A worker claims a job with an atomic mkdir,
so two workers never run the same job. See tools/run_rev0920_worker.sh.

  first.tsv (card 1 only, short jobs first)
    apple_*      AppleMOT grid with the detector validated on sequence 0000
                 (runs/detect/caout_stage2_20260825/applemot_1280), 12 cells
    byte_s*      the ten-sequence surface with ByteTrack instead of BoT-SORT
    strong_*     the ten-sequence surface with StrongSORT (appearance association)
  pool.tsv (both cards)
    train_*      27 retrained checkpoints (tools/build_rev_lovo_0920.py)
    eval_*       each retrained checkpoint read on its own test video, 20 cells
"""
import json
from pathlib import Path

P = "/home/kou/my_env/yolo26"
Q = f"{P}/runs/rev_0920"
PY = f"{P}/.venv/bin/python"
BOX = "/home/kou/my_env/boxmot_venv/bin/python"
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
SIGMAS = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
MAP = json.load(open(f"{P}/runs/grapemots_journal_phase2/results/lovo_weights_map.json"))
TRACK = f"{PY} tools/track_grapemots_mot.py"
COMMON = "--detector-mode resize --conf 0.25 --match-iou 0.5 --save-frame-boxes --save-frame-tracks"


def cell(weights, root, split, video, tracker, sz, dl, out):
    vid = f"--videos {video}" if video else ""
    # parenthesised: an existing output skips the cell without breaking the chain
    return (f"( [ -f {out} ] || ( {TRACK} --weights {weights} --root {root} --split {split} {vid} "
            f"--tracker {tracker} --imgsz {sz} --global-imgsz {sz} --frame-step {dl} {COMMON} --out {out}.tmp "
            f"&& mv {out}.tmp {out} ) )")


first = []
AW = f"{P}/runs/detect/caout_stage2_20260825/applemot_1280/weights/best.pt"
for sz in (640, 960, 1280):
    for dl in DELTAS:
        out = f"{Q}/apple_val0000/apple_s{sz}_d{dl}.json"
        first.append((f"apple_s{sz}_d{dl}", cell(AW, "datasets/applemots_det", "test", "", "botsort.yaml", sz, dl, out)))
for sz in SIGMAS:
    cmds = [cell(MAP[v], "datasets/grapemots_det_721", "all", v, "bytetrack.yaml", sz, dl,
                 f"{Q}/bytetrack/{v}_s{sz}_d{dl}.json") for v in VIDEOS for dl in DELTAS]
    first.append((f"byte_s{sz}", " && ".join(cmds)))
for v in VIDEOS:
    cmds = []
    for sz in SIGMAS:
        dets = f"{Q}/dets/{v}_s{sz}.json"
        cmds.append(f"([ -f {dets} ] || {PY} tools/dump_dets_0920.py --weights {MAP[v]} --root datasets/grapemots_det_721 "
                    f"--video {v} --imgsz {sz} --out {dets})")
        cmds.append(f"{BOX} tools/strongsort_track_0920.py --dets {dets} --out-dir {Q}/strongsort --device cuda:0")
        cmds.append(f"rm -f {dets}")
    first.append((f"strong_{v}", " && ".join(cmds)))

pool = []
index = json.load(open(f"{P}/datasets/rev_lovo_0920/index.json"))
for r in index:
    seed = r["seed"]; run = r["run"]
    pool.append((f"train_{run}",
                 f"find datasets/rev_lovo_0920/{run}/labels -name '*.cache' -delete; "
                 f"DATA=datasets/rev_lovo_0920/{run}/data.yaml IMGSZ=1280 EPOCHS=15 PATIENCE=100 BATCH=8 DEVICE=__CARD__ "
                 f"WORKERS=1 LR0=0.001 LRF=0.01 COS_LR=False WARMUP=3.0 SEED={seed} SCALE=0.5 MOSAIC=0.0 "
                 f"PROJECT={P}/runs/detect/rev_lovo_0920 EXIST_OK=True bash tools/train_grapemots_det.sh yolo26s.pt {run}"))
for r in index:
    run = r["run"]; v = r["test"]
    w = f"{P}/runs/detect/rev_lovo_0920/{run}/weights/best.pt"
    wait = f"until [ -f {Q}/done/train_{run} ]; do sleep 120; done"
    cmds = [cell(w, "datasets/grapemots_det_721", "all", v, "cfg/trackers/botsort_gmc.yaml", sz, dl,
                 f"{Q}/retrain/{run}/{v}_s{sz}_d{dl}.json") for sz in SIGMAS for dl in DELTAS]
    pool.append((f"eval_{run}", wait + " && " + " && ".join(cmds)))

Path(Q).mkdir(parents=True, exist_ok=True)
for name, jobs in (("first", first), ("pool2", pool)):
    Path(f"{Q}/jobs_{name}.tsv").write_text("".join(f"{j}\t{c}\n" for j, c in jobs))
    print(name, len(jobs), "jobs")
