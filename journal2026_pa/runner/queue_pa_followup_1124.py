#!/usr/bin/env python3
"""Pre-submission checks for the Precision Agriculture manuscript, queued 2026-11-24.

Rules for reading the results were written before any run: RESEARCH_AUDIT_2026-09-05.md, 10.28.17.

L  LettuceMOTS with a real detector, on the authors' own split (train and test folders of the release).
   Test sequences 0003, 0007, 0011 are the three return passes (G = 40, 37, 37). Validation sequence by a
   rule fixed beforehand: the highest-numbered training sequence, 0010. Boxes from the masks with the
   GrapeMOTS converter (tools/create_grapemots_detection_dataset.py, min-area 20); every third annotated
   frame for training and validation, as for GrapeMOTS (make_split_manifests.select). YOLO26s from
   yolo26s.pt with the paper's crop-detector recipe (the train line of tools/rev0920_jobs.py) except
   imgsz 1088, the native long side of the 810x1080 frames. Whole frames read at 272/544/816/1088
   (0.25-1.0 of native), dumped at confidence 0.05 by tools/dump_dets_0920.py on GPU, then error
   substitution by tools/error_sources_lettuce_1124.py (the GrapeMOTS harness, paths changed only).

M  A larger detector on GrapeMOTS: YOLO26m crop detectors on the paper's plant-disjoint folds
   (datasets/rev_lovo_0920/<video>_s<seed>), the recipe of the crop detectors word for word with only
   the initial weights changed (yolo26m.pt), seeds 0, 1, 2 (all of seed 0 first).
   Each checkpoint is dumped on its own test video at the five inference scales (GPU, confidence 0.05)
   and replayed by tools/error_sources_1120.py: replay-config with the main-grid tracker (no fix, miss,
   miss+score, all) at every scale, and the 32-subset substitution at 3072, as for the six detector sets.

R  The batch of one published checkpoint. Ultralytics 8.4.46 halves the batch when the first epoch runs out
   of GPU memory and only warns (engine/trainer.py, the OutOfMemoryError branch): the crop checkpoint
   PathPlanning_4_s2 of the paper was trained at batch 4 (runs/rev_0920/logs/train_PathPlanning_4_s2.log,
   "Reducing to batch=4"), while the manuscript says batch 8 for all crop detectors. It is retrained at
   batch 8 on an idle GPU, same data, seed and recipe, and read and replayed like M, to measure what the
   lower batch changed. For every training here the batch actually used is parsed from the log and
   written to trained.json next to the requested one.

The disease project has priority. GPU work starts only after its two controllers
(tools/run_ldd_stage_20261123.py, tools/run_third_detector_20261123.py) have exited and both GPUs have
had no compute process for 15 minutes with at least 8 GiB of RAM available. Before every later GPU job
the queue waits while any disease controller (tools/run_*_2026MMDD.py) is alive, so a disease job that
starts in the meantime gets the GPUs first. CPU replays run at most two at a time, under a RAM floor.

    .venv/bin/python tools/queue_pa_followup_1124.py --preflight     # CPU only: checks, data, plan
    tmux new-session -d -s pa_followup_1124 -c /home/kou/my_env/yolo26 \\
        '.venv/bin/python -u tools/queue_pa_followup_1124.py --run >> runs/pa_followup_1124/controller.log 2>&1'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/kou/my_env/yolo26")
OUT = ROOT / "runs/pa_followup_1124"
PY = ROOT / ".venv/bin/python"

# ---- L: LettuceMOTS, authors' split
L_SRC = Path("/home/kou/my_env/lettucemots")          # release: {train,test}/{images,instances}/<seq>
L_SHIM = Path("/home/kou/my_env/LettuceMOTS_shim")    # symlink tree from tools/build_lettuce_shim_1122.py
L_DATA = ROOT / "datasets/lettucemots_det_official"
L_TRAIN = ("0000", "0001", "0002", "0004", "0005", "0006", "0008", "0009")
L_VAL = ("0010",)
L_TEST = ("0003", "0007", "0011")
L_G = {"0003": 40, "0007": 37, "0011": 37}             # reference plants, as in the oracle check (10.28.16)
L_IMGSZ = 1088
L_SCALES = (272, 544, 816, 1088)
L_PROJECT = ROOT / "runs/detect/lettuce_official_1124"
L_NAME = "yolo26s_official_s0"

# ---- M: YOLO26m crop detectors on GrapeMOTS
M_WEIGHTS = ROOT / "yolo26m.pt"
M_SHA = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
S_WEIGHTS = ROOT / "yolo26s.pt"
S_SHA = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
M_PROJECT = ROOT / "runs/detect/m_lovo_1124"
VIDEOS = tuple(f"PathPlanning_{i}" for i in range(2, 9))
SEEDS = (0, 1, 2)
SIGMAS = (1536, 2048, 2560, 3072, 3840)
TRACKER = "cfg/trackers/botsort_gmc.yaml"
R_RUN = "PathPlanning_4_s2"
R_PROJECT = ROOT / "runs/detect/rev_lovo_0920_b8check_1124"

DISEASE_FIRST = ("run_ldd_stage_20261123.py", "run_third_detector_20261123.py")
DISEASE_RE = re.compile(r"tools/run_[A-Za-z0-9_]+_2026\d{4}\.py")
LOCK = threading.Lock()


# ------------------------------------------------------------------ bookkeeping
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(state, **kw):
    row = dict(state=state, updated=now(), **kw)
    with LOCK:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "status.json").write_text(json.dumps(row) + "\n")
        with (OUT / "queue.log").open("a") as f:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def done(job):
    return (OUT / "done" / job).exists()


def mark(job, **kw):
    (OUT / "done").mkdir(parents=True, exist_ok=True)
    (OUT / "done" / job).write_text(json.dumps(dict(finished=now(), **kw)) + "\n")


def sh(cmd, logname, env=None):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with (OUT / "logs" / f"{logname}.log").open("a") as f:
        f.write(f"\n# {now()}\n# {cmd}\n")
        f.flush()
        r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env=None if env is None else {**os.environ, **env})
    return r.returncode


# ------------------------------------------------------------------ priority gate
def mem_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def disease_controllers():
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if DISEASE_RE.search(cmd) and "queue_pa_followup_1124" not in cmd:
            found.append((int(p.name), DISEASE_RE.search(cmd).group(0)))
    return found


def gpu_clients(card):
    out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return [x for x in out.splitlines() if x.strip()]


LAST_WAIT = {}


def wait_gate(cards, idle_minutes, ram_gib, label):
    """Return once no disease controller is alive and the cards have been idle for idle_minutes."""
    while True:
        alive = disease_controllers()
        if alive:
            names = sorted({c for _, c in alive})
            if LAST_WAIT.get(label) != names:
                LAST_WAIT[label] = names
                log("waiting_disease", job=label, controllers=names)
            time.sleep(300)
            continue
        start, ok = time.time(), True
        while time.time() - start < idle_minutes * 60:
            if disease_controllers() or any(gpu_clients(c) for c in cards) or mem_gib() < ram_gib:
                ok = False
                break
            time.sleep(30)
        if ok:
            LAST_WAIT.pop(label, None)
            return
        if LAST_WAIT.get(label) != "idle":
            LAST_WAIT[label] = "idle"
            log("waiting_idle_gpu", job=label, cards=list(cards), mem_gib=round(mem_gib(), 1))
        time.sleep(120)


# ------------------------------------------------------------------ preflight (CPU only)
def every_third(split, seqs):
    rows = []
    for seq in seqs:
        frames = sorted(p for p in (L_DATA / "images" / split).iterdir() if p.name.split("__")[0] == seq)
        if not frames:
            raise SystemExit(f"no frames for {seq} in {split}")
        rows += [str(p.absolute()) for p in frames[::3]]
    return rows


def lettuce_reference(seq):
    sys.path.insert(0, str(ROOT / "tools"))
    import cv2
    from track_grapemots_mot import load_gt_tracks
    ids = set()
    for p in sorted((L_DATA / "images/test").iterdir()):
        if p.name.split("__")[0] != seq:
            continue
        h, w = cv2.imread(str(p)).shape[:2]
        gi, _ = load_gt_tracks(L_DATA / "tracks/test" / f"{p.stem}.txt", w, h)
        ids |= set(gi)
    return len(ids)


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    problems = []
    if sha(M_WEIGHTS) != M_SHA:
        problems.append("yolo26m.pt sha256 changed")
    if sha(S_WEIGHTS) != S_SHA:
        problems.append("yolo26s.pt sha256 changed")

    # M: the paper's crop folds must be intact (images and labels of every listed tile)
    folds = {}
    for seed in SEEDS:
        for v in VIDEOS:
            run = f"{v}_s{seed}"
            d = ROOT / "datasets/rev_lovo_0920" / run
            spec = json.loads((d / "split_spec.json").read_text())
            missing = 0
            for part in ("train", "val"):
                for line in (d / f"{part}.txt").read_text().split():
                    img = Path(line)
                    lab = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
                    missing += (not img.exists()) + (not lab.exists())
            if missing:
                problems.append(f"{run}: {missing} missing tiles or labels")
            folds[run] = dict(train=spec["train"], val=spec["val"], test=spec["test"],
                              train_tiles=spec["train_tiles"], val_tiles=spec["val_tiles"],
                              reference_checkpoint=str(ROOT / f"runs/detect/rev_lovo_0920/{run}/weights/best.pt"))
            if (spec["seed"], spec["test"]) != (seed, [v]):
                problems.append(f"{run}: split_spec does not match")

    # L: release split as shipped, then the converter, then every third frame
    shipped = {s: sorted(p.name for p in (L_SRC / s / "images").iterdir() if p.is_dir()) for s in ("train", "test")}
    if shipped["train"] != sorted(L_TRAIN + L_VAL) or shipped["test"] != sorted(L_TEST):
        problems.append(f"LettuceMOTS split differs from the plan: {shipped}")
    split_json = OUT / "lettuce" / "split_official.json"
    split_json.parent.mkdir(parents=True, exist_ok=True)
    split_json.write_text(json.dumps({"train": list(L_TRAIN), "val": list(L_VAL), "test": list(L_TEST)}, indent=1) + "\n")
    if not (L_DATA / "dataset_stats.json").exists():
        rc = sh(f"nice -n 19 {PY} tools/create_grapemots_detection_dataset.py --src {L_SHIM} --dst {L_DATA} "
                f"--classes lettuce --split-json {split_json}", "lettuce_dataset")
        if rc:
            raise SystemExit(f"converter failed ({rc}); see {OUT / 'logs/lettuce_dataset.log'}")
    train, val = every_third("train", L_TRAIN), every_third("val", L_VAL)
    (L_DATA / "train_official.txt").write_text("\n".join(train) + "\n")
    (L_DATA / "val_official.txt").write_text("\n".join(val) + "\n")
    (L_DATA / "data_official.yaml").write_text(
        f"path: {L_DATA}\ntrain: train_official.txt\nval: val_official.txt\nnc: 1\nnames: ['lettuce']\n")
    G = {s: lettuce_reference(s) for s in L_TEST}
    if G != L_G:
        problems.append(f"lettuce reference counts {G} differ from the oracle check {L_G}")

    plan = dict(
        written=now(), ledger="RESEARCH_AUDIT_2026-09-05.md 10.28.17 (rules fixed before any run)",
        priority=dict(first=list(DISEASE_FIRST), gate="both GPUs free of compute processes for 15 min, RAM >= 8 GiB",
                      later="before each GPU job: no tools/run_*_2026MMDD.py alive, card idle 2 min, RAM >= 6 GiB"),
        lettuce=dict(source=str(L_SRC), train=list(L_TRAIN), val=list(L_VAL), test=list(L_TEST),
                     reference=G, train_frames=len(train), val_frames=len(val),
                     weights=str(S_WEIGHTS), weights_sha256=S_SHA, imgsz=L_IMGSZ, scales=list(L_SCALES),
                     recipe="tools/rev0920_jobs.py train line: EPOCHS=15 PATIENCE=100 BATCH=8 WORKERS=1 LR0=0.001 "
                            "LRF=0.01 COS_LR=False WARMUP=3.0 SEED=0 SCALE=0.5 MOSAIC=0.0 (IMGSZ=1088)",
                     dump_conf=0.05, replay="tools/error_sources_lettuce_1124.py substitute, deltas 1 2 4 8"),
        larger_detector=dict(weights=str(M_WEIGHTS), weights_sha256=M_SHA, seeds=list(SEEDS), folds=folds,
                             recipe="tools/rev0920_jobs.py train line with yolo26m.pt: IMGSZ=1280 EPOCHS=15 "
                                    "PATIENCE=100 BATCH=8 WORKERS=1 LR0=0.001 LRF=0.01 COS_LR=False WARMUP=3.0 "
                                    "SEED=<seed> SCALE=0.5 MOSAIC=0.0",
                             batch_check="batch actually used parsed from the training log (Ultralytics halves it on a "
                                         "first-epoch OOM and only warns), written to trained.json",
                             sigmas=list(SIGMAS), dump_conf=0.05, tracker=TRACKER,
                             replay=["error_sources_1120.py replay-config (none, miss, miss+score, all) at every scale",
                                     "error_sources_1120.py substitute (32 subsets) at 3072"]),
        batch_check_retrain=dict(run=R_RUN, reason="runs/rev_0920/logs/train_PathPlanning_4_s2.log: 'Reducing to batch=4'",
                                 weights=str(S_WEIGHTS), weights_sha256=S_SHA, seed=2, batch=8,
                                 data=str(ROOT / "datasets/rev_lovo_0920" / R_RUN / "data.yaml"),
                                 published_checkpoint=str(ROOT / f"runs/detect/rev_lovo_0920/{R_RUN}/weights/best.pt"),
                                 sigmas=list(SIGMAS)),
        problems=problems)
    plan_path = OUT / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        keys = ("lettuce", "larger_detector", "batch_check_retrain")
        if {k: old.get(k) for k in keys} != {k: plan[k] for k in keys}:
            raise SystemExit("plan.json exists and differs: the plan is fixed once written")
    else:
        plan_path.write_text(json.dumps(plan, indent=1) + "\n")
    log("preflight_passed" if not problems else "preflight_failed", problems=problems,
        lettuce_reference=G, lettuce_train_frames=len(train), lettuce_val_frames=len(val))
    return not problems


# ------------------------------------------------------------------ GPU jobs
def train_env(data, imgsz, batch, card, seed, project):
    return dict(DATA=str(data), IMGSZ=str(imgsz), EPOCHS="15", PATIENCE="100", BATCH=str(batch), DEVICE=str(card),
                WORKERS="1", LR0="0.001", LRF="0.01", COS_LR="False", WARMUP="3.0", SEED=str(seed),
                SCALE="0.5", MOSAIC="0.0", PROJECT=str(project), EXIST_OK="True")


BATCH_RE = re.compile(r"Reducing to batch=(\d+)")


def batch_used(logname, requested):
    """The batch Ultralytics actually trained with: it halves the batch on a first-epoch OOM and only warns."""
    text = (OUT / "logs" / f"{logname}.log").read_text(errors="replace")
    text = text[text.rfind("# bash tools/train_grapemots_det.sh"):]      # the last attempt only (log is appended)
    found = BATCH_RE.findall(text)
    return int(found[-1]) if found else requested


def finished_training(folder, epochs=15):
    best = folder / "weights/best.pt"
    res = folder / "results.csv"
    if not best.exists() or not res.exists():
        return False
    return len(res.read_text().strip().splitlines()) - 1 == epochs


def train(job, weights, data, imgsz, card, seed, project, name, label_dirs, batch=8):
    folder = project / name
    logname = f"{job}_train"
    if not finished_training(folder):
        for d in label_dirs:
            for c in Path(d).rglob("*.cache"):
                c.unlink()
        log("training", job=job, card=card, batch=batch)
        rc = sh(f"bash tools/train_grapemots_det.sh {weights} {name}", logname,
                env=train_env(data, imgsz, batch, card, seed, project))
        if rc or not finished_training(folder):
            log("failed", job=job, card=card, step="train", returncode=rc)
            return None
    best = folder / "weights/best.pt"
    used = batch_used(logname, batch)
    (folder / "trained.json").write_text(json.dumps(dict(
        job=job, weights=str(best), sha256=sha(best), initial=str(weights), initial_sha256=sha(weights),
        batch_requested=batch, batch_used=used, seed=seed, imgsz=imgsz, data=str(data), card=card,
        finished=now()), indent=1) + "\n")
    if used != batch:
        log("batch_reduced", job=job, card=card, requested=batch, used=used)
    return best


def dump(job, weights, root, split, video, imgsz, card, dest):
    if dest.exists():
        return True
    tmp = dest.parent.parent / "dets_tmp" / dest.name      # outside dets/, which the CPU loop globs
    tmp.parent.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rc = sh(f"CUDA_VISIBLE_DEVICES={card} {PY} tools/dump_dets_0920.py --weights {weights} --root {root} "
            f"--split {split} --video {video} --imgsz {imgsz} --conf 0.05 --out {tmp}", f"{job}_dump")
    if rc or not tmp.exists():
        log("failed", job=job, card=card, step=f"dump {video} s{imgsz}", returncode=rc)
        return False
    tmp.rename(dest)
    return True


def job_lettuce(card):
    w = train("L", S_WEIGHTS, L_DATA / "data_official.yaml", L_IMGSZ, card, 0, L_PROJECT, L_NAME,
              [L_DATA / "labels"])
    if w is None:
        return False
    ok = all(dump("L", w, L_DATA, "test", seq, s, card, OUT / "lettuce/dets" / f"{seq}_s{s}.json")
             for seq in L_TEST for s in L_SCALES)
    return ok


def job_m(seed, video, card):
    run = f"{video}_s{seed}"
    d = ROOT / "datasets/rev_lovo_0920" / run
    w = train(f"M_{run}", M_WEIGHTS, d / "data.yaml", 1280, card, seed, M_PROJECT, run, [d / "labels"])
    if w is None:
        return False
    return all(dump(f"M_{run}", w, ROOT / "datasets/grapemots_det_721", "all", video, s, card,
                    OUT / "m/dets" / f"m_s{seed}__{video}_s{s}.json") for s in SIGMAS)


def job_r(card):
    d = ROOT / "datasets/rev_lovo_0920" / R_RUN
    w = train("R", S_WEIGHTS, d / "data.yaml", 1280, card, 2, R_PROJECT, R_RUN, [d / "labels"])
    if w is None:
        return False
    video = R_RUN.rsplit("_s", 1)[0]
    return all(dump("R", w, ROOT / "datasets/grapemots_det_721", "all", video, s, card,
                    OUT / "r/dets" / f"r_s2__{video}_s{s}.json") for s in SIGMAS)


def gpu_worker(card, jobs):
    while True:
        with LOCK:
            pending = [j for j in jobs if not j["claimed"]]
            if not pending:
                return
            job = pending[0]
            job["claimed"] = True
        if done(job["id"]):
            continue
        wait_gate([card], 2, 6, job["id"])
        log("running", job=job["id"], card=card)
        ok = job["fn"](card)
        if ok:
            mark(job["id"], card=card)
            log("job_done", job=job["id"], card=card)


# ------------------------------------------------------------------ CPU replays
def replay_commands():
    cmds = []
    for dp in sorted((OUT / "lettuce/dets").glob("*_s*.json")):
        res = OUT / "lettuce/substitution" / dp.name
        if not res.exists():
            cmds.append((f"L_sub_{dp.stem}", f"nice -n 19 {PY} tools/error_sources_lettuce_1124.py substitute --dets {dp}"))
    for part, prefix in (("m", "M"), ("r", "R")):
        for dp in sorted((OUT / part / "dets").glob("*__*_s*.json")):
            if not (OUT / part / "replay" / dp.name).exists():
                cmds.append((f"{prefix}_rep_{dp.stem}",
                             f"nice -n 19 {PY} tools/error_sources_1120.py replay-config --dets {dp} --cfg {TRACKER} "
                             f"--deltas 1 2 4 8 --out {OUT / part / 'replay'} --tag {dp.stem}"))
            if dp.stem.endswith("_s3072") and not (OUT / part / "substitution" / dp.name).exists():
                cmds.append((f"{prefix}_sub_{dp.stem}",
                             f"nice -n 19 {PY} tools/error_sources_1120.py substitute --dets {dp} "
                             f"--deltas 1 2 4 8 --out {OUT / part / 'substitution'} --tag {dp.stem}"))
    return cmds


def cpu_loop(gpu_threads, max_parallel=2, floor_gib=3.0):
    running = {}
    failed = set()
    while True:
        for name, p in list(running.items()):
            if p.poll() is not None:
                if p.returncode:
                    failed.add(name)
                    log("failed", job=name, step="replay", returncode=p.returncode)
                running.pop(name)
        todo = [(n, c) for n, c in replay_commands() if n not in running and n not in failed]
        if not todo and not running and not any(t.is_alive() for t in gpu_threads):
            return failed
        while todo and len(running) < max_parallel and mem_gib() >= floor_gib:
            n, c = todo.pop(0)
            (OUT / "logs").mkdir(parents=True, exist_ok=True)
            f = (OUT / "logs" / f"{n}.log").open("a")
            running[n] = subprocess.Popen(c, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        time.sleep(60)


def run():
    if not preflight():
        raise SystemExit("preflight failed; see status.json")
    sh(f"nice -n 19 {PY} tools/error_sources_lettuce_1124.py gmc", "lettuce_gmc")   # CPU, before the GPU wait
    jobs = [dict(id="L", fn=job_lettuce, claimed=False), dict(id="R", fn=job_r, claimed=False)]
    jobs += [dict(id=f"M_{v}_s{s}", fn=(lambda card, s=s, v=v: job_m(s, v, card)), claimed=False)
             for s in SEEDS for v in VIDEOS]
    if not all(done(j["id"]) for j in jobs):
        wait_gate([0, 1], 15, 8, "start")
    threads = [threading.Thread(target=gpu_worker, args=(c, jobs), daemon=True) for c in (0, 1)]
    for t in threads:
        t.start()
        time.sleep(90)            # let the first job claim its card before the second gate check
    failed = cpu_loop(threads)
    for t in threads:
        t.join()
    summary = dict(
        finished=now(),
        jobs={j["id"]: done(j["id"]) for j in jobs},
        lettuce=dict(dets=sorted(p.name for p in (OUT / "lettuce/dets").glob("*.json")),
                     substitution=sorted(p.name for p in (OUT / "lettuce/substitution").glob("*_s*.json")
                                         if "_d" not in p.stem)),
        m=dict(dets=sorted(p.name for p in (OUT / "m/dets").glob("*.json")),
               replay=sorted(p.name for p in (OUT / "m/replay").glob("*.json")),
               substitution=sorted(p.name for p in (OUT / "m/substitution").glob("*.json") if "_d" not in p.stem)),
        r=dict(dets=sorted(p.name for p in (OUT / "r/dets").glob("*.json")),
               replay=sorted(p.name for p in (OUT / "r/replay").glob("*.json")),
               substitution=sorted(p.name for p in (OUT / "r/substitution").glob("*.json") if "_d" not in p.stem)),
        failed_replays=sorted(failed))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log("completed" if all(summary["jobs"].values()) and not failed else "completed_with_failures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.preflight:
        sys.exit(0 if preflight() else 1)
    run()
