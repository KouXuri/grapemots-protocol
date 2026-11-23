#!/usr/bin/env python3
"""Plant-disjoint controls for the EXISTING crop-trained seed-0 checkpoints.

These are not full-frame-trained models. They form a labelled sensitivity suite
if a future main detector is trained on full frames. No training, checkpoint
selection or test-driven parameter search is performed by this queue.

Reuses track_grapemots_mot.py, the existing GPU worker, and the existing idle
gate. The currently scheduled disease queues finish first. All inputs are
fingerprinted before submission and checked again before GPU work begins.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs/ctl_0922_pponly"
VIDEOS = tuple(f"PathPlanning_{i}" for i in range(2, 9))
SCALES = (1536, 2048, 2560, 3072, 3840)
DELTAS = (1, 2, 4, 8)
BUFFERS = {2: 15, 4: 8, 8: 4}
UPSTREAM = (
    ("reports/research_direction_20260922", "correction_queue"),
    ("reports/research_direction_20260922", "deim_adaptation"),
    ("reports/field_replay_adaptation_20260922", "new_queue"),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value, immutable=False):
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if immutable and path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Refusing to replace a different frozen file: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def make_job(root, family, video, scale, delta, phase=0):
    q = root / "runs/ctl_0922_pponly"
    tracker = "cfg/trackers/botsort_gmc.yaml"
    if family in {"ret", "rt"}:
        tracker = f"cfg/trackers/botsort_gmc_buf{BUFFERS[delta]}.yaml"
    elif family == "bytetrack":
        tracker = "ultralytics/cfg/trackers/bytetrack.yaml"
    weight = root / f"runs/detect/rev_lovo_0920/{video}_s0/weights/best.pt"
    name = f"{video}_s{scale}_d{delta}"
    if family == "phase":
        name += f"_o{phase}"
    output = q / family / (name + ".json")
    expected = dict(weights=str(weight), root=str(root / "datasets/grapemots_det_721"),
                    split="all", tracker=tracker, detector_mode="resize", imgsz=scale,
                    global_imgsz=scale, conf=0.25, match_iou=0.5, merge="iou",
                    merge_fusion="nms", merge_threshold=0.5, frame_step=delta,
                    frame_offset=phase, catchup=family in {"cu", "rt"})
    command = [str(root / ".venv/bin/python"), "tools/track_grapemots_mot.py",
               "--weights", str(weight), "--root", expected["root"], "--split", "all",
               "--videos", video, "--tracker", tracker, "--detector-mode", "resize",
               "--imgsz", str(scale), "--global-imgsz", str(scale), "--conf", "0.25",
               "--match-iou", "0.5", "--merge", "iou", "--merge-fusion", "nms",
               "--merge-threshold", "0.5", "--frame-step", str(delta),
               "--frame-offset", str(phase), "--save-frame-boxes", "--save-frame-tracks",
               "--out", str(output) + ".tmp"]
    if expected["catchup"]:
        command.append("--catchup")
    return dict(id=f"{family}_{name}", family=family, video=video, expected=expected,
                output=str(output), command=command)


def jobs(root):
    out = []
    for family in ("ret", "cu", "rt", "phase", "bytetrack"):
        for video in VIDEOS:
            for scale in SCALES:
                for delta in (DELTAS if family == "bytetrack" else DELTAS[1:]):
                    for phase in (range(1, delta) if family == "phase" else (0,)):
                        out.append(make_job(root, family, video, scale, delta, phase))
    assert len(out) == 840 and len({j["id"] for j in out}) == 840
    return out


def validate_output(payload, job):
    for key, value in job["expected"].items():
        if payload["config"].get(key) != value:
            raise ValueError(f"{job['id']}: wrong {key}")
    videos = payload["videos"]
    if len(videos) != 1 or videos[0]["video"] != job["video"]:
        raise ValueError(f"{job['id']}: wrong test video")
    v = videos[0]
    columns = [v[k] for k in ("frame_names", "frame_predicted_ids", "frame_gt_ids",
                              "frame_predicted_boxes", "frame_gt_boxes")]
    if not columns[0] or len({len(x) for x in columns}) != 1:
        raise ValueError(f"{job['id']}: missing or unequal frame dumps")
    for pi, pb, gi, gb in zip(columns[1], columns[3], columns[2], columns[4]):
        if len(pi) != len(pb) or len(gi) != len(gb):
            raise ValueError(f"{job['id']}: box/identity length mismatch")


def check_upstream(plan, ledger):
    if any(row["returncode"] != 0 for row in ledger):
        raise RuntimeError("An upstream queue failed; inspect it before proceeding")
    names = [j["name"] for j in plan["jobs"]]
    if [j["name"] for j in ledger] != names[:len(ledger)]:
        raise RuntimeError("Upstream ledger does not match its frozen plan")
    return len(ledger) == len(names)


def prepare():
    import yaml
    QUEUE.mkdir(parents=True, exist_ok=True)
    queue_jobs = jobs(ROOT)
    fingerprints = {}
    def record(path):
        fingerprints[str(path)] = sha(path)
    for rel in ("tools/track_grapemots_mot.py", "tools/run_rev0920_worker3.sh",
                "tools/queue_pponly_controls_0922.py", "tools/run_algorithm_v7_queue_v12.py",
                "ultralytics/trackers/byte_tracker.py", "ultralytics/trackers/bot_sort.py",
                "ultralytics/cfg/trackers/bytetrack.yaml"):
        record(ROOT / rel)
    base_cfg = yaml.safe_load((ROOT / "cfg/trackers/botsort_gmc.yaml").read_text())
    for buf in (None, 15, 8, 4):
        p = ROOT / ("cfg/trackers/botsort_gmc.yaml" if buf is None
                    else f"cfg/trackers/botsort_gmc_buf{buf}.yaml")
        cfg = yaml.safe_load(p.read_text())
        assert cfg == dict(base_cfg, track_buffer=30 if buf is None else buf)
        record(p)
    curve_rows = []
    for video in VIDEOS:
        for seed in (0, 1, 2):
            data = ROOT / f"datasets/rev_lovo_0920/{video}_s{seed}/split_spec.json"
            spec = json.loads(data.read_text())
            tr, va, te = (set(spec[k]) for k in ("train", "val", "test"))
            assert len(tr) == 6 and len(va) == 1 and te == {video}
            assert not (tr & va or tr & te or va & te)
            assert all(v.startswith("PathPlanning_") for v in tr | va | te)
            folder = ROOT / f"runs/detect/rev_lovo_0920/{video}_s{seed}"
            args = yaml.safe_load((folder / "args.yaml").read_text())
            assert args["model"] == "yolo26s.pt" and not args["resume"]
            assert args["seed"] == seed and args["imgsz"] == 1280 and args["epochs"] == 15
            curves = [{k.strip(): float(v.strip()) for k, v in r.items()}
                      for r in csv.DictReader((folder / "results.csv").open())]
            assert len(curves) == 15
            best = max(curves, key=lambda x: x["metrics/mAP50-95(B)"])
            curve_rows.append(dict(video=video, seed=seed, best_epoch=int(best["epoch"]),
                                   best_ap=best["metrics/mAP50-95(B)"],
                                   final_ap=curves[-1]["metrics/mAP50-95(B)"], curve=curves))
            for p in (data, folder / "args.yaml", folder / "results.csv"):
                record(p)
            if seed == 0:
                record(folder / "weights/best.pt")
        for scale in SCALES:
            for delta in DELTAS:
                p = ROOT / f"runs/rev_0920/retrain/{video}_s0/{video}_s{scale}_d{delta}.json"
                reference_job = make_job(ROOT, "smoke", video, scale, delta)
                payload = json.loads(p.read_text())
                # Historical outputs used either absolute or relative forms of the same root.
                historical_root = Path(payload["config"]["root"])
                if not historical_root.is_absolute():
                    historical_root = ROOT / historical_root
                assert historical_root.resolve() == (ROOT / "datasets/grapemots_det_721").resolve()
                reference_job["expected"]["root"] = payload["config"]["root"]
                validate_output(payload, reference_job)
                record(p)
    for folder, prefix in UPSTREAM:
        p = ROOT / folder / (prefix + "_plan.json")
        record(p)
    for family in ("phase", "ret", "cu", "rt", "bytetrack", "smoke", "logs"):
        (QUEUE / family).mkdir(exist_ok=True)
    smoke = make_job(ROOT, "smoke", "PathPlanning_2", 1536, 8)
    plan = dict(protocol=__doc__, seed=0, training_representation="1280 crop-trained",
                jobs=queue_jobs, smoke=smoke, baseline_reused=140,
                counts=dict(Counter(j["family"] for j in queue_jobs)), sha256=fingerprints)
    save(QUEUE / "plan.json", plan, immutable=True)
    save(QUEUE / "training_curve_audit.json", dict(scope="crop validation, not whole-frame validation",
                                                 runs=curve_rows), immutable=True)
    commands = []
    for job in queue_jobs:
        out = job["output"]
        verify = shlex.join([str(ROOT / ".venv/bin/python"), "tools/queue_pponly_controls_0922.py",
                             "--verify-output", out, "--job-id", job["id"]])
        run = f"( [ -f {shlex.quote(out)} ] || ( {shlex.join(job['command'])} && mv {shlex.quote(out+'.tmp')} {shlex.quote(out)} ) )"
        commands.append(job["id"] + "\t" + run + " && " + verify)
    tsv = "\n".join(commands) + "\n"
    p = QUEUE / "jobs.tsv"
    if p.exists() and p.read_text() != tsv:
        raise RuntimeError("Existing job list differs")
    p.write_text(tsv)
    print(json.dumps(dict(status="preflight_passed", jobs=len(queue_jobs), counts=plan["counts"],
                          training_runs_audited=len(curve_rows), smoke_pending=True)), flush=True)


def run():
    plan = json.loads((QUEUE / "plan.json").read_text())
    lock = (QUEUE / "master.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(state, **kw):
        row = dict(state=state, updated=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
        save(QUEUE / "status.json", row)
        print(json.dumps(row), flush=True)
    try:
        for folder, prefix in UPSTREAM:
            p = ROOT / folder / (prefix + "_plan.json")
            upstream = json.loads(p.read_text())
            previous_count = None
            while True:
                lp = ROOT / folder / (prefix + "_ledger.json")
                ledger = json.loads(lp.read_text()) if lp.exists() else []
                if check_upstream(upstream, ledger):
                    break
                if len(ledger) != previous_count:
                    status("waiting_upstream", queue=prefix, completed=len(ledger), total=len(upstream["jobs"]))
                    previous_count = len(ledger)
                time.sleep(30)
        from run_algorithm_v7_queue_v12 import wait_for_free_gpus
        status("waiting_gpu_idle")
        wait_for_free_gpus()
        for p, h in plan["sha256"].items():
            if sha(p) != h:
                raise RuntimeError(f"Frozen input changed: {p}")
        status("smoke_test")
        smoke = plan["smoke"]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
                   REV_QUEUE_DIR=str(QUEUE), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with (QUEUE / "logs/smoke.log").open("a") as f:
            subprocess.run(smoke["command"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        tmp = Path(smoke["output"] + ".tmp")
        actual = json.loads(tmp.read_text())
        validate_output(actual, smoke)
        baseline = json.loads((ROOT / "runs/rev_0920/retrain/PathPlanning_2_s0/PathPlanning_2_s1536_d8.json").read_text())
        for key in ("frame_names", "frame_gt_ids", "frame_predicted_ids"):
            if actual["videos"][0][key] != baseline["videos"][0][key]:
                raise RuntimeError(f"Baseline reproduction failed: {key}")
        tmp.replace(smoke["output"])
        status("running", jobs=len(plan["jobs"]), baseline_reproduction="passed")
        processes, handles = [], []
        for card in (0, 1):
            f = (QUEUE / f"worker{card}.out").open("a")
            handles.append(f)
            processes.append(subprocess.Popen(["bash", "tools/run_rev0920_worker3.sh", str(card),
                                                str(QUEUE / "jobs.tsv")], cwd=ROOT, env=env,
                                               stdout=f, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        for f in handles:
            f.close()
        missing = [j["id"] for j in plan["jobs"] if not (QUEUE / "done" / j["id"]).exists()]
        if any(codes) or missing:
            raise RuntimeError(f"workers={codes}; {len(missing)} jobs incomplete")
        status("completed", jobs=len(plan["jobs"]))
    except Exception as e:
        status("failed", error=str(e))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--verify-output", type=Path)
    p.add_argument("--job-id")
    args = p.parse_args()
    if args.verify_output:
        plan = json.loads((QUEUE / "plan.json").read_text())
        job = next(j for j in plan["jobs"] if j["id"] == args.job_id)
        if args.verify_output != Path(job["output"]):
            raise ValueError("Wrong output path")
        validate_output(json.loads(args.verify_output.read_text()), job)
    elif args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        p.error("Choose --prepare, --run or --verify-output")


if __name__ == "__main__":
    main()
