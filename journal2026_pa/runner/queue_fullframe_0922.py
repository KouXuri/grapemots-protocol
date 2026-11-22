#!/usr/bin/env python3
"""Whole-frame, plant-disjoint replacement queue with explicit resource gates.

3072, rectangular batches, AMP, two images on one GPU; no image slicing and no
automatic batch/resolution reduction. A training/validation memory probe and a
3840 inference probe precede the first full training run. Every detector starts
from generic pretraining. All 21 training runs must pass their validation-budget
gate before the 420 primary conditions and 840 seed-0 controls are evaluated.
The old crop-trained control queue is superseded, not relabelled or erased.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from queue_pponly_controls_0922 import (
    VIDEOS, SCALES, DELTAS, make_job, jobs as control_jobs,
    validate_output, save, sha,
)

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs/fullframe_pponly_0922"
DATA = ROOT / "datasets/grapemots_fullframe_pponly_0922"
TRAIN = ROOT / "runs/detect/fullframe_pponly_0922"
PRIORITY = ROOT / "reports/disease_pair_patience20_20260922"
GIB = 1024 ** 3
MIN_START_RAM = 8 * GIB
MIN_RUNNING_RAM = 2 * GIB
EPOCHS = 60
PATIENCE = 20
IMGSZ = 3072
BATCH = 2


def priority_complete(plan, ledger, status):
    """Resource priority only: disease outputs are not counting-model inputs."""
    jobs = plan["jobs"]
    if [r["name"] for r in ledger] != [j["name"] for j in jobs[:len(ledger)]]:
        raise RuntimeError("Priority ledger does not match its frozen plan")
    for job, row in zip(jobs, ledger):
        if row["returncode"] != 0 and not job.get("optional"):
            raise RuntimeError("Priority disease experiment failed; inspect it before proceeding")
    state = status.get("state")
    if state in {"failed", "superseded"}:
        raise RuntimeError("Priority disease queue did not complete successfully")
    if state not in {"complete", "core_complete_export_failed"}:
        return False
    if len(ledger) != len(jobs):
        raise RuntimeError("Priority completion status has an incomplete ledger")
    if any("train" in job and len(row.get("checkpoint_sha256", "")) != 64
           for job, row in zip(jobs, ledger)):
        raise RuntimeError("Priority training checkpoint verification is missing")
    return True


def available_ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024


def train_args(data, name, seed, probe=False):
    return dict(data=str(data), project=str(TRAIN), name=name, exist_ok=False,
                imgsz=IMGSZ, rect=True, batch=BATCH, device="0", workers=0,
                cache=False, amp=True, epochs=1 if probe else EPOCHS,
                patience=PATIENCE, optimizer="AdamW", lr0=0.001, lrf=0.01,
                cos_lr=False, warmup_epochs=3.0, seed=seed, deterministic=True,
                mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, cutmix=0.0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5,
                scale=0.5, translate=0.1, multi_scale=0.0, nbs=64,
                val=True, save=True, save_period=-1, plots=False, resume=False)


def budget_gate(curve):
    key = "metrics/mAP50-95(B)"
    if not curve or any(not math.isfinite(float(r[key])) for r in curve):
        raise ValueError("Missing/non-finite validation curve")
    best = max(curve, key=lambda r: float(r[key]))
    epoch, last = int(float(best["epoch"])), int(float(curve[-1]["epoch"]))
    if float(best[key]) <= 0 or last - epoch < PATIENCE:
        raise ValueError(f"Budget unverified: best epoch {epoch}, last {last}; need {PATIENCE} epochs without improvement")
    return dict(best_epoch=epoch, final_epoch=last, best_ap=float(best[key]),
                rule="20 epochs without a strict improvement on this fold's whole-frame validation")


def remap_job(job, seed=0):
    j = json.loads(json.dumps(job))
    old_weight, old_out = j["expected"]["weights"], j["output"]
    weight = str(TRAIN / f"{j['video']}_s{seed}" / "weights/best.pt")
    out = str(QUEUE / j["family"] / Path(old_out).name)
    j["output"] = out
    j["expected"]["weights"] = weight
    j["command"] = [weight if x == old_weight else out + ".tmp" if x == old_out + ".tmp" else x
                     for x in j["command"]]
    return j


def inference_jobs():
    out = [remap_job(make_job(ROOT, f"surface_seed{seed}", v, s, d), seed)
           for seed in (0, 1, 2) for v in VIDEOS for s in SCALES for d in DELTAS]
    out += [remap_job(j) for j in control_jobs(ROOT)]
    assert len(out) == len({j["output"] for j in out}) == 1260
    return out


def write_text_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Refusing to overwrite {path}")
    else:
        path.write_text(text)


def link_dataset(folder, split_rows):
    for split, rows in split_rows.items():
        for kind in ("images", "labels"):
            (folder / kind / split).mkdir(parents=True, exist_ok=True)
        for row in rows:
            for kind, src in (("images", row["image"]), ("labels", row["label"])):
                dest = folder / kind / split / Path(src).name
                if dest.is_symlink():
                    assert dest.resolve() == Path(src).resolve()
                else:
                    assert not dest.exists()
                    dest.symlink_to(src)
        for kind in ("images", "labels"):
            assert {p.name for p in (folder / kind / split).iterdir()} == {
                Path(r["image" if kind == "images" else "label"]).name for r in rows}
    cfg = dict(path=str(folder), train="images/train", val="images/val", nc=1, names=["grape"])
    write_text_once(folder / "data.yaml", json.dumps(cfg, indent=2) + "\n")


def prepare():
    from PIL import Image
    QUEUE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "datasets/grapemots_det_721"
    labels = {}
    for split in ("train", "val", "test"):
        for p in (raw / "labels" / split).glob("PathPlanning_*.txt"):
            assert p.name not in labels
            labels[p.name] = p.resolve()
    records = {}
    by_video = {}
    for i in range(1, 9):
        video = f"PathPlanning_{i}"
        by_video[video] = []
        for image in sorted((raw / "images/all").glob(video + "__*")):
            label = labels[image.stem + ".txt"]
            track = raw / "tracks/all" / label.name
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
            expected = [[p[0], *p[2:]] for line in track.read_text().splitlines() if (p := line.split())]
            assert rows == expected, label
            assert all(len(r) == 5 and r[0] == "0" and all(0 <= float(x) <= 1 for x in r[1:]) for r in rows)
            with Image.open(image) as im:
                size = list(im.size)
                assert size in ([3840, 2160], [1920, 1080]) and im.format == "PNG"
            st = image.stat()
            record = dict(image=str(image.resolve()), label=str(label), video=video,
                          basename=image.name, size_wh=size, image_bytes=st.st_size,
                          image_mtime_ns=st.st_mtime_ns, label_sha256=sha(label), objects=len(rows))
            records[image.name] = record
            by_video[video].append(record)
    # Keep prefixed symlink names, not the unprefixed source-image basename.
    # image points at the existing prefixed all-view symlink; label names already match it.
    for name, row in records.items():
        row["image"] = str(raw / "images/all" / name)
    folds, train_jobs, fingerprints = {}, [], {}
    for video in VIDEOS:
        sp = ROOT / f"datasets/rev_lovo_0920/{video}_s0/split_spec.json"
        spec = json.loads(sp.read_text())
        tr, va, te = (set(spec[k]) for k in ("train", "val", "test"))
        assert len(tr) == 6 and len(va) == 1 and te == {video}
        assert not (tr & va or tr & te or va & te)
        assert all(x.startswith("PathPlanning_") for x in tr | va | te)
        selected = {"train": [r for v in sorted(tr) for r in by_video[v][::3]],
                    "val": [r for v in sorted(va) for r in by_video[v]]}
        folder = DATA / video
        link_dataset(folder, selected)
        folds[video] = dict(spec=spec, train=[r["basename"] for r in selected["train"]],
                            val=[r["basename"] for r in selected["val"]],
                            train_frames=len(selected["train"]), val_frames=len(selected["val"]))
        fingerprints[str(sp)] = sha(sp)
        fingerprints[str(folder / "data.yaml")] = sha(folder / "data.yaml")
    first_train = [records[n] for n in folds[VIDEOS[0]]["train"]]
    # Label-dense training images plus one image from each training video; no test/val labels are used to choose the probe.
    selected = {r["basename"]: r for r in sorted(first_train, key=lambda r: (-r["objects"], r["basename"]))[:16]}
    for v in folds[VIDEOS[0]]["spec"]["train"]:
        r = next(r for r in first_train if r["video"] == v)
        selected[r["basename"]] = r
    probe = list(selected.values())
    link_dataset(DATA / "resource_probe", dict(train=probe, val=probe[:4]))
    fingerprints[str(DATA / "resource_probe/data.yaml")] = sha(DATA / "resource_probe/data.yaml")
    for seed in (0, 1, 2):
        for video in VIDEOS:
            name = f"{video}_s{seed}"
            train_jobs.append(dict(name=name, video=video, seed=seed,
                                   args=train_args(DATA / video / "data.yaml", name, seed)))
    runtime = ["tools/queue_fullframe_0922.py", "tools/queue_pponly_controls_0922.py",
               "tools/track_grapemots_mot.py", "tools/run_rev0920_worker3.sh",
               "tools/run_algorithm_v7_queue_v12.py", "yolo26s.pt"]
    runtime += [str(p.relative_to(ROOT)) for p in (ROOT / "ultralytics").rglob("*.py")
                if not any(x.startswith(".") for x in p.relative_to(ROOT / "ultralytics").parts)]
    runtime += ["ultralytics/cfg/default.yaml", "ultralytics/cfg/trackers/bytetrack.yaml"]
    runtime += [f"cfg/trackers/{n}.yaml" for n in ("botsort_gmc", "botsort_gmc_buf15", "botsort_gmc_buf8", "botsort_gmc_buf4")]
    for rel in runtime:
        fingerprints[str(ROOT / rel)] = sha(ROOT / rel)
    # Direct tools/*.py execution imports the installed package, not necessarily
    # the repository copy used by an interactive Python session in the root.
    installed = Path(importlib.util.find_spec("ultralytics").origin).parent
    for p in list(installed.rglob("*.py")) + list((installed / "cfg").rglob("*.yaml")):
        fingerprints[str(p)] = sha(p)
    fingerprints[str(PRIORITY / "plan.json")] = sha(PRIORITY / "plan.json")
    js = inference_jobs()
    for name in {j["family"] for j in js} | {"logs"}:
        (QUEUE / name).mkdir(exist_ok=True)
    manifest = dict(records=records, folds=folds, probe=[r["basename"] for r in probe],
                    train_step=3, validation_step=1, pixel_representation="whole source frames, no tiles")
    save(DATA / "manifest.json", manifest, immutable=True)
    fingerprints[str(DATA / "manifest.json")] = sha(DATA / "manifest.json")
    plan = dict(protocol=__doc__, train_jobs=train_jobs, jobs=js, counts=dict(Counter(j["family"] for j in js)),
                ultralytics_package=str(installed), priority_dependency=str(PRIORITY),
                sha256=fingerprints, min_start_ram_bytes=MIN_START_RAM, min_running_ram_bytes=MIN_RUNNING_RAM,
                training=train_args(DATA / "resource_probe/data.yaml", "resource_probe", 0, True))
    save(QUEUE / "plan.json", plan, immutable=True)
    tsv = []
    for j in js:
        verify = shlex.join([str(ROOT / ".venv/bin/python"), "tools/queue_fullframe_0922.py",
                             "--verify-output", j["output"], "--job-id", j["id"]])
        out = shlex.quote(j["output"])
        run = f"( [ -f {out} ] || ( {shlex.join(j['command'])} && mv {shlex.quote(j['output']+'.tmp')} {out} ) )"
        tsv.append(j["id"] + "\t" + run + " && " + verify)
    write_text_once(QUEUE / "jobs.tsv", "\n".join(tsv) + "\n")
    verify_inputs(plan, manifest)
    print(json.dumps(dict(status="preflight_passed", training_runs=len(train_jobs),
                          inference_conditions=len(js), counts=plan["counts"],
                          fold_frames={v: (r["train_frames"], r["val_frames"]) for v, r in folds.items()},
                          gpu_probe="pending; not claimed to fit")), flush=True)


def verify_inputs(plan, manifest):
    for p, h in plan["sha256"].items():
        if sha(p) != h:
            raise RuntimeError(f"Frozen input changed: {p}")
    for r in manifest["records"].values():
        st = Path(r["image"]).stat()
        if (st.st_size, st.st_mtime_ns) != (r["image_bytes"], r["image_mtime_ns"]) or sha(r["label"]) != r["label_sha256"]:
            raise RuntimeError(f"Source image metadata or label changed: {r['basename']}")
    for video, fold in manifest["folds"].items():
        for split in ("train", "val"):
            folder = DATA / video
            if {p.name for p in (folder / "images" / split).iterdir()} != set(fold[split]):
                raise RuntimeError(f"Wrong {video} {split} image membership")
            for name in fold[split]:
                r = manifest["records"][name]
                assert (folder / "images" / split / name).resolve() == Path(r["image"]).resolve()
                assert (folder / "labels" / split / (Path(name).stem + ".txt")).resolve() == Path(r["label"]).resolve()


def train_one(name, probe=False):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    torch.set_num_threads(2)
    plan = json.loads((QUEUE / "plan.json").read_text())
    args = plan["training"] if probe else next(j["args"] for j in plan["train_jobs"] if j["name"] == name)
    assert not (TRAIN / name).exists(), "Do not overwrite an existing training run"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    total = torch.cuda.get_device_properties(0).total_memory
    assert total >= 15 * GIB
    telemetry = dict(train_batches=0, optimiser_steps=0, train_shapes=[],
                     peak_allocated=0, peak_reserved=0, min_available_ram=available_ram(),
                     device_total_bytes=total, args=args)

    class GuardedTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if self.args.batch != BATCH or (mode == "train" and batch_size != BATCH):
                raise RuntimeError("Automatic batch reduction is forbidden; resource check failed")
            return super().get_dataloader(dataset_path, BATCH, rank, mode)

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            shape = list(batch["img"].shape)
            if shape[-2:] != [1728, IMGSZ]:
                raise RuntimeError(f"Unexpected training shape {shape}; no square padding or silent resize accepted")
            if shape not in telemetry["train_shapes"]:
                telemetry["train_shapes"].append(shape)
            return batch

        def optimizer_step(self):
            super().optimizer_step()
            telemetry["optimiser_steps"] += 1

    def check_memory(_=None):
        telemetry["peak_allocated"] = max(telemetry["peak_allocated"], torch.cuda.max_memory_allocated())
        telemetry["peak_reserved"] = max(telemetry["peak_reserved"], torch.cuda.max_memory_reserved())
        free = available_ram()
        telemetry["min_available_ram"] = min(telemetry["min_available_ram"], free)
        if free < MIN_RUNNING_RAM or telemetry["peak_reserved"] > 0.9 * total:
            raise RuntimeError("Resource margin failed (need >=2 GiB RAM and >=10% unreserved GPU memory)")

    def batch_end(trainer):
        telemetry["train_batches"] += 1
        check_memory(trainer)

    def epoch_end(trainer):
        check_memory(trainer)
        save(QUEUE / f"{name}_resources.json", telemetry)

    model = YOLO(str(ROOT / "yolo26s.pt"))
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_val_batch_end", check_memory)
    model.add_callback("on_fit_epoch_end", epoch_end)
    try:
        model.train(trainer=GuardedTrainer, **args)
        check_memory()
        folder = TRAIN / name
        curves = [{k.strip(): v.strip() for k, v in row.items()}
                  for row in csv.DictReader((folder / "results.csv").open())]
        result = dict(status="passed", resources=telemetry, weights=str(folder / "weights/best.pt"),
                      weight_sha256=sha(folder / "weights/best.pt"))
        if not probe:
            result["budget"] = budget_gate(curves)
            # The CSV rounds metrics; also check the trainer's unrounded selection state.
            selected_epoch = int(model.trainer.stopper.best_epoch)
            if len(curves) - selected_epoch < PATIENCE:
                raise RuntimeError("The trainer still improved within the last 20 epochs; budget gate failed")
            result["budget"]["trainer_selected_epoch"] = selected_epoch
        else:
            assert len(curves) == 1 and telemetry["train_batches"] >= 8
        save(QUEUE / f"{name}_verified.json", result)
    finally:
        save(QUEUE / f"{name}_resources.json", telemetry)


def inference_probe():
    import torch
    import cv2
    from ultralytics import YOLO
    torch.set_num_threads(2)
    manifest = json.loads((DATA / "manifest.json").read_text())
    r = next(manifest["records"][n] for n in manifest["probe"] if manifest["records"][n]["size_wh"][0] == 3840)
    m = YOLO(str(TRAIN / "resource_probe/weights/best.pt"))
    im = cv2.imread(r["image"])
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        m.predict(im, imgsz=3840, conf=0.25, device="0", verbose=False, half=False)
    peak = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    assert peak <= 0.9 * total and available_ram() >= MIN_RUNNING_RAM
    save(QUEUE / "inference_3840_probe.json", dict(status="passed", peak_reserved_bytes=peak,
                                                 device_total_bytes=total, available_ram=available_ram()))


def monitored(command, log, env):
    with log.open("a") as f:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            if available_ram() < MIN_RUNNING_RAM:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("Stopped this queue's child to preserve 2 GiB host RAM; no other job was touched")
            time.sleep(1)
        if process.returncode:
            raise RuntimeError(f"Stage failed ({process.returncode}); inspect {log}")


def run():
    plan = json.loads((QUEUE / "plan.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert json.loads((QUEUE / "cpu_data_preflight.json").read_text())["status"] == "passed"
    lock = (QUEUE / "master.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(state, **kw):
        row = dict(state=state, updated=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
        save(QUEUE / "status.json", row)
        print(json.dumps(row), flush=True)
    try:
        dependency = Path(plan["priority_dependency"])
        if sha(dependency / "plan.json") != plan["sha256"][str(dependency / "plan.json")]:
            raise RuntimeError("Priority disease plan changed; review resource ordering")
        upstream = json.loads((dependency / "plan.json").read_text())
        last = None
        while True:
            p = dependency / "status.json"
            upstream_status = json.loads(p.read_text()) if p.exists() else {}
            p = dependency / "ledger.json"
            ledger = json.loads(p.read_text()) if p.exists() else []
            if priority_complete(upstream, ledger, upstream_status):
                break
            progress = (len(ledger), upstream_status.get("state"), upstream_status.get("job"))
            if progress != last:
                status("waiting_priority_disease", queue=dependency.name, completed=len(ledger),
                       total=len(upstream["jobs"]), upstream_state=progress[1], upstream_job=progress[2])
                last = progress
            time.sleep(1)
        from run_algorithm_v7_queue_v12 import wait_for_free_gpus
        status("waiting_resources")
        wait_for_free_gpus()
        while available_ram() < MIN_START_RAM:
            time.sleep(1)
        # The RAM wait may outlast an idle window: check both GPU facts again.
        from run_algorithm_v7_queue_v12 import gpu_compute_pids
        if gpu_compute_pids():
            wait_for_free_gpus()
        verify_inputs(plan, manifest)
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   REV_QUEUE_DIR=str(QUEUE), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        py = str(ROOT / ".venv/bin/python")
        script = "tools/queue_fullframe_0922.py"
        status("resource_probe", imgsz=IMGSZ, batch=BATCH, gpu=0)
        monitored([py, script, "--train-one", "resource_probe", "--probe"], QUEUE / "logs/resource_probe.log", env)
        monitored([py, script, "--inference-probe"], QUEUE / "logs/inference_probe.log", env)
        ledger = []
        for job in plan["train_jobs"]:
            status("whole_frame_training", run=job["name"], completed=len(ledger), total=len(plan["train_jobs"]))
            while available_ram() < MIN_START_RAM:
                time.sleep(1)
            verify_inputs(plan, manifest)
            monitored([py, script, "--train-one", job["name"]], QUEUE / "logs" / (job["name"] + ".log"), env)
            result = json.loads((QUEUE / (job["name"] + "_verified.json")).read_text())
            ledger.append(dict(name=job["name"], returncode=0, **result))
            save(QUEUE / "training_ledger.json", ledger)
        status("inference_and_controls", conditions=len(plan["jobs"]), training_runs=len(ledger))
        # A single worker bounds host RAM; no DDP or concurrent detector processes.
        monitored(["bash", "tools/run_rev0920_worker3.sh", "0", str(QUEUE / "jobs.tsv")],
                  QUEUE / "worker0.out", env)
        missing = [j["id"] for j in plan["jobs"] if not (QUEUE / "done" / j["id"]).exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} inference conditions incomplete; inspect fail/logs")
        status("completed", training_runs=21, inference_conditions=1260)
    except Exception as e:
        status("failed", error=str(e))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--train-one")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--inference-probe", action="store_true")
    p.add_argument("--verify-output", type=Path)
    p.add_argument("--job-id")
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.run:
        run()
    elif a.train_one:
        train_one(a.train_one, a.probe)
    elif a.inference_probe:
        inference_probe()
    elif a.verify_output:
        plan = json.loads((QUEUE / "plan.json").read_text())
        j = next(j for j in plan["jobs"] if j["id"] == a.job_id)
        assert a.verify_output == Path(j["output"])
        validate_output(json.loads(a.verify_output.read_text()), j)
    else:
        p.error("Choose a queue operation")


if __name__ == "__main__":
    main()
