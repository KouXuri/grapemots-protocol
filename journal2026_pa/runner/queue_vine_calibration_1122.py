#!/usr/bin/env python3
"""Test vine-level calibration with one detector shared by the calibration and test vines."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from queue_fullframe_0922 import available_ram, link_dataset, train_args
from queue_pponly_controls_0922 import save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/pa_vine_calibration_1122"
DATA = ROOT / "datasets/grapemots_det_721"
SOURCE_MANIFEST = ROOT / "datasets/grapemots_fullframe_pponly_0922/manifest.json"
WEIGHTS = ROOT / "yolo26s.pt"
VINES = tuple(f"PathPlanning_{number}" for number in range(2, 9))
INTERVALS = (1, 8)
SCALE = 3072
GPU = "0"


def fold_plan():
    folds = []
    for index, test in enumerate(VINES):
        rotation = VINES[index:] + VINES[:index]
        folds.append(dict(test=test, calibration=list(rotation[1:3]),
                          validation=rotation[3], training=["PathPlanning_1", *rotation[4:]]))
    return folds


def prepare():
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    records = {vine: [] for vine in ("PathPlanning_1", *VINES)}
    for row in sorted(manifest["records"].values(), key=lambda row: row["basename"]):
        image, label = Path(row["image"]), Path(row["label"])
        if not image.is_file() or sha(label) != row["label_sha256"]:
            raise ValueError(f"Source image missing or label changed: {image}")
        records[row["video"]].append(row)
    if not all(records.values()):
        raise ValueError("The source manifest must cover all eight vines")

    folds = fold_plan()
    for fold in folds:
        roles = [*fold["training"], fold["validation"], *fold["calibration"], fold["test"]]
        if len(roles) != len(set(roles)) or set(roles) != set(records):
            raise ValueError(f"Overlapping or missing vine roles: {fold}")
        folder = OUT / "splits" / fold["test"]
        split = dict(train=[row for vine in fold["training"] for row in records[vine][::3]],
                     val=records[fold["validation"]])
        link_dataset(folder, split)
        recipe = train_args(folder / "data.yaml", "attempt_01", 0)
        recipe["project"] = str(OUT / "folds" / fold["test"] / "attempts")
        fold.update(training_args=recipe, train_frames=len(split["train"]),
                    validation_frames=len(split["val"]))

    plan = dict(folds=folds, source_manifest_sha256=sha(SOURCE_MANIFEST),
                initial_checkpoint_sha256=sha(WEIGHTS), seed=0, scale=SCALE,
                confidence=0.25, intervals=list(INTERVALS), phase=0,
                tracker="cfg/trackers/botsort_gmc.yaml", minimum_track_frames=1,
                target="Visible annotated bunch trajectories over all frames of the vine video")
    save(OUT / "plan.json", plan, immutable=True)
    print(json.dumps(dict(state="preflight_passed", folds=len(folds),
                          training_frames=[fold["train_frames"] for fold in folds])), flush=True)


def status(state, **details):
    row = dict(state=state, updated=time.strftime("%Y-%m-%d %H:%M:%S %Z"), **details)
    save(OUT / "status.json", row)
    print(json.dumps(row), flush=True)


def wait_for_gpu():
    while True:
        clients = subprocess.check_output(
            ["nvidia-smi", "-i", GPU, "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not clients and available_ram() >= 8 * 1024**3:
            return
        status("waiting_gpu", gpu=int(GPU))
        time.sleep(30)


def train(job_path):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires exactly one visible GPU")
    job = json.loads(job_path.read_text())
    batch = job["args"]["batch"]

    class FixedBatchTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train" and batch_size != batch:
                raise RuntimeError(f"Expected training batch {batch}, got {batch_size}")
            # The source experiment also uses two images per validation batch.
            return super().get_dataloader(dataset_path, batch, rank, mode)

    YOLO(str(WEIGHTS)).train(trainer=FixedBatchTrainer, **job["args"])
    folder = Path(job["args"]["project"]) / job["args"]["name"]
    checkpoint = folder / "weights/best.pt"
    with (folder / "results.csv").open() as stream:
        curve = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    metrics = [float(row["metrics/mAP50-95(B)"]) for row in curve]
    if not checkpoint.is_file() or not metrics or not all(map(math.isfinite, metrics)):
        raise RuntimeError(f"Training output incomplete: {folder}")
    save(job_path.parent / "trained.json",
         dict(weights=str(checkpoint), sha256=sha(checkpoint), epochs=len(metrics),
              validation_map50_95=max(metrics), args=job["args"]), immutable=True)


def get_checkpoint(fold, folder, env):
    marker = folder / "trained.json"
    if not marker.exists():
        # An interrupted attempt cannot be reused as a finished training run.
        attempts = folder / "attempts"
        attempts.mkdir(exist_ok=True)
        name = f"attempt_{len(list(attempts.glob('attempt_*'))) + 1:02d}"
        args = dict(fold["training_args"], name=name)
        job_path = folder / f"{name}.json"
        save(job_path, dict(args=args), immutable=True)
        wait_for_gpu()
        status("training", fold=fold["test"], gpu=int(GPU), attempt=name)
        with (folder / f"{name}.log").open("a") as stream:
            subprocess.run([sys.executable, __file__, "--train", str(job_path)],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    info = json.loads(marker.read_text())
    checkpoint = Path(info["weights"])
    if sha(checkpoint) != info["sha256"]:
        raise ValueError(f"Checkpoint changed: {checkpoint}")
    return checkpoint


def detection_dump(vine, weights, folder, env):
    dest = folder / f"{vine}_detections.json"
    if not dest.exists():
        temp = dest.with_suffix(".tmp")
        wait_for_gpu()
        status("detecting", fold=folder.name, vine=vine, gpu=int(GPU))
        with (folder / "inference.log").open("a") as stream:
            subprocess.run([sys.executable, "tools/dump_dets_0920.py", "--weights", str(weights),
                            "--root", str(DATA), "--video", vine, "--imgsz", str(SCALE),
                            "--conf", "0.25", "--out", str(temp)], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        temp.replace(dest)
    return dest


def evaluate(vine, weights, folder, env):
    import numpy as np
    import error_sources_1120 as replay

    dest = folder / f"{vine}_counts.json"
    if dest.exists():
        return json.loads(dest.read_text())
    dump, frames = replay.load_dump(detection_dump(vine, weights, folder, env))
    ground_truth = len({track for frame in frames for track in frame["gt_ids"]})
    if ground_truth == 0 or dump["video"] != vine or dump["weights"] != str(weights):
        raise ValueError(f"Unexpected detection dump for {vine}")
    hw = replay.size_of(vine)
    scores = {}
    for interval in INTERVALS:
        warps = np.load(replay.gmc_cache(vine, interval))
        record = replay.track(frames[::interval], warps, hw)
        record["video"] = vine
        result = replay.decompose(record, 0.5, 1)
        scores[str(interval)] = dict(P=result["P"], U=result["U"], D=result["D"],
                                     M=result["M"] + ground_truth - result["G"])
        if result["P"] - ground_truth != result["U"] + result["D"] - scores[str(interval)]["M"]:
            raise ValueError(f"Count decomposition failed for {vine} at interval {interval}")
    result = dict(vine=vine, frames=len(frames), G=ground_truth, intervals=scores)
    save(dest, result, immutable=True)
    return result


def calibration(test, donors):
    donor_truth = sum(row["G"] for row in donors)
    estimates = dict(no_video_mean=donor_truth / len(donors))
    factors = {}
    for delta in map(str, INTERVALS):
        count = test["intervals"][delta]["P"]
        donor_count = sum(row["intervals"][delta]["P"] for row in donors)
        factor = donor_truth / donor_count if donor_count else None
        factors[delta] = factor
        estimates[f"raw_d{delta}"] = count
        estimates[f"calibrated_d{delta}"] = count * factor if factor is not None else None
    factor = factors["1"]
    estimates["transfer_d1_to_d8"] = test["intervals"]["8"]["P"] * factor if factor is not None else None
    errors = {key: value / test["G"] - 1 if value is not None else None
              for key, value in estimates.items()}
    return dict(factors=factors, estimates=estimates, signed_relative_errors=errors)


def run():
    plan = json.loads((OUT / "plan.json").read_text())
    if sha(SOURCE_MANIFEST) != plan["source_manifest_sha256"] or sha(WEIGHTS) != plan["initial_checkpoint_sha256"]:
        raise ValueError("A frozen input has changed since preflight")
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU,
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        try:
            for fold in plan["folds"]:
                folder = OUT / "folds" / fold["test"]
                folder.mkdir(parents=True, exist_ok=True)
                weights = get_checkpoint(fold, folder, env)
                donors = [evaluate(vine, weights, folder, env) for vine in fold["calibration"]]
                test = evaluate(fold["test"], weights, folder, env)
                row = dict(split=fold, checkpoint=str(weights), test=test, donors=donors,
                           **calibration(test, donors))
                save(folder / "result.json", row, immutable=True)
                rows.append(row)
                status("fold_complete", fold=fold["test"], completed=len(rows), total=len(plan["folds"]))
            metrics = {}
            for name in rows[0]["signed_relative_errors"]:
                errors = [r["signed_relative_errors"][name] for r in rows]
                absolute = [abs(e) for e in errors if e is not None]
                metrics[name] = dict(n=len(absolute), median_absolute_relative_error=
                                     statistics.median(absolute) if absolute else None)
            save(OUT / "summary.json", dict(plan=plan, vines=rows, metrics=metrics), immutable=True)
            status("completed", folds=len(rows))
        except Exception as exc:
            status("failed", error=str(exc), completed=len(rows))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--train", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.train:
        train(args.train)
    else:
        run()
