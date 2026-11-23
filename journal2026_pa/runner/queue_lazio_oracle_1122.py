#!/usr/bin/env python3
"""Replay the GrapeMOTS reference-box protocol on the four AgriSORT grape sequences."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
from pathlib import Path
import re
import time

from queue_pponly_controls_0922 import save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/pa_lazio_1122"
SOURCE = OUT / "source"
DRIVE_FOLDER = "1sbsUwCtGQtA58cLVP5sd37tjjkMFCh05"
# Table I of the AgriSORT paper gives these frame, track and box counts.
SEQUENCES = {
    "CloseUp1": (300, 23, 2583, 30),
    "CloseUp2": (300, 22, 3581, 30),
    "Overview1": (100, 20, 1040, 10),
    "Overview2": (100, 31, 721, 10),
}
TRACKERS = ("botsort_gmc", "bytetrack", "strongsort", "ocsort", "deepocsort")
INTERVALS = (1, 2, 4, 8)


def status(state, **details):
    row = dict(state=state, updated=time.strftime("%Y-%m-%d %H:%M:%S %Z"), **details)
    save(OUT / "status.json", row)
    print(json.dumps(row), flush=True)


def download_inputs():
    import gdown
    from gdown.exceptions import FileURLRetrievalError
    from requests import RequestException

    files = gdown.download_folder(id=DRIVE_FOLDER, skip_download=True, quiet=True, use_cookies=False)
    paths = [Path(file.path) for file in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate dataset paths in the source listing")
    expected = set()
    for video, (frames, *_rest) in SEQUENCES.items():
        prefix = "frame" if video.startswith("Overview") else ""
        expected.add(Path(video) / "gt/gt.txt")
        expected.update(Path(video) / "Images" / f"{prefix}{index:04}.png"
                        for index in range(1, frames + 1))
    if set(paths) != expected:
        raise ValueError("Published dataset listing no longer matches the four benchmark sequences")
    missing = [file for file in files if not (SOURCE / file.path).is_file()]
    status("downloading", total=len(files), missing=len(missing))

    def download(file):
        destination = SOURCE / file.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".download")
        last_error = None
        for attempt in range(6):
            try:
                result = gdown.download(id=file.id, output=str(partial), quiet=True,
                                        use_cookies=False, resume=True)
                if result and partial.stat().st_size:
                    partial.replace(destination)
                    return
            except (FileURLRetrievalError, RequestException) as exc:
                last_error = exc
            # gdown resolves through drive.google.com/uc, which Google throttles after a few hundred
            # files; the same file id on drive.usercontent.google.com still answers (as in
            # tools/fetch_lettucemots2.py), so fall back to it before backing off.
            try:
                import requests
                with requests.get("https://drive.usercontent.google.com/download",
                                  params={"id": file.id, "export": "download", "confirm": "t"},
                                  stream=True, timeout=90) as r:
                    if r.status_code == 200 and "text/html" not in r.headers.get("content-type", ""):
                        with partial.open("wb") as fh:
                            for block in r.iter_content(1 << 16):
                                fh.write(block)
                        if partial.stat().st_size:
                            partial.replace(destination)
                            return
            except RequestException as exc:
                last_error = exc
            time.sleep(min(60, 4 * 2**attempt))
        raise RuntimeError(f"Could not download {file.path}") from last_error

    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, _ in enumerate(pool.map(download, missing), 1):
            if index % 50 == 0 or index == len(missing):
                status("downloading", total=len(files), missing=len(missing) - index)


def reference_frames(video, images, annotations, width, height):
    with annotations.open() as stream:
        rows = list(csv.reader(stream))
    expected_n, expected_ids, expected_boxes, _ = SEQUENCES[video]
    if len(images) != expected_n or len(rows) != expected_boxes or len({r[1] for r in rows}) != expected_ids:
        raise ValueError(f"Source counts disagree with AgriSORT Table I for {video}")
    boxes = defaultdict(dict)
    invisible = clipped = 0
    for row in rows:
        frame, identity = map(int, row[:2])
        x, y, w, h = map(float, row[2:6])
        if len(row) != 9 or not (1 <= frame <= expected_n) or row[7] != "1":
            raise ValueError(f"Unexpected MOT annotation for {video}: {row}")
        if float(row[6]) <= 0 or float(row[8]) <= 0:
            invisible += 1
            continue
        original = [x, y, x + w, y + h]
        box = [max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h)]
        if not (box[2] > box[0] and box[3] > box[1]) or identity in boxes[frame]:
            raise ValueError(f"Invalid or duplicate reference box for {video}: {row}")
        clipped += box != original
        boxes[frame][identity] = box
    frames = []
    for index, image in enumerate(images, 1):
        if int(re.search(r"\d+$", image.stem).group()) != index:
            raise ValueError(f"Missing or misordered source frame: {image}")
        references = boxes[index]
        frames.append(dict(name=image.name, path=str(image), gt_ids=list(references),
                           gt_boxes=list(references.values()), boxes=list(references.values()),
                           scores=[0.9] * len(references)))
    ground_truth = len({t for frame in frames for t in frame["gt_ids"]})
    return frames, dict(frames=len(frames), G=ground_truth, invisible_rows=invisible, clipped_boxes=clipped)


def prepare_references():
    from PIL import Image

    reference = {}
    dumps = OUT / "lazio/dets"
    dumps.mkdir(parents=True, exist_ok=True)
    for video, (_, _, _, fps) in SEQUENCES.items():
        annotations = SOURCE / video / "gt/gt.txt"
        images = sorted((SOURCE / video / "Images").glob("*.png"))
        sizes = set()
        for image in images:
            with Image.open(image) as picture:
                sizes.add(picture.size)
                picture.verify()
        if len(sizes) != 1:
            raise ValueError(f"Expected one image size per sequence: {video} {sizes}")
        width, height = sizes.pop()
        frames, info = reference_frames(video, images, annotations, width, height)
        reference[video] = dict(info, fps=fps, size=[width, height], gt_sha256=sha(annotations))
        save(dumps / f"{video}_s0.json", dict(video=video, weights="reference boxes", imgsz=0,
                                             conf=0.9, size=[width, height], frames=frames), immutable=True)
    save(OUT / "reference.json", reference, immutable=True)
    return reference


def summarise(reference):
    from decompose_count_error import decompose
    from external_summary_1122 import classify

    result = []
    for video, info in reference.items():
        for tracker in TRACKERS:
            for interval in INTERVALS:
                suffix = f"_d{interval}" if tracker in TRACKERS[:2] else f"_s0_d{interval}"
                path = OUT / "lazio" / tracker / f"{video}{suffix}.json"
                record = json.loads(path.read_text())["videos"][0]
                record["video"] = video
                for minimum in (1, 3):
                    counts = decompose(record, 0.5, minimum)
                    truth = info["G"]
                    missed = counts["M"] + truth - counts["G"]
                    if counts["P"] - truth != counts["U"] + counts["D"] - missed:
                        raise ValueError(f"Error decomposition failed for {path}")
                    row = dict(video=video, tracker=tracker, interval=interval, min_track_frames=minimum,
                               processing_fps=info["fps"] / interval, G=truth,
                               visible_at_interval=counts["G"], P=counts["P"], U=counts["U"],
                               D=counts["D"], M=missed, e=(counts["P"] - truth) / truth)
                    if interval == minimum == 1:
                        row["duplicate_origins"] = dict(classify(record))
                    result.append(row)
    save(OUT / "summary.json", dict(reference=reference, results=result), immutable=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(OUT / "plan.json", dict(source=f"https://drive.google.com/drive/folders/{DRIVE_FOLDER}",
             source_paper="10.1109/ICRA57147.2024.10610231", sequences=list(SEQUENCES),
             trackers=list(TRACKERS), intervals=list(INTERVALS), primary_interval=1,
             minimum_track_frames=[1, 3], scores=0.9, match_iou=0.5, phase=0,
             reference="Positive-visibility labelled boxes; full-sequence trajectory count"),
             immutable=True)
        try:
            download_inputs()
            reference = prepare_references()
            status("tracking", sequences=len(reference), tracker_runs=len(TRACKERS) * len(INTERVALS) * len(reference))
            import oracle_external_1122 as oracle

            oracle.OUT = OUT
            oracle.DATASETS["lazio"] = (SOURCE, list(SEQUENCES))
            oracle.main(["lazio"])
            summarise(reference)
            status("completed", sequences=len(reference))
        except Exception as exc:
            status("failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
