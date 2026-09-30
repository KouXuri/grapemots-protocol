#!/usr/bin/env python3
"""Audit the released Bodegas files and recover source indices without clamping.

Read-only with respect to the original dataset. Official ZIP central directories
are fetched with bounded range requests, not by downloading the image archives.
Generated reports retain missing annotations and ambiguous mappings explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets/Bodegas Terras Gauda"
OUT = ROOT / "reports/bodegas_20260922"
RECORD = "https://zenodo.org/api/records/7330951"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def file_digests(path):
    digest = hashlib.sha256()
    md5 = hashlib.md5()
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block); md5.update(block); crc = zlib.crc32(block, crc)
    return dict(sha256=digest.hexdigest(), md5=md5.hexdigest(), crc32=crc, bytes=path.stat().st_size)


def parse_central(data):
    entries = []
    at = 0
    while at < len(data):
        values = struct.unpack_from("<4s6H3L5H2L", data, at)
        if values[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central-directory entry")
        name_size, extra_size, comment_size = values[10:13]
        name = data[at + 46:at + 46 + name_size].decode("utf-8")
        entries.append(dict(name=name, crc32=values[7], bytes=values[9]))
        at += 46 + name_size + extra_size + comment_size
    return entries


def release():
    import requests
    session = requests.Session()
    response = session.get(RECORD, timeout=45)
    response.raise_for_status()
    metadata = response.json()
    save(OUT / "zenodo_record.json", metadata)
    report = {}
    for entry in metadata["files"]:
        name = entry["key"]
        if not name.endswith(".zip"):
            continue
        url = f"https://zenodo.org/records/7330951/files/{name}?download=1"
        with session.get(url, headers={"Range": "bytes=-65536"}, stream=True, timeout=45) as response:
            if response.status_code != 206:
                raise RuntimeError(f"Range request refused for {name}; no full download attempted")
            tail = response.raw.read(65537)
            if len(tail) > 65536:
                raise RuntimeError("Unbounded range response")
            total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        end = tail.rfind(b"PK\x05\x06")
        eocd = struct.unpack_from("<4s4H2LH", tail, end)
        count, size, offset = eocd[4:7]
        begin = offset - (total - len(tail))
        if begin < 0 or begin + size > len(tail):
            raise RuntimeError("ZIP directory exceeds the bounded tail; needs explicit review")
        entries = parse_central(tail[begin:begin + size])
        assert len(entries) == count
        report[name] = dict(official_zip_md5=entry["checksum"], bytes=total, entries=entries)
        save(OUT / "release_inventory.json", report)
        print(name, len(entries), flush=True)
        time.sleep(0.1)


def raw_sequences():
    return sorted(p for p in RAW.iterdir() if p.is_dir() and (p / "instances").is_dir())


def audit():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    official = json.loads((OUT / "release_inventory.json").read_text())
    result = dict(sequences={}, files={}, errors=[])
    for seq in raw_sequences():
        remote = official[seq.name + ".zip"]["entries"]
        images = {p.name: p for p in (seq / "images").glob("*.png")}
        masks = {p.name: p for p in (seq / "instances").glob("*.png")}
        row = dict(images=len(images), masks=len(masks), unlabelled_images=sorted(images.keys() - masks.keys()),
                   masks_without_images=sorted(masks.keys() - images.keys()), paired=[], duplicates=[])
        remote_pngs = {}
        for entry in remote:
            parts = Path(entry["name"]).parts
            if len(parts) >= 2 and parts[-2] in {"images", "instances"} and parts[-1].endswith(".png"):
                remote_pngs[(parts[-2], parts[-1])] = entry
        local_keys = {("images", n) for n in images} | {("instances", n) for n in masks}
        row["official_missing_locally"] = sorted(str(k) for k in remote_pngs.keys() - local_keys)
        row["local_not_in_official"] = sorted(str(k) for k in local_keys - remote_pngs.keys())
        if row["official_missing_locally"] or row["local_not_in_official"]:
            result["errors"].append(seq.name + ": local/official membership differs")
        hashes = {}
        for kind, entries in (("images", images), ("instances", masks)):
            for name, path in sorted(entries.items()):
                digests = file_digests(path)
                result["files"][str(path.relative_to(RAW))] = digests
                expected = remote_pngs.get((kind, name), {})
                if (digests["bytes"], digests["crc32"]) != (expected.get("bytes"), expected.get("crc32")):
                    result["errors"].append(str(path.relative_to(RAW)) + ": CRC/size mismatch")
                if kind == "images":
                    hashes.setdefault(digests["sha256"], []).append(name)
        row["duplicates"] = [v for v in hashes.values() if len(v) > 1]
        for name in sorted(images.keys() & masks.keys()):
            mask = cv2.imread(str(masks[name]), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (2160, 4096) or mask.dtype != np.uint16:
                raise ValueError(f"Unexpected mask: {masks[name]}")
            objects = []
            for value in np.unique(mask):
                if value == 0:
                    continue
                if int(value) // 1000 != 1:
                    raise ValueError(f"Unexpected MOTS class {value} in {masks[name]}")
                ys, xs = np.nonzero(mask == value)
                objects.append(dict(id=int(value), area=int(len(xs)),
                                    box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
            row["paired"].append(dict(image=name, objects=objects))
        row["track_ids"] = sorted({o["id"] for f in row["paired"] for o in f["objects"]})
        row["flight_row"] = int(re.match(r"row_(\d+)", seq.name)[1])
        result["sequences"][seq.name] = row
        save(OUT / "raw_audit.json", result)
        print(seq.name, "images",len(images), "masks",len(masks), "duplicates",row["duplicates"], flush=True)
    result["totals"] = dict(sequences=len(result["sequences"]), images=sum(r["images"] for r in result["sequences"].values()),
                            masks=sum(r["masks"] for r in result["sequences"].values()),
                            sequence_scoped_tracks=sum(len(r["track_ids"]) for r in result["sequences"].values()),
                            instances=sum(len(f["objects"]) for r in result["sequences"].values() for f in r["paired"]))
    result["status"] = "passed" if not result["errors"] else "failed"
    save(OUT / "raw_audit.json", result)
    print(result["status"], result["totals"], flush=True)


def align():
    import cv2
    import numpy as np
    from align_annotated_to_source import thumbnail, match
    cv2.setNumThreads(1)
    audited = json.loads((OUT / "raw_audit.json").read_text())
    assert audited["status"] == "passed"
    metadata = json.loads((OUT / "zenodo_record.json").read_text())
    official = {x["key"]: x["checksum"] for x in metadata["files"]}
    for seq in raw_sequences():
        dest = OUT / "alignment" / (seq.name + ".json")
        if dest.exists():
            print("existing audit", seq.name, flush=True)
            continue
        video_name = seq.name.replace("row_", "Row") + ("_1" if seq.name == "row_6.3" else "") + ".mp4"
        video = RAW / video_name
        digests = file_digests(video)
        assert official[video_name] == "md5:" + digests["md5"], video_name
        files = sorted((seq / "images").glob("*.png"))
        annotated = np.asarray([thumbnail(cv2.imread(str(p)), 64, 36) for p in files], dtype=np.int16)
        cap = cv2.VideoCapture(str(video)); assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        source = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            source.append(thumbnail(frame, 64, 36))
        cap.release()
        indices, residuals, margins = match(annotated, np.asarray(source, dtype=np.int16))
        wanted = {int(i) for i in indices}
        checked = {}
        cap = cv2.VideoCapture(str(video))
        for i in range(max(wanted) + 1):
            ok, frame = cap.read(); assert ok
            if i not in wanted: continue
            for k in np.where(indices == i)[0]:
                reference = cv2.imread(str(files[k]))
                diff = np.abs(reference.astype(np.int16) - frame.astype(np.int16))
                checked[files[k].name] = dict(source_index=i, thumbnail_mae=float(residuals[k]),
                                              nonadjacent_margin=float(margins[k]), full_mae=float(diff.mean()),
                                              max_abs=int(diff.max()), exact=bool(not diff.any()),
                                              labelled=(seq / "instances" / files[k].name).exists())
        cap.release()
        raw_indices = [int(x) for x in indices]
        result = dict(video=video_name, video_digests=digests, source_frames=len(source), fps=fps,
                      mappings=checked, strictly_increasing=all(b > a for a, b in zip(raw_indices, raw_indices[1:])),
                      policy="Independent appearance match, sequential full-resolution verification; never clamped",
                      exact_matches=sum(r["exact"] for r in checked.values()), total=len(files))
        save(dest, result)
        print(seq.name, "exact",result["exact_matches"], "/",len(files), "monotone",result["strictly_increasing"],flush=True)


def summarise():
    import cv2
    import numpy as np
    raw = json.loads((OUT / "raw_audit.json").read_text())
    reports = {p.stem: json.loads(p.read_text()) for p in (OUT / "alignment").glob("*.json")}
    if set(reports) != set(raw["sequences"]):
        raise RuntimeError("The source alignment audit is incomplete")
    all_frames = [(seq, name, row) for seq, report in reports.items() for name, row in report["mappings"].items()]
    maes = [r["full_mae"] for _, _, r in all_frames]
    nonmonotone = sorted(s for s, r in reports.items() if not r["strictly_increasing"])
    # Thresholds are descriptive diagnostics, not a rule accepting or deleting
    # test examples. The source-rate suite remains blocked for explicit review.
    diagnostics = {str(t): sum(x > t for x in maes) for t in (5, 10, 20)}
    report = dict(raw_file_status=raw["status"], totals=raw["totals"],
        local_files_match_official_crc_and_size=not raw["errors"],
        missing_mask_images={s: r["unlabelled_images"] for s, r in raw["sequences"].items() if r["unlabelled_images"]},
        flight_row_groups=dict(Counter(str(r["flight_row"]) for r in raw["sequences"].values())),
        source_alignment=dict(status="not_accepted_for_source_rate_experiments", audited_sequences=len(reports),
            audited_images=len(all_frames), exact_rgb_matches=sum(r["exact"] for _, _, r in all_frames),
            full_rgb_mae_quantiles={str(q): float(np.quantile(maes, q)) for q in (0, .5, .9, .95, 1)},
            full_rgb_mae_above_diagnostic_levels=diagnostics,
            nonmonotone_sequences=nonmonotone,
            large_residual_frames=[dict(sequence=s, image=n, **r) for s, n, r in all_frames if r["full_mae"] > 5],
            decoder=dict(opencv=cv2.__version__, platform="local macOS", colour_differences_not_resolved=True),
            video_md5_matches_official=True,
            note="No frame indices were clamped. Small RGB residuals may be decoder-dependent; large residuals and non-injective mappings still require diagnosis."),
        primary_annotation_replay=dict(status="prepared_not_submitted", usable_labelled_frames=664,
            target_training=False, missing_masks_are_not_negative_examples=True,
            source_mapping_required=False, interval_unit="available labelled-frame index"))
    save(OUT / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_alignment"}, indent=2))
    print(json.dumps({k: v for k, v in report["source_alignment"].items() if k != "large_residual_frames"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("release", "audit", "align", "summarise"))
    args = parser.parse_args()
    {"release": release, "audit": audit, "align": align, "summarise": summarise}[args.mode]()


if __name__ == "__main__":
    main()
