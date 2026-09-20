#!/usr/bin/env python3
"""Sensitivity of the count terms to the mask-area cutoff that defines the reference.

The reference boxes were derived from GrapeMOTS's instance masks by
tools/create_grapemots_detection_dataset.py, dropping any instance whose mask
covers fewer than 20 pixels in a frame. This rebuilds the reference at cutoffs
1, 20, 100 and 400 pixels with the SAME code path (mask_to_boxes, the same 6-digit
normalised text round trip, load_gt_tracks-style parsing) and re-scores the frozen
predictions of the ten-sequence surface (runs/lovo_surface_0918). Predictions do
not depend on the reference, so no tracker is rerun.

Self-check: at cutoff 20 every arm's P, G, U, D, M must equal the frozen values.
"""
from __future__ import annotations
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, "tools")
from create_grapemots_detection_dataset import base_dir, read_labels  # noqa: E402
from decompose_count_error import decompose  # noqa: E402

SRC = Path("/home/kou/my_env/MOTS2024")
VIDEOS = [f"PathPlanning_{i}" for i in range(2, 9)] + [f"NoPathPlanning_{i}" for i in range(1, 4)]
SIGMAS = (1536, 2048, 2560, 3072, 3840); DELTAS = (1, 2, 4, 8)
CUTS = (1, 20, 100, 400)
OUT = Path("runs/rev_0920/cpu")


def instances(video):
    """key -> [(gid, area, xyxy parsed back exactly as load_gt_tracks would)]"""
    cache = OUT / f"masks_{video}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    inst_dir = base_dir(SRC, video) / "instances"
    names = read_labels(inst_dir)
    keep = {i + 1 for i, n in enumerate(names) if n == "grape"}
    res = {}
    for f in sorted(inst_dir.glob("*.png")):
        mask = np.array(Image.open(f)); h, w = mask.shape[:2]
        vals, inv = np.unique(mask, return_inverse=True)
        inv = inv.reshape(mask.shape)
        areas = np.bincount(inv.ravel())
        objs = ndimage.find_objects(inv + 1)
        rows = []
        for k, v in enumerate(vals):
            v = int(v)
            if v == 0 or v // 1000 not in keep or objs[k] is None:
                continue
            sy, sx = objs[k]
            x0, x1, y0, y1 = sx.start, sx.stop - 1, sy.start, sy.stop - 1
            if x1 <= x0 or y1 <= y0:
                continue
            # the builder's text round trip, then load_gt_tracks' parse
            xc, yc = float(f"{((x0 + x1) / 2) / w:.6f}"), float(f"{((y0 + y1) / 2) / h:.6f}")
            bw, bh = float(f"{(x1 - x0) / w:.6f}"), float(f"{(y1 - y0) / h:.6f}")
            xc, yc, bw, bh = xc * w, yc * h, bw * w, bh * h
            box = [float(np.float32(xc - bw / 2)), float(np.float32(yc - bh / 2)),
                   float(np.float32(xc + bw / 2)), float(np.float32(yc + bh / 2))]
            rows.append([(v // 1000) * 1000 + v % 1000, int(areas[k]), box])
        res[f"{video}__{f.stem}"] = rows
    cache.write_text(json.dumps(res))
    return res


def rescore(args):
    video, inst = args
    out = {}
    for c in CUTS:
        gt = {k: [(g, b) for g, a, b in rows if a >= c] for k, rows in inst.items()}
        gfull = len({g for rows in gt.values() for g, _ in rows})
        for s in SIGMAS:
            for d in DELTAS:
                v = json.load(open(f"runs/lovo_surface_0918/results/{video}_s{s}_d{d}.json"))["videos"][0]
                keys = [Path(n).stem for n in v["frame_names"]]
                e = dict(v)
                e["frame_gt_ids"] = [[g for g, _ in gt.get(k, [])] for k in keys]
                e["frame_gt_boxes"] = [[b for _, b in gt.get(k, [])] for k in keys]
                r = decompose(e, 0.5, 1)
                out[f"{c}|{s}|{d}"] = {k: r[k] for k in ("P", "G", "U", "D", "M")} | {"G_full": gfull}
    return video, out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as pool:
        insts = dict(zip(VIDEOS, pool.map(instances, VIDEOS)))
        results = dict(pool.map(rescore, [(v, insts[v]) for v in VIDEOS]))
    json.dump(results, open(OUT / "cutoff_sensitivity.json", "w"))
    # self-check at 20 px against the frozen decomposition
    bad = 0
    for v in VIDEOS:
        for s in SIGMAS:
            for d in DELTAS:
                fv = json.load(open(f"runs/lovo_surface_0918/results/{v}_s{s}_d{d}.json"))["videos"][0]
                ref = decompose(fv, 0.5, 1)
                got = results[v][f"20|{s}|{d}"]
                bad += any(ref[k] != got[k] for k in ("P", "G", "U", "D", "M"))
    print("self-check at 20 px: mismatching arms", bad, "of 200")
