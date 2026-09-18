#!/usr/bin/env python3
"""Freeze the camera-ready evidence into release stubs (grapemots-protocol r24).

Every per-arm JSON written by tools/track_grapemots_mot.py is cut to what the
paper's numbers need -- config, overall metrics, and per video the frame ids and
boxes the decomposition reads, plus timing -- and gzipped, values untouched (no
rounding, so U/D/M recompute bit for bit). One tarball per family.

    families                       source
    lovo_surface   (200 arms)      runs/lovo_surface_0918/results
    phase          (550)           runs/ctl_0919/phase
    ret, cu, rt    (150 each)      runs/ctl_0919/{ret,cu,rt}
    timing         (72)            runs/ctl_0919/timing
    apple          (8 + 5)         runs/caout_stage13_20260829/pa/apple_* + runs/ctl_0919/apple
    heldout        (23)            the files listed in tools/surface_cost_audit_0919.py

Writes /tmp/r24/raw/<family>.tar.gz and /tmp/r24/raw/MANIFEST.json (per stub:
source path, source sha256, stub sha256).
"""
from __future__ import annotations
import glob, gzip, hashlib, io, json, os, tarfile

OUT = "/tmp/r24/raw"
KEEP = ("video", "frames", "annotated_frames", "effective_fps", "mean_ms_per_frame",
        "frame_ms", "frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids",
        "frame_gt_boxes")
B = "runs/caout_cadence_budget_20260819/results"
HELDOUT = sorted(glob.glob("runs/grid_0906/results/grid_*_s*.json")) + [
    f"{B}/full1536_s1.json", f"{B}/full2048_s1.json", f"{B}/full2048_s2.json",
    f"{B}/full2560_s1.json", "runs/caout_stage2_20260825/predicted/pred_full3072_s4.json",
    f"{B}/tiled_s1.json", f"{B}/tiled_s4.json", f"{B}/tiled_s8.json"]
FAMILIES = {
    "lovo_surface": sorted(glob.glob("runs/lovo_surface_0918/results/*_s*_d*.json")),
    "phase": sorted(glob.glob("runs/ctl_0919/phase/*.json")),
    "ret": sorted(glob.glob("runs/ctl_0919/ret/*.json")),
    "cu": sorted(glob.glob("runs/ctl_0919/cu/*.json")),
    "rt": sorted(glob.glob("runs/ctl_0919/rt/*.json")),
    "timing": sorted(glob.glob("runs/ctl_0919/timing/*_d*_r*.json")),
    "apple": sorted(glob.glob("runs/caout_stage13_20260829/pa/apple_*.json"))
             + sorted(glob.glob("runs/ctl_0919/apple/*.json")),
    "heldout": HELDOUT,
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for fam, paths in FAMILIES.items():
        tpath = f"{OUT}/{fam}.tar.gz"
        with tarfile.open(tpath, "w:gz", compresslevel=9) as tar:
            for p in paths:
                raw = open(p, "rb").read()
                j = json.loads(raw)
                stub = {"source": p, "config": j["config"], "overall": j.get("overall"),
                        "videos": [{k: v[k] for k in KEEP if k in v} for v in j["videos"]]}
                data = json.dumps(stub, separators=(",", ":")).encode()
                name = f"{fam}/{os.path.basename(p)}"
                info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
                manifest[name] = {"source": p, "source_sha256": sha(raw), "stub_sha256": sha(data)}
        print(f"{fam:13s} {len(paths):4d} arms  {os.path.getsize(tpath)/1e6:6.1f} MB")
    json.dump(manifest, open(f"{OUT}/MANIFEST.json", "w"), indent=1)


if __name__ == "__main__":
    main()
