#!/usr/bin/env python3
"""Smoke test for cadence2026_1001: every result this directory adds is rebuilt from
archived inputs, with no imagery, weights or GPU, and compared with the frozen
file; then every manuscript number outside the tables is re-checked. Run from the
archive root:

    python3 cadence2026_1001/tools/smoke_test.py

SciPy is required. Without it cadence2026_0919/tools/decompose_count_error.py
falls back to greedy box matching instead of the maximum-total-IoU assignment the
manuscript defines; P, G and every count claim are unchanged, but U and D can move
by one track (on the 2021 source-rate arm, 346/87 instead of 347/86).
"""
import hashlib, json, subprocess, sys, tarfile, tempfile
from pathlib import Path

try:
    import scipy  # noqa: F401
except ImportError:
    sys.exit("SciPy is required: without it the decomposition falls back to greedy matching")

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "cadence2026_1001"
RAW = ROOT / "cadence2026_0919/raw"
DEC = ROOT / "cadence2026_0919/tools"
ok = True

def same(a, b):
    return json.dumps(json.loads(Path(a).read_text()), sort_keys=True) == \
           json.dumps(json.loads(Path(b).read_text()), sort_keys=True)

print("== SHA-256 ==")
lines = [l for l in (HERE / "SHA256SUMS").read_text().splitlines() if l.strip()]
mismatch = [rel for h, rel in (l.split("  ", 1) for l in lines)
            if hashlib.sha256((HERE / rel).read_bytes()).hexdigest() != h]
print(f"{len(lines)} files hashed, {len(mismatch)} mismatched")
ok &= not mismatch

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for family in ("apple_val0000", "lovo_surface", "bytetrack", "strongsort", "retrain", "heldout"):
        with tarfile.open(RAW / f"{family}.tar.gz") as t:
            t.extractall(tmp)
    # retrained detectors: one directory per seed; the three frontal folds keep their
    # published seed-0 checkpoint, so their seed 0 is the lovo_surface arms
    for seed in (0, 1, 2):
        d = tmp / f"retrain_s{seed}"; d.mkdir()
        for f in (tmp / "retrain").glob(f"*_s{seed}__*.json"):
            (d / f.name.split("__", 1)[1]).write_bytes(f.read_bytes())
        if seed == 0:
            for f in (tmp / "lovo_surface").glob("NoPathPlanning_*.json"):
                (d / f.name).write_bytes(f.read_bytes())
    (tmp / "wholeframe").mkdir()
    with tarfile.open(HERE / "raw/wholeframe_2021.tar.gz") as t:
        t.extractall(tmp / "wholeframe")
    apple = next(tmp.rglob("apple_s1280_d1.json")).parent
    gm = HERE / "tools/gm_matched.py"
    print("== rebuilt from frozen inputs ==")
    jobs = [
        ([gm, DEC, tmp / "lovo_surface", tmp / "gm.json"], "gm_matched.json", tmp / "gm.json"),
        ([gm, DEC, tmp / "bytetrack", tmp / "bt.json"], "gm_matched_bytetrack.json", tmp / "bt.json"),
        ([gm, DEC, tmp / "strongsort", tmp / "ss.json"], "gm_matched_strongsort.json", tmp / "ss.json"),
        *[([gm, DEC, tmp / f"retrain_s{s}", tmp / f"r{s}.json"], f"gm_matched_retrain_s{s}.json", tmp / f"r{s}.json")
          for s in (0, 1, 2)],
        ([HERE / "tools/heldout_readmode.py", DEC, tmp / "heldout", tmp / "h.json"], "heldout_readmode.json", tmp / "h.json"),
        ([HERE / "tools/align_sensitivity.py", tmp / "a.json"], "align_sensitivity.json", tmp / "a.json"),
        ([HERE / "tools/wholeframe_table.py", tmp / "wholeframe", tmp / "w.json"], "wholeframe_2021.json", tmp / "w.json"),
        ([HERE / "tools/review_checks.py", tmp / "lovo_surface", tmp / "rc.json"], "review_checks.json", tmp / "rc.json"),
        ([HERE / "tools/apple_matched.py", DEC, apple, tmp / "m.json"], "apple_matched.json", tmp / "m.json"),
        ([HERE / "tools/apple_geometry.py", DEC, apple, tmp / "g.json"], "geometry_applemot.json", tmp / "g.json"),
    ]
    try:                       # TrackEval 1.3.0 computes IDF1 and HOTA; skipped if absent
        import trackeval  # noqa: F401
        jobs.append(([HERE / "tools/conf_fill_identity.py", tmp / "cf.json"], "conf_fill_identity.json", tmp / "cf.json"))
    except ImportError:
        print("  conf_fill_identity.json        skipped (TrackEval not installed)")
    for cmd, frozen, built in jobs:
        subprocess.run([sys.executable, *map(str, cmd)], check=True, capture_output=True, cwd=ROOT)
        match = same(built, HERE / "results" / frozen)
        ok &= match
        print(f"  {frozen:30} {'identical' if match else 'DIFFERS'}")

print("== manuscript numbers outside the tables ==")
audit = subprocess.run([sys.executable, str(HERE / "tools/claims_audit.py")], capture_output=True, text=True, cwd=ROOT)
print("  " + audit.stdout.strip().splitlines()[-1])
ok &= audit.returncode == 0
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
