#!/usr/bin/env python3
"""Smoke test for cadence2026_1001: every result this directory adds is rebuilt from
archived inputs, with stock Python, no imagery, weights or GPU, and compared with
the frozen file. Run from the archive root:

    python3 cadence2026_1001/tools/smoke_test.py
"""
import hashlib, json, subprocess, sys, tarfile, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "cadence2026_1001"
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
    with tarfile.open(ROOT / "cadence2026_0919/raw/apple_val0000.tar.gz") as t:
        t.extractall(tmp)
    raw = next(tmp.rglob("apple_s1280_d1.json")).parent
    dec = ROOT / "cadence2026_0919/tools"
    print("== rebuilt from frozen inputs ==")
    jobs = [
        ([sys.executable, HERE / "tools/apple_matched.py", dec, raw, tmp / "m.json"], "apple_matched.json", tmp / "m.json"),
        ([sys.executable, HERE / "tools/apple_geometry.py", dec, raw, tmp / "g.json"], "geometry_applemot.json", tmp / "g.json"),
        ([sys.executable, HERE / "tools/align_sensitivity.py", tmp / "a.json"], "align_sensitivity.json", tmp / "a.json"),
    ]
    for cmd, frozen, built in jobs:
        subprocess.run([str(c) for c in cmd], check=True, capture_output=True, cwd=ROOT)
        match = same(built, HERE / "results" / frozen)
        ok &= match
        print(f"  {frozen:28} {'identical' if match else 'DIFFERS'}")
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
