#!/usr/bin/env python3
"""Check this release from a fresh checkout, without imagery, weights or a GPU.

Three things are verified, in the order a reader would want them:

1. every file listed in SHA256SUMS is present and hashes to what it says;
2. the analyses that run off the frozen per-frame dumps still produce those
   frozen outputs, byte for byte, when re-run here;
3. the inputs that are *not* redistributed are named, so the boundary between
   "auditable from frozen outputs" and "rebuildable from released inputs" is
   visible rather than implied.

    python3 tools/smoke_test.py

Requires Python 3.9+ and numpy. The HOTA rows additionally need scipy and
trackeval, and their per-frame inputs for four of the eleven configurations are
detection caches this release does not carry; the test reports that rather than
failing on it.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("GRAPEMOTS_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"
SUMS = ROOT / "SHA256SUMS"

# analysis, output it regenerates
REBUILDS = [
    ("tools/aggregate_decomp.py", "cadence_decomposition.json"),
    ("tools/aggregate_decomp_all28.py", "cadence_decomposition_all28.json"),
]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def check_sums() -> int:
    if not SUMS.is_file():
        print(f"FAIL  no SHA256SUMS at {SUMS}")
        return 1
    bad = missing = checked = 0
    lines = [line.strip() for line in SUMS.read_text().splitlines() if line.strip()]
    # the manifest was written from the repository root, so its paths carry this
    # directory's name; resolve against whichever base makes the first entry exist
    base = ROOT
    if lines:
        first = lines[0].partition("  ")[2].strip()
        if not (ROOT / first).is_file() and (ROOT.parent / first).is_file():
            base = ROOT.parent
    for line in lines:
        if line.startswith("#"):
            continue
        expected, _, name = line.partition("  ")
        target = base / name.strip()
        if not target.is_file():
            print(f"FAIL  missing {name.strip()}")
            missing += 1
            continue
        checked += 1
        if digest(target) != expected.strip():
            print(f"FAIL  hash differs: {name.strip()}")
            bad += 1
    print(f"{'PASS' if not (bad or missing) else 'FAIL'}  "
          f"checksums: {checked} verified, {bad} differ, {missing} missing")
    return int(bool(bad or missing))


def check_rebuilds() -> int:
    failures = 0
    for script, output in REBUILDS:
        target = RESULTS / output
        if not target.is_file():
            print(f"SKIP  {output} is not in this checkout")
            continue
        before = digest(target)
        run = subprocess.run([sys.executable, str(ROOT / script)],
                             capture_output=True, text=True, cwd=ROOT)
        if run.returncode != 0:
            tail = (run.stderr or run.stdout).strip().splitlines()[-1:]
            print(f"FAIL  {script} exited {run.returncode}: {' '.join(tail)}")
            failures += 1
            continue
        after = digest(target)
        if before == after:
            print(f"PASS  {script} reproduces {output} byte for byte")
        else:
            print(f"FAIL  {script} changed {output}")
            failures += 1
    return failures


def report_boundary() -> None:
    print("\nWhat this checkout supports:")
    print("  auditable    every table and figure, against the frozen outputs above")
    print("  rebuildable  the cadence decomposition and its bootstrap, from the")
    print("               per-frame dumps in results/, by the scripts just run")
    print("  not carried  detection caches for four configuration rows, the")
    print("               corpora themselves and the detector checkpoints; the")
    print("               HOTA scripts name the file they cannot find")


def main() -> int:
    print(f"root: {ROOT}\n")
    failures = check_sums() + check_rebuilds()
    report_boundary()
    print("\n" + ("smoke test passed" if not failures else f"{failures} check(s) failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
