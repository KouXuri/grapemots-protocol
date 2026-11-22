#!/usr/bin/env python3
"""Rebuild every frozen result of the Precision Agriculture article from the archived per-frame outputs and
compare it with the frozen file; then check every number printed in the article (tools/claims_pa_1121.py).

Run inside a workspace prepared by setup_workspace.sh (needs Python 3.10+ with SciPy and NumPy; no GPU,
imagery or model weights):
    bash journal2026_pa/setup_workspace.sh /tmp/ws
    python3 journal2026_pa/smoke_test.py /tmp/ws
Not rebuilt here: identity_metrics.json (TrackEval on the annotation files, which GrapeMOTS distributes) and
the figures (tools/make_figures_pa119_1121.py draws them from results/ alone).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

W = Path(sys.argv[1]).resolve()
T = W / "cadence_journal_1119/tools"
RES = W / "cadence_journal_1119/results"
FROZEN = W / "frozen_results"
if not FROZEN.exists():
    shutil.copytree(RES, FROZEN)
PY = sys.executable
STEPS = [  # (command, frozen outputs it rewrites)
    (["journal_numbers.py", "--check"], []),
    (["story_checks_1120.py"], ["story_checks_1120.json"]),
    (["reentry_analysis_1120.py"], ["reentry_analysis_1120.json"]),
    (["residual_motion_1120.py"], ["residual_motion_1120.json"]),
    (["oracle_trackers_summary_1120.py"], ["oracle_trackers_summary_1120.json"]),
    (["crop_vs_whole.py", "runs/fullframe_pponly_0922", "grapemots-protocol/cadence2026_0919/raw/retrain.tar.gz",
      "cadence_journal_1119/results/crop_vs_whole.json"], ["crop_vs_whole.json"]),
    (["error_sources_summary.py", "--sets", "whole_s0"], ["error_sources_summary.json"]),
    (["error_sources_summary.py", "--sigma", "3072"], ["error_sources_summary_s3072.json"]),
    (["calibration_transfer_1120.py"], ["calibration_transfer_1120.json"]),
    (["calibration_robustness_1121.py"], ["calibration_robustness_1121.json"]),
    (["calibration_under_fixes_1120.py", "--set", "whole_s0"], ["calibration_under_fixes_whole_s0.json"]),
    *[(["calibration_under_fixes_1120.py", "--set", s, "--sigma", "3072"], [f"calibration_under_fixes_{s}_s3072.json"])
      for s in ("crop_s0", "crop_s1", "crop_s2", "whole_s0", "whole_s1", "whole_s2")],
    (["real_trackers_vine_1121.py"], ["real_trackers_vine_1121.json"]),
    (["gate_buffer_summary_1121.py"], ["gate_buffer_summary_1121.json"]),
    (["supplementary_checks_1121.py"], ["supplementary_checks_1121.json"]),
    (["claims_pa_1121.py"], ["claims_pa_1121.csv"]),
]
bad = []
for cmd, outs in STEPS:
    p = subprocess.run([PY, str(T / cmd[0]), *cmd[1:]], cwd=W, capture_output=True, text=True)
    ok = p.returncode == 0
    for o in outs:
        new, old = RES / o, FROZEN / o
        if o.endswith(".json"):
            same = json.loads(new.read_text()) == json.loads(old.read_text())
        else:
            same = new.read_text() == old.read_text()
        ok = ok and same
        if not same:
            bad.append(f"{o} differs")
    print(("ok   " if ok else "FAIL ") + " ".join(cmd), flush=True)
    if p.returncode != 0:
        bad.append(f"{' '.join(cmd)} exited {p.returncode}: {p.stderr[-400:]}")
print("all frozen results reproduced and all claims pass" if not bad else "\n".join(bad))
sys.exit(1 if bad else 0)
