#!/usr/bin/env python3
"""Checks for the 2026-10-01 camera-ready (cadence_1001/main.tex).

Runs tools/verify_cbdcom_paper.py's checks, with two deliberate differences --
Panel B of the configuration table is summarised in the text rather than
tabulated, and a multi-line \\shortstack header is not a table row -- then checks
everything this revision adds against the frozen files it was read from:
the AppleMOT table and sentences (runs/apple_matched_1001), the alignment row
(runs/align_sens_1001) and the Fig. 1 decomposition (runs/decomp_0812).

    python3 tools/verify_cbdcom_paper_1001.py cadence_1001/main.tex
"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cadence_1001/main.tex"
tex = PAPER.read_text()
bad = []

# 1. the accepted version's checks, minus the two known, intended differences
out = subprocess.run([sys.executable, str(ROOT / "tools/verify_cbdcom_paper.py"), str(PAPER)],
                     capture_output=True, text=True).stdout
for line in out.splitlines():
    if line.startswith("FAIL"):
        if "panel B row absent" in line:
            continue                                   # summarised in the text
        if re.search(r"row has 1 fields, spec has 6: (circling|frontal)", line):
            continue                                   # \shortstack header line
        bad.append(line[6:])
print(out.splitlines()[0])

FLAT = " ".join(tex.split())

def need(s, what):
    if " ".join(s.split()) not in FLAT:
        bad.append(f"{what}: '{s}' not in text")

# 2. AppleMOT table: every cell, dense/sparse, rises, coverage
A = json.loads((ROOT / "runs/apple_matched_1001/results/apple_matched.json").read_text())
for s in (640, 960, 1280):
    cells = []
    rises = 0
    for k in (2, 4, 8):
        r = A[f"s{s}_k{k}_six"]
        cells.append(f"${r['dense']['e']:+.3f}/{r['sparse']['e']:+.3f}$".replace("+", "") if False else
                     f"${r['dense']['e']:.3f}/{r['sparse']['e']:.3f}$".replace("-", "-"))
        rises += int(r["up/down/tie"].split("/")[0])
    row = f"{s}" + ("  " if s < 1000 else " ") + " & " + " & ".join(cells) + f" & {rises}/18 \\\\"
    need(row, f"AppleMOT row sigma={s}")
cov = " & ".join(f"${A[f's1280_k{k}_six']['dense']['coverage']:.3f}/{A[f's1280_k{k}_six']['sparse']['coverage']:.3f}$" for k in (2, 4, 8))
need(cov, "AppleMOT coverage row")
up6 = sum(int(A[f"s{s}_k{k}_six"]["up/down/tie"].split("/")[0]) for s in (640, 960, 1280) for k in (2, 4, 8))
up3 = sum(int(A[f"s{s}_k{k}_unique3"]["up/down/tie"].split("/")[0]) for s in (640, 960, 1280) for k in (2, 4, 8))
need(f"{up6} of\n54", "AppleMOT comparisons rising") if f"{up6} of\n54" in tex else need(f"{up6} of 54", "AppleMOT comparisons rising")
need(f"all {up3} on the", "AppleMOT unique3 rising")
u3 = A["s1280_k2_unique3"]["dense"]["e"]
need(f"${u3:+.3f}$", "AppleMOT unique3 dense k=2")
d1280 = [A[f"s1280_k{k}_six"] for k in (2, 4, 8)]
need(f"${d1280[0]['dense']['e']:+.3f}$ to ${d1280[2]['dense']['e']:+.3f}$", "AppleMOT dense range")
need(f"${d1280[0]['sparse']['e']:+.3f}$ to ${d1280[2]['sparse']['e']:+.3f}$", "AppleMOT sparse range")
if any(A[f"s{s}_k{k}_six"][arm]["e"] >= 0 for s in (640, 960, 1280) for k in (2, 4, 8) for arm in ("dense", "sparse")):
    bad.append("abstract says the pooled AppleMOT error stays negative, but a six-sequence cell is >= 0")

G = json.loads((ROOT / "runs/apple_matched_1001/results/geometry_applemot.json").read_text())
need(f"(U+D)/G={G['base_UD_over_G_s1280_k1']:.2f}$", "AppleMOT base surplus")
need(f"{G['by_step']['1']['sequence_median_r']:.2f} target sizes", "AppleMOT r at full rate")

# 3. the alignment row
S = json.loads((ROOT / "runs/align_sens_1001/results/align_sensitivity.json").read_text())
row = S["table_ii_rows"]["suspect frames dropped"]
d = row["delta_median"]
d3 = f"{int(d * 1000 + 0.5) / 1000:+.3f}"                         # half-up, as printed
need(f"Alignment out & ${row['sparse_median']:+.3f}$ & ${row['source_median']:+.3f}$ & ${d3}$ & "
     f"$[{row['ci95'][0]:+.2f},{row['ci95'][1]:+.2f}]$ & 17\\,/\\,0\\,/\\,0", "alignment row")
if S["suspect frames dropped"]["frames_removed_per_arm"] != 13:
    bad.append("alignment: frames removed is not 13")
need(" 13 fall in the evaluated set", "alignment frame count")

# 4. Fig. 1 and the mechanism paragraph
D = json.loads((ROOT / "runs/decomp_0812/results/cadence_decomposition.json").read_text())["decomposition"]
rel, src = D["rel_buf30"], D["src_buf30"]
need(f"$U$ rises from {rel['U']} to {src['U']} and $D$ from {rel['D']} to {src['D']}", "U/D in mechanism")
need(f"$M$ falls from {rel['M']} to\n{src['M']}", "M in mechanism") if f"$M$ falls from {rel['M']} to\n{src['M']}" in tex else need(f"$M$ falls from {rel['M']} to {src['M']}", "M in mechanism")
need(f"{src['P'] - rel['P']} identities", "identities added")

print("\n".join("FAIL: " + b for b in bad) if bad else "all checks pass")
sys.exit(1 if bad else 0)
