#!/usr/bin/env python3
"""Checks for the 2026-10-01 camera-ready (cadence_1001/main.tex).

Runs tools/verify_cbdcom_paper.py's checks, with two deliberate differences --
Panel B of the configuration table is summarised in the text rather than
tabulated, and a multi-line \\shortstack header is not a table row -- then checks
everything this revision adds against the frozen files it was read from:
the GrapeMOTS replication (runs/gm_matched_1001), the alignment row
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

# 2. GrapeMOTS replication: every cell of Table III and every number in its text
GM = ROOT / "runs/gm_matched_1001/results"
B = json.loads((GM / "gm_matched.json").read_text())
for s_ in (1536, 2048, 2560, 3072, 3840):
    cells = [f"${B['by_sigma_k'][f's{s_}_k{k}']['pooled_e_dense']:+.3f}/{B['by_sigma_k'][f's{s_}_k{k}']['pooled_e_sparse']:+.3f}$" for k in (2, 4, 8)]
    rises = sum(B['by_sigma_k'][f's{s_}_k{k}']['rises'] for k in (2, 4, 8))
    opp = sum(B['by_sigma_k'][f's{s_}_k{k}']['opposite_sign'] for k in (2, 4, 8))
    need(f"{s_} & " + " & ".join(cells) + f" & {rises}/30 & {opp} \\\\", f"GrapeMOTS row sigma={s_}")
a = B["all"]
need(f"raises the count in {a['rises']} of the {a['n']} paired comparisons", "replication rises")
WORD = ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen']
need(f"In {a['opposite_sign']} comparisons, on {WORD[len(a['opposite_sign_sequences'])]} of the ten", "opposite-sign count")
falls = [c for c in B["cells"] if c["dense"]["P"] < c["sparse"]["P"]]
if len(falls) != 1 or falls[0]["sparse"]["P"] - falls[0]["dense"]["P"] != 4 or a["ties"] != 1:
    bad.append("replication: not one fall of four tracks and one tie")
near = [(c, arm) for c in B["cells"] for arm in ("dense", "sparse") if abs((c[arm]["P"] - c["G"]) / c["G"]) <= 0.10]
cov = sorted(1 - c[arm]["M"] / c["G"] for c, arm in near)
import statistics as st
if round(100 * st.median(cov)) != 44 or round(100 * cov[-1]) != 64:
    bad.append(f"near-zero coverage median/max {st.median(cov):.3f}/{cov[-1]:.3f}, text says 44%/64%")
BT = json.loads((GM / "gm_matched_bytetrack.json").read_text())["all"]
need(f"ByteTrack on the same detections raises the count in {BT['rises']} of", "ByteTrack rises")
RT = [json.loads((GM / f"gm_matched_retrain_s{i}.json").read_text()) for i in (0, 1, 2)]
need("raise it in " + ", ".join(str(r["all"]["rises"]) for r in RT[:2]) + f" and {RT[2]['all']['rises']}", "retrained rises")
exc = sum(r["all"]["n"] - r["all"]["rises"] for r in RT)
np1 = sum(1 for c in RT[1]["cells"] if c["video"] == "NoPathPlanning_1" and c["dense"]["P"] <= c["sparse"]["P"])
need(f"{WORD[np1]} of the {WORD[exc]} exceptions are one frontal sequence", "retrain exceptions")
if (exc, np1) != (16, 10):
    bad.append(f"retrain exceptions {exc}, on NP1 seed 1 {np1}; text says 16 and 10")
mv = sum(r["by_group"]["multi-view"]["rises"] for r in RT); mvn = sum(r["by_group"]["multi-view"]["n"] for r in RT)
need(f"{mv} of {mvn} comparisons rise", "plant-disjoint circling rises")
SS = json.loads((GM / "gm_matched_strongsort.json").read_text())
if (SS["all"]["rises"], SS["by_group"]["multi-view"]["rises"], SS["by_group"]["multi-view"]["n"]) != (100, 58, 105):
    bad.append("StrongSORT counts changed")
need(f"only {SS['all']['rises']} of the 150 comparisons", "StrongSORT rises")
H = json.loads((GM / "heldout_readmode.json").read_text())
eight = [v for k, v in H.items() if k.startswith(("8 tiles", "whole frame 3840"))]
if sum(d > s2 for v in eight for d, s2, _ in v["per_sequence"].values()) != 8:
    bad.append("held-out read modes: not 8 of 8 rising")
for k in ("8 tiles, k=8", "whole frame 3840, k=8"):
    d, sp, g = H[k]["per_sequence"]["PathPlanning_2"]
    if not (d > g > sp):
        bad.append(f"{k}: PP2 not opposite signs")

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
