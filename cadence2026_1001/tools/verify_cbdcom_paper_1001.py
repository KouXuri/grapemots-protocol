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
        if re.search(r"row has 1 fields, spec has 6: (\\textbf\{)?(circling|multi-view|frontal)", line):
            continue                                   # \shortstack header line
        if "caption over two lines" in line:
            continue                                   # re-checked below, brace-matched
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

# 3b. the whole-frame row and its paragraph (runs/wholeframe_2021_1001)
WF = json.loads((ROOT / "runs/wholeframe_2021_1001/results/wholeframe_2021.json").read_text())
w = WF["whole frame, imgsz 4096 / unseen"]["tau=1"]
d3 = f"{int(w['delta_median'] * 1000 + 0.5) / 1000:+.3f}"
need(f"Whole frame & ${w['sparse_median']:+.3f}$ & ${w['source_median']:+.3f}$ & ${d3}$ & "
     f"$[{w['ci95'][0]:+.2f},{w['ci95'][1]:+.2f}]$ & 17\\,/\\,0\\,/\\,0", "whole-frame row")
if w["up/down/tie"] != "17/0/0" or w["sequences"] != 17:
    bad.append("whole frame: not 17 of 17 rising")
need(f"rises on all 17 sequences by a paired ${d3}$", "whole-frame paired median")
need(f"from ${w['pooled_rel']['e']:+.3f}$ to ${w['pooled_src']['e']:+.3f}$", "whole-frame pooled e")
if not (w["pooled_src"]["U"] > w["pooled_rel"]["U"] and w["pooled_src"]["D"] > w["pooled_rel"]["D"]
        and w["pooled_src"]["M"] < w["pooled_rel"]["M"]
        and w["pooled_rel"]["P"] < 238 and w["pooled_src"]["P"] < 666):
    bad.append("whole frame: U/D/M or 'fewer tracks in both arms' not as stated")
R = lambda f: {r["arm"]: r["decomposition"]["1"]["P"] for r in json.loads(
    (ROOT / f).read_text())["runs"] if r["video"] == "row_4.3_2" and r["arm"] in ("src_buf30", "rel_buf30")}
srv = R("grapemots-protocol/cbdcom2026_r3/results/decomp_fold2_eleven.json")
mac = R("runs/wholeframe_2021_1001/results/repro_tiled_row_4.3_2_mps.json")
need(f"from {srv['src_buf30']} to {mac['src_buf30']} tracks and from {srv['rel_buf30']} to {mac['rel_buf30']}", "Mac tiled re-run")

# 3c. numbers added for the Track 3 review (archive: cadence2026_1001/results)
AR = ROOT / "grapemots-protocol/cadence2026_1001/results"
CF = json.loads((AR / "conf_fill_identity.json").read_text())
for lab in ("Confidence 0.80", "Confidence 0.75"):
    r = CF[lab]
    need(f"& {r['U']} & {r['D']} & {r['M']} & ${r['assigned_fraction']:.3f}$ & ${r['signed_error']:+.3f}$ "
         f"& ${r['IDF1']:.3f}$ & ${r['HOTA']:.3f}$", f"Table III {lab}")
r75 = CF["Confidence 0.75"]
need(f"IDF1 of {r75['IDF1']:.3f} and a HOTA of {r75['HOTA']:.3f}", "0.75 identity in IV-A")
need(f"an IDF1 of {r75['IDF1']:.3f}, while IDF1", "0.75 IDF1 in III-E")
RC = json.loads((AR / "review_checks.json").read_text())
c = RC["2021_held_out"]
need(f"gives {c['rel']['contact_coverage']:.3f} and {c['src']['contact_coverage']:.3f} for the two 2021",
     "contact coverage")
need(f"assigned coverages of {c['rel']['ownership_coverage']:.3f} and {c['src']['ownership_coverage']:.3f}",
     "ownership coverage")
WORDN = {4: "four"}
need(f"because only {WORDN[c['src']['multi_trajectory_tracks'] + c['rel']['multi_trajectory_tracks']]} tracks",
     "multi-trajectory tracks")
lo = RC["2021_leave_one_flight_out"].values()
sp = sorted(x["pooled_e_sparse"] for x in lo); so = sorted(x["pooled_e_source"] for x in lo)
if not all(x["rises"] == x["sequences_left"] for x in lo):
    bad.append("leave-one-flight-out: not every remaining sequence rises")
need(f"between ${sp[0]:+.2f}$ and ${sp[-1]:+.2f}$ and the source rate between ${so[0]:+.2f}$ and ${so[-1]:+.2f}$",
     "leave-one-flight-out ranges")
pv = RC["grapemots_per_video"].values()
md = sorted(x["median_delta_e"] for x in pv)
need(f"each video rises in at least {min(x['rises'] for x in pv)} of its 15, with a median paired change in $e$ "
     f"between ${md[0]:+.2f}$ and ${md[-1]:+.2f}$", "per-video replication")
# 4. Fig. 1 and the mechanism paragraph
D = json.loads((ROOT / "runs/decomp_0812/results/cadence_decomposition.json").read_text())["decomposition"]
rel, src = D["rel_buf30"], D["src_buf30"]
need(f"$U$ rises from {rel['U']} to {src['U']} and $D$ from {rel['D']} to {src['D']}", "U/D in mechanism")
need(f"$M$ falls from {rel['M']} to\n{src['M']}", "M in mechanism") if f"$M$ falls from {rel['M']} to\n{src['M']}" in tex else need(f"$M$ falls from {rel['M']} to {src['M']}", "M in mechanism")
need(f"{src['P'] - rel['P']} identities", "identities added")

# captions, brace-matched (in the template layout a table's \\label follows its tabular)
def braced(t, start):
    depth, i = 0, start
    while True:
        if t[i] == "{": depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0: return t[start + 1:i]
        i += 1
for m in re.finditer(r"\\caption\{", tex):
    cap = " ".join(braced(tex, m.end() - 1).split())
    if len(cap) > 260:
        bad.append(f"caption over two lines ({len(cap)} chars): {cap[:40]}")

# 5. the IEEE template, conference_101719.tex, as the authors were told to follow it
TEMPLATE = (ROOT / "conference_101719.tex").read_text()
def preamble(t):
    head = t.split("\\begin{document}")[0]
    head = head[head.index("\\documentclass"):]
    return [l.strip() for l in head.splitlines() if l.strip()]
if preamble(tex) != preamble(TEMPLATE):
    bad.append("preamble differs from conference_101719.tex")
for phrase in ("This document is a model", "Identify applicable funding", "Given Name Surname",
               "IEEE conference templates contain guidance text", "\\section*{Reproducibility}"):
    if phrase in tex:
        bad.append(f"template or retired text left in: {phrase}")
abstract = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
title = tex.split("\\title{")[1].split("\\thanks")[0]
for name, part in (("abstract", abstract), ("title", title)):
    if "$" in part or "\\cite" in part or "\\footnote" in part:
        bad.append(f"{name} holds mathematics, a citation or a footnote (template forbids)")
# authors listed one by one, in reading order, never grouped by affiliation
if "\\IEEEauthorrefmark" in tex or tex.count("\\IEEEauthorblockN") != 4:
    bad.append("author blocks are not one per author as in the template")
# "Unless there are six authors or more give all authors' names; do not use et al."
SIX_OR_MORE = {"zhang2022", "luiten2021", "jocher2026", "du2023"}   # feng2024 has five (Crossref)
for key, body in re.findall(r"\\bibitem\{([^}]*)\}(.*)", tex):
    if "et al." in body and key not in SIX_OR_MORE:
        bad.append(f"{key}: et al. for fewer than six authors")
# conclusion answers 'so what'; it should not repeat the abstract's numbers
concl = tex.split("\\section{Conclusion}")[1].split("\\section*")[0].split("\\begin{thebibliography}")[0]
num = lambda t: set(re.findall(r"\d+(?:\.\d+)?", t))
shared = num(abstract) & num(concl)
if shared:
    bad.append(f"conclusion repeats abstract numbers: {sorted(shared)}")
if "%" in abstract.replace("\\%", ""):
    pass
if any(t in tex for t in ("%WHOLEFRAME", "%CONCLUSION%", "%ARCHIVE_BIB%")):
    bad.append("placeholder left in the text")

print("\n".join("FAIL: " + b for b in bad) if bad else "all checks pass")
sys.exit(1 if bad else 0)
