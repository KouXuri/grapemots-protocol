#!/usr/bin/env python3
"""Full-length Precision Agriculture version from the verified long manuscript (2026-11-21).

Source: manuscript/main.tex (every number recomputed by journal_numbers.py --check, "identical", and the
remaining ones checked against results/*.json on 2026-11-21). The body is copied verbatim except for the
edits listed in EDITS, each asserted to match exactly once:
  - wording corrected after the source check of 2026-11-21 (Piazolo et al. report 96.8% for SORT+ and 100%
    for DeepSORT+; Ariza-Senti's et al. 2023 judged the tracking results insufficient, not counting; Tu et al.
    2024 give 95.1% and 93.28% for the same counting accuracy, so only their R^2 is cited; ByteTrack's
    ownerless tracks are 3.4% of the reference; the frontal ratio runs from 5.8 to 10; PP6 is counted at
    3.7 times its reference);
  - the language-model paragraph reduced to one sentence;
  - the training table moved to Online Resource 1 (PA template: figures plus tables about 11 at most);
  - hard-coded table numbers replaced by references;
  - figures moved next to their first citation and redrawn at the template's 131-mm text width.
Front matter (abstract of at most 200 words, key points, impact) is written here.
Output: manuscript_pa/main.tex and manuscript_pa/online_resource_1.tex; compile both with tectonic.
    .venv/bin/python cadence_journal_1119/tools/build_pa_full_1121.py
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
src = (HERE / "manuscript/main.tex").read_text()


def ws(old: str) -> str:
    return r"\s+".join(map(re.escape, old.split()))


def rep(text: str, old: str, new: str, count: int = 1) -> str:
    out, n = re.subn(ws(old), lambda m: new, text)
    assert n == count, (n, old[:80])
    return out


body = src[src.index("%% ===================================================================================\n\\section{Introduction}"):
           src.index("%% ===================================================================================\n\\section{Statements and Declarations}")]

EDITS = [
    # Introduction: what the cited sources say
    (r"""Counting studies report the mean absolute error, the root mean square error or the
coefficient of determination of the count \citep{farjon2023}, and a count error near zero is
read as a correct count.""",
     r"""Counting studies report the mean absolute error, the root mean square error or the
coefficient of determination of the count \citep{farjon2023}. A recent passion-fruit counter, for
example, reported an $R^2$ of 0.96 between tracked and manual counts \citep{tu2024}. A count error
near zero is read as a correct count."""),
    (r"""Published counts on two releases from one vineyard in Tomi\~{n}o,
Spain, run from a judgement that tracking was insufficient for counting
\citep{arizasentis2023cea} through a counting accuracy of 96.8\% \citep{piazolo2026} to
consistent under-counting \citep{arizasentis2025jafr}.""",
     r"""Published results on two releases from one vineyard in Tomi\~{n}o, Spain,
run from tracking judged insufficient \citep{arizasentis2023cea} through counting accuracies of
96.8\% and 100\% \citep{piazolo2026} to consistent under-counting \citep{arizasentis2025jafr}."""),
    # Data and detectors: table references
    (r"""as test videos, with 2,138 annotated frames and 472 reference trajectories (Table 1).""",
     r"""as test videos, with 2,138 annotated frames and 472 reference trajectories (Table~\ref{tab:data})."""),
    (r"""Each
test video has its own fold (Table 1).""", r"""Each
test video has its own fold (Table~\ref{tab:data})."""),
    (r"""The whole-frame runs stopped between
epochs 24 and 44 and selected checkpoints from epochs 4 to 24 (Table 2).""",
     r"""The whole-frame runs stopped between
epochs 24 and 44 and selected checkpoints from epochs 4 to 24 (Online Resource 1, Table S1)."""),
    # language models: one sentence
    (r"""\emph{Use of language models.} \pend{To be written by the authors: the Methods must state
what a large language model was used for, beyond copy editing.}""",
     r"""\emph{Use of language models.} A large language model (Claude, Anthropic) assisted with
analysis code and wording, and the authors checked every analysis and number."""),
    # Results: exact values
    (r"""and it was six to ten
times the surplus each associator produced on the frontal passes at the same rate.""",
     r"""and it was 5.8 to 10
times the surplus each associator produced on the frontal passes at the same rate."""),
    (r"""ownerless tracks amounted to at most 3\% of the reference,""",
     r"""ownerless tracks amounted to at most 3.4\% of the reference,"""),
    (r"""the seed with the higher validation mAP$_{50:95}$ (Table 2) had the
larger absolute count error""",
     r"""the seed with the higher validation mAP$_{50:95}$ (Online Resource 1) had the
larger absolute count error"""),
    (r"""five vines came within 14\%, PP8 was counted 44\% too low and PP6 3.7 times too high.""",
     r"""five vines came within 14\%, PP8 was counted 44\% too low and PP6 at 3.7 times its
reference."""),
    # Discussion: table references
    (r"""and a
perfect-detection run that bounds what the associator alone contributes (Table 6).""",
     r"""and a
perfect-detection run that bounds what the associator alone contributes
(Table~\ref{tab:checklist})."""),
    # table headers that overflow the 131-mm text width
    (r"""Test video & Resolution & Annotated frames & Reference trajectories $G$ & Validation vine \\""",
     r"""Test video & Resolution & Annotated frames & Trajectories $G$ & Validation vine \\"""),
    (r"""count. The last row is the change from the real detections to the reference boxes}""",
     r"""count. Change: from $\Delta=1$ to $\Delta=8$. The last row is the change from the real detections
to the reference boxes}"""),
    (r"""Fix & $\Delta=1$ & $\Delta=8$ & Change $\Delta=1\to8$ & Term moved at $\Delta=1$ \\""",
     r"""Fix & $\Delta=1$ & $\Delta=8$ & Change & Term moved at $\Delta=1$ \\"""),
    (r"""they cannot be compared until each is reported in the terms of Table 6.""",
     r"""they cannot be compared until each is reported in the terms of
Table~\ref{tab:checklist}."""),
]
for old, new in EDITS:
    body = rep(body, old, new)

# training table -> Online Resource 1
m = re.search(r"\\begin\{table\}\[t\]\s*\\caption\{Whole-frame detectors:.*?\\end\{table\}\n", body, re.S)
assert m
training_table = m.group(0)
body = body.replace(training_table, "")
assert "Table 2" not in body and "Table 6" not in body and "Table 1)" not in body

# figures: take each figure from the end of the long version, swap the file, put it after the paragraph
# that first cites it
FIGS = [("fig:oracle", "fig1_oracle_associators", "pa_fig1_associators"),
        ("fig:missalone", "fig2_miss_alone", "pa_fig2_miss_alone"),
        ("fig:terms", "fig4_terms_and_cancellation", "pa_fig_terms"),
        ("fig:map", "fig3_sign_change_maps", "pa_fig_maps"),
        ("fig:hota", "fig6_count_vs_hota", "pa_fig_hota"),
        ("fig:vines", "fig5_vines", "pa_fig3_vines")]
for label, old_file, new_file in FIGS:
    fm = re.search(r"\\begin\{figure\}\[p\]\s*\\centering\s*\\includegraphics\[width=\\linewidth\]\{"
                   + old_file + r"\.pdf\}.*?\\end\{figure\}", src, re.S)
    assert fm, old_file
    fig = fm.group(0).replace("[p]", "[t]").replace(old_file + ".pdf", new_file + ".pdf")
    i = body.index("\\ref{" + label + "}")
    j = body.index("\n\n", i)
    body = body[:j] + "\n\n" + fig + body[j:]
body = body.replace("\\begin{table}[t]", "\\begin{table}[t]")

HEAD = r"""% Precision Agriculture (Springer), full version, 2026-11-21. Generated by tools/build_pa_full_1121.py
% from manuscript/main.tex; edit the long version or the builder, not this file.
% Springer Nature LaTeX template v3.1 (Dec 2024), sn-apa reference style. Journal rules (submission
% guidelines and PRAG template 2024, read 2026-11-21): structured abstract (here at most 200 words);
% 4-6 keywords; key points and impact statement; sections not numbered; past tense; LLM use stated in the
% Methods; Statements and Declarations; figures readable in greyscale; figures plus tables about 11 at most.
\documentclass[pdflatex,sn-apa]{sn-jnl}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\graphicspath{{../figures/}}
\setcounter{secnumdepth}{0}
\newcommand{\pend}[1]{\textcolor{red}{[#1]}}
% reference list set small and tight, as in the typeset journal
\let\bibliographytypesize\small
\AtBeginDocument{\setlength{\bibsep}{1pt}}

\begin{document}

\title[Finding more bunches makes the count worse]{Finding more bunches makes the count worse:
identity errors behind tracking-based grape-bunch counts from multi-view UAV video}

\author*[1]{\fnm{Xuri} \sur{Kou}}\email{kouxuri@gmail.com}
\author[2]{\fnm{Musen} \sur{Cheng}}
\author[3]{\fnm{Taihui} \sur{Liu}}
\author[1]{\fnm{Hiroshi} \sur{Masui}}

\affil*[1]{\orgdiv{Division of Information and Communication Engineering}, \orgname{Kitami Institute of
Technology}, \orgaddress{\city{Kitami}, \country{Japan}}}
\affil[2]{\orgname{Pundit (Beijing) Information Technology Limited}, \orgaddress{\city{Beijing},
\country{China}}}
\affil[3]{\orgdiv{School of Computer Science and Technology}, \orgname{Beihua University},
\orgaddress{\city{Jilin}, \country{China}}}

\abstract{\textbf{Purpose} Tracking-based grape-bunch counts from unmanned aerial vehicle (UAV) video are
validated by agreement with a reference. This study asked what the associator and the detector each
add on multi-view flights, and whether agreement holds per vine.

\textbf{Methods} Five associators tracked the reference boxes of the GrapeMOTS videos (Tomi\~{n}o,
Spain, 2023). Six sets of plant-disjoint YOLO26s detectors read seven multi-view videos, one vine each.
Their errors were removed type by type, and counts were calibrated leaving one vine out.

\textbf{Results} With perfect detection every associator over-counted the multi-view videos at 15~Hz,
by 66\% to 279\%, about six to ten times its frontal-pass error. At the working scale the detectors found
22\% to 36\% of the bunches in a frame, and these misses offset the duplicates: supplying them alone
increased the count error in all six detector sets. Totals within 10\% of the reference
mixed over- and under-counted vines, and after calibration one vine was counted at 3.7 times its reference.

\textbf{Conclusion} Agreement with a reference does not validate a multi-view counting pipeline, and a
detector that finds more bunches, with association unchanged, makes the count worse. Duplicate and
missed identities should be reported per vine.}

\keywords{Fruit counting, Multi-object tracking, Unmanned aerial vehicle, Viticulture, Yield estimation}

\maketitle

\noindent\textbf{Key points}
\begin{itemize}
\item With every bunch detected, five trackers over-counted multi-view videos by 66--279\%.
\item Missed bunches offset duplicate identities, so finding more bunches raised the error.
\item Totals that agreed with the reference were wrong vine by vine and did not survive retraining.
\end{itemize}

\noindent\textbf{Impact} UAV bunch counts meant to map yield variability between vines can agree in total
yet fail per vine. The study shows where the errors come from and what must be reported before such
counts inform management.

"""

TAIL = r"""%% ===================================================================================
\backmatter

\bmhead{Supplementary information} Online Resource 1: Table S1, checkpoint selection of the whole-frame
detectors per fold and seed.

\section*{Statements and Declarations}

\bmhead{Funding} This work received no external funding.

\bmhead{Competing interests} The authors declare no competing interests.

\bmhead{Author contributions} \pend{To be confirmed by the authors (CRediT).}

\bmhead{Data availability} GrapeMOTS is available under CC BY 4.0 at
https://doi.org/10.5281/zenodo.10625595 \citep{arizasentis2024dib}. The per-frame outputs of every arm,
the analysis code and a claim-to-file table are archived at \pend{Zenodo version DOI}. Detector weights
are available from the corresponding author on request.

\bibliography{refs}

\end{document}
"""
(HERE / "manuscript_pa/main_full.tex").write_text(HEAD + body + TAIL)

OR1 = r"""% Online Resource 1 of the full Precision Agriculture version. Generated by tools/build_pa_full_1121.py.
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[margin=25mm]{geometry}
\usepackage{booktabs}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}
{\large\bfseries Online Resource 1}\\[6pt]
{\bfseries Finding more bunches makes the count worse: identity errors behind tracking-based grape-bunch
counts from multi-view UAV video}\\[4pt]
\emph{Precision Agriculture}\\[4pt]
Xuri Kou, Musen Cheng, Taihui Liu, Hiroshi Masui\\[2pt]
\small Corresponding author: Xuri Kou, Division of Information and Communication Engineering, Kitami
Institute of Technology, Kitami, Japan; kouxuri@gmail.com
\end{center}

""" + training_table.replace("[t]", "[h]") + r"""
\end{document}
"""
(HERE / "manuscript_pa/online_resource_1_full.tex").write_text(OR1)
print("written main.tex and online_resource_1.tex")
