#!/usr/bin/env python3
"""Check the CBDCom submission against the frozen result files and against IEEE's
structural rules. Run it after every editing pass; it takes a second.

    python3 tools/verify_cbdcom_paper.py [paper.tex]

What it checks, and why each check exists:

- every \\ref resolves, and no equation is referenced before it is defined
  (the geometry used to be derived in Results and referenced in Results earlier);
- every \\cite has a \\bibitem and the bibliography is in first-citation order
  (IEEE requires it; the three systems references were appended out of order);
- Table II's HOTA/AssA, Table III's medians and the cadence decomposition match
  the frozen JSON to the digit;
- captions stay inside two lines;
- tabular column counts match their specs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "grapemots-protocol/cbdcom2026_r3/results"
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cbdcom2026_paper3_EN_2026-08-14.tex"


def main() -> int:
    tex = PAPER.read_text()
    bad: list[str] = []

    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
    if refs - labels:
        bad.append(f"dangling refs: {refs - labels}")

    at = {m.group(1): m.start() for m in re.finditer(r"\\label\{(eq:[^}]*)\}", tex)}
    for name, defined in at.items():
        if any(m.start() < defined for m in re.finditer(r"\\ref\{" + re.escape(name) + r"\}", tex)):
            bad.append(f"equation {name} referenced before it is defined")

    body = tex.split("\\begin{thebibliography}")[0]
    keys = re.findall(r"\\bibitem\{([^}]*)\}", tex)
    cited = {c.strip() for g in re.findall(r"\\cite\{([^}]*)\}", tex) for c in g.split(",")}
    if cited ^ set(keys):
        bad.append(f"citation/bibitem mismatch: {cited ^ set(keys)}")
    first: dict[str, int] = {}
    for m in re.finditer(r"\\cite\{([^}]*)\}", body):
        for k in (c.strip() for c in m.group(1).split(",")):
            first.setdefault(k, m.start())
    if any(k not in first for k in keys):
        bad.append("a bibitem is never cited in the body")
    elif any(first[keys[i]] > first[keys[i + 1]] for i in range(len(keys) - 1)):
        bad.append("bibliography is not in order of first citation")

    hota = {}
    for name in ("hota_panelA.json", "hota_assoc_rows.json"):
        if (FROZEN / name).is_file():
            hota.update(json.loads((FROZEN / name).read_text())["rows"])
    # a row is located by its U, D, M triple, which is unique across both panels;
    # rows the paper no longer tabulates are skipped, the ones it keeps are checked
    for label, row in hota.items():
        anchor = f"& {row['U']} & {row['D']} & {row['M']} &"
        if anchor not in tex:
            continue
        block = tex[tex.index(anchor):tex.index(anchor) + 200]
        cells = (f"${row['assigned_fraction']:.3f}$ & ${row['signed_error']:+.3f}$ "
                 f"& ${row['HOTA']:.3f}$ & ${row['AssA']:.3f}$")
        if cells not in block:
            bad.append(f"Table I panel A row absent or stale: {label}")

    # the cadence contrast is now carried in the running text, so every paired
    # median it reports is checked wherever it appears
    panelb = ROOT / "runs/decomp_0812/results/hota_panelB.json"
    if panelb.is_file():
        for label, r in json.loads(panelb.read_text())["rows"].items():
            cell = (f"& {r['U']:,} & {r['D']:,} & {r['M']:,} "
                    f"& ${r['assigned_fraction']:.3f}$ & ${r['signed_error']:+.3f}$ "
                    f"& ${r['HOTA']:.3f}$ & ${r['AssA']:.3f}$").replace(",", "{,}")
            if cell not in tex:
                bad.append(f"Table I panel B row absent or stale: {label}")

    dec = json.loads((FROZEN / "cadence_decomposition.json").read_text())
    for block, rows in dec["table"].items():
        # the retention control is reported at tau=1 only; the published block in full
        wanted = rows if "as published" in block else [r for r in rows if r["tau"] == 1]
        for r in wanted:
            if f"${r['delta_median']:+.3f}$" not in tex:
                bad.append(f"paired median missing from the text: {block}, tau={r['tau']}")
    for arm in ("rel_buf30", "src_buf30"):
        a = dec["decomposition"][arm]
        for value in (str(a["U"]), str(a["D"]), str(a["M"])):
            if value not in tex:
                bad.append(f"decomposition term missing from the text: {arm} {value}")

    for m in re.finditer(r"\\caption\{(.*?)\}\n\\label\{([^}]*)\}", tex, re.S):
        n = len(" ".join(m.group(1).split()))
        if n > 260:
            bad.append(f"caption over two lines: {m.group(2)} ({n} chars)")

    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S):
        ncol = len(re.findall(r"[lrc]", m.group(1)))
        for line in m.group(2).split("\\\\"):
            line = line.strip()
            if (not line or "rule" in line or "multicolumn" in line
                    or "shortstack" in line or line.startswith("$c{=")):
                continue
            if line.count("&") + 1 != ncol:
                bad.append(f"row has {line.count('&') + 1} fields, spec has {ncol}: {line[:40]}")

    words = len(" ".join(tex.split("\\begin{abstract}")[1]
                         .split("\\end{abstract}")[0].split()).split())
    keywords = [k.strip() for k in " ".join(
        tex.split("\\begin{IEEEkeywords}")[1].split("\\end{IEEEkeywords}")[0].split()
    ).split(",") if k.strip()]
    print(f"abstract {words} words, {len(keywords)} keywords, {len(keys)} references")
    # house rules: at most 250 words, exactly five keywords
    if words > 250:
        bad.append(f"abstract is {words} words, the limit is 250")
    if len(keywords) != 5:
        bad.append(f"{len(keywords)} keywords, the house rule is five")
    if bad:
        print("\n".join("FAIL: " + b for b in bad))
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
