#!/usr/bin/env python3
"""De-AI prose check for the Precision Agriculture version (manuscript_pa/main.tex), same rules as
prose_check.py: sentence lengths, semicolons, stock phrases, first person, abstract words (headings counted),
page count of the compiled PDF.
    .venv/bin/python cadence_journal_1119/tools/prose_check_pa.py
"""
import re
import subprocess
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "manuscript_pa"
s = (d / "main.tex").read_text()


def clean(x):
    x = re.sub(r'(?<!\\)%.*', '', x)
    for env in ['table', 'figure', 'equation']:
        x = re.sub(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}', '', x, flags=re.S)
    x = re.sub(r'\\(sub)*section\{([^}]*)\}', r'\2. ', x)
    x = re.sub(r'\\(citet|citep)\{[^}]*\}', 'X', x)
    x = re.sub(r'\\pend\{[^}]*\}', '', x)
    x = re.sub(r'\$[^$]*\$', 'x', x)
    x = x.replace('\\%', '%').replace('~', ' ')
    x = re.sub(r'\\textbf\{([^}]*)\}', r'\1. ', x)
    x = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', '', x)
    x = x.replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', x)


body = clean(s[s.index('\\abstract{'):s.index('\\backmatter')])
sents = [x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', body) if len(x.split()) > 3]
lens = [len(x.split()) for x in sents]
longs = [x for x in sents if len(x.split()) >= 42]
print(f"sentences {len(sents)}, mean {sum(lens)/len(lens):.1f} words, >=42 words: {len(longs)} "
      f"({100*len(longs)/len(sents):.1f}%), semicolons {body.count(';')} "
      f"(one per {len(sents)/max(1, body.count(';')):.1f} sentences)")
for x in longs:
    print(f"  LONG {len(x.split())}: {x[:150]}")
stock = ['furthermore', 'moreover', 'notably', 'crucially', 'plainly', 'to the best of our knowledge',
         'we present', 'novel', 'robust', 'comprehensive', 'seamless', 'leverage', 'delve', 'underscore',
         'pivotal', 'paradigm', 'remarkabl', 'in conclusion', 'it is worth noting', 'shed light', 'landscape']
print("stock phrases:", [b for b in stock if b in body.lower()])
main = clean(s[s.index('\\section{Introduction}'):s.index('\\backmatter')])
print("body words (Introduction to Conclusion, floats excluded):", len(main.split()))
print("first person:", re.findall(r'\b(we|our|us)\b', main, flags=re.I))
a = re.search(r"\\abstract\{(.*?)\}\n\n\\keywords", s, re.S).group(1)
a = re.sub(r"\\textbf\{([^}]*)\}", r"\1", a).replace("\\~{n}", "n").replace("\\%", "%").replace("~", " ")
print("abstract words (headings included):", len(a.split()))
pdf = d / "main.pdf"
if pdf.exists():
    print(subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout.split("Pages:")[1].split()[0],
          "pages")
