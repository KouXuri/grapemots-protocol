#!/usr/bin/env python3
"""De-AI prose check for manuscript/main.tex: sentence lengths, semicolons, stock phrases,
abstract length, numbers shared by abstract and conclusions.
    python3 cadence_journal_1119/tools/prose_check.py
"""
import re, sys
from pathlib import Path
src = Path(__file__).resolve().parents[1] / "manuscript/main.tex"
s = src.read_text()


def clean(x):
    x = re.sub(r'(?<!\\)%.*', '', x)
    for env in ['table', 'figure', 'equation']:
        x = re.sub(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}', '', x, flags=re.S)
    x = re.sub(r'\\(sub)*section\{([^}]*)\}', r'\2. ', x)
    x = re.sub(r'\\paragraph\{([^}]*)\}', r'\1. ', x)
    x = re.sub(r'\\(citet|citep)\{[^}]*\}', 'X', x)
    x = re.sub(r'\\pend\{[^}]*\}', '', x)
    x = re.sub(r'\$[^$]*\$', 'x', x)
    x = x.replace('\\%', '%').replace('~', ' ')
    x = re.sub(r'\\textbf\{([^}]*)\}', r'\1. ', x)
    x = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', '', x)
    x = x.replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', x)


body = clean(s[s.index('\\begin{abstract}'):s.index('\\section{Statements and Declarations}')])
sents = [x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', body) if len(x.split()) > 3]
lens = [len(x.split()) for x in sents]
longs = [x for x in sents if len(x.split()) >= 42]
print(f"sentences {len(sents)}, mean {sum(lens)/len(lens):.1f} words, >=42 words: {len(longs)} "
      f"({100*len(longs)/len(sents):.1f}%), semicolons {body.count(';')} (one per {len(sents)/max(1,body.count(';')):.1f} sentences)")
for x in longs:
    print(f"  LONG {len(x.split())}: {x[:150]}")
stock = ['furthermore', 'moreover', 'notably', 'crucially', 'plainly', 'to the best of our knowledge',
         'we present', 'novel', 'robust', 'comprehensive', 'seamless', 'leverage', 'delve', 'underscore',
         'pivotal', 'paradigm', 'remarkabl', 'in conclusion', 'it is worth noting', 'shed light', 'landscape']
print("stock phrases:", [b for b in stock if b in body.lower()])
ab = clean(s[s.index('\\begin{abstract}'):s.index('\\end{abstract}')])
ab = re.sub(r'\b(Purpose|Methods|Results|Conclusion)\.\s', '', ab)
print("abstract words:", len(ab.split()))
con = clean(s[s.index('\\section{Conclusions}'):s.index('\\section{Statements and Declarations}')])
num = lambda x: set(re.findall(r'\d+(?:\.\d+)?', x))
print("numbers shared by abstract and conclusions:", sorted(num(ab) & num(con)))
