#!/usr/bin/env python3
"""Table III and the decomposition of its two arms, from one run over 17 sequences.

Both arms share the decode, the detections and the scoring instants, so the only
difference between them is which frames reached the tracker and how long a lost
track was kept. Every cell here is rebuilt from the per-frame dumps of that run.

The bootstrap resamples sequences, 10,000 times, percentile interval; the exact
sign test counts ties out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from math import comb

ROOT = ROOT / "runs/decomp_0812/results"
FILES = ["decomp_fold1_six.json", "decomp_fold2_eleven.json",
         "decomp_seen_a.json", "decomp_seen_b.json"]
TAUS = ["1", "3", "5", "8"]
PAIRS = [("Buffer 30 processed frames, as published", "rel_buf30", "src_buf30"),
         ("Buffer 3,000: neither arm can time a track out", "rel_buf3000", "src_buf3000")]


def sign_test(up: int, down: int) -> float:
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def main() -> None:
    runs = []
    for name in FILES:
        runs.extend(json.loads((ROOT / name).read_text())["runs"])
    videos = sorted({r["video"] for r in runs})
    arms = sorted({r["arm"] for r in runs})
    print(f"{len(videos)} sequences, arms {arms}\n")

    by = {(r["video"], r["arm"]): r for r in runs}
    for video in videos:
        for arm in arms:
            if (video, arm) not in by:
                raise SystemExit(f"missing {video}/{arm}")

    out = {"videos": videos, "table": {}, "decomposition": {}}

    print("=== Table III, rebuilt ===")
    for label, rel_arm, src_arm in PAIRS:
        print(f"\n{label}")
        print(f"{'tau':>4} {'released':>10} {'source':>10} {'delta':>10} "
              f"{'95% CI':>18} {'up/down/tie':>13} {'p':>10}")
        rows = []
        for tau in TAUS:
            rel = np.array([by[(v, rel_arm)]["decomposition"][tau]["signed_error"] for v in videos])
            src = np.array([by[(v, src_arm)]["decomposition"][tau]["signed_error"] for v in videos])
            d = src - rel
            rng = np.random.default_rng(0)
            boot = np.array([np.median(rng.choice(d, d.size, replace=True)) for _ in range(10000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            up, down, tie = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
            p = sign_test(up, down)
            rows.append({"tau": int(tau), "released_median": float(np.median(rel)),
                         "source_median": float(np.median(src)), "delta_median": float(np.median(d)),
                         "ci95": [float(lo), float(hi)], "up": up, "down": down, "tie": tie,
                         "sign_test_p": p})
            print(f"{tau:>4} {np.median(rel):>+10.3f} {np.median(src):>+10.3f} "
                  f"{np.median(d):>+10.3f} {f'[{lo:+.2f},{hi:+.2f}]':>18} "
                  f"{f'{up}/{down}/{tie}':>13} {p:>10.2e}")
        out["table"][label] = rows

    print("\n=== Decomposition, pooled over the 17 sequences, tau=1 ===")
    print(f"{'arm':>14} {'P':>6} {'G':>6} {'U':>6} {'D':>6} {'M':>6} {'1-M/G':>8} {'e':>9}")
    for arm in ["rel_buf30", "src_buf30", "rel_buf3000", "src_buf3000"]:
        t = {k: 0 for k in ("P", "G", "U", "D", "M")}
        for v in videos:
            one = by[(v, arm)]["decomposition"]["1"]
            if not one["identity_holds"]:
                raise SystemExit(f"{v}/{arm}: identity fails")
            for k in t:
                t[k] += one[k]
        e = (t["P"] - t["G"]) / t["G"]
        assigned = 1 - t["M"] / t["G"]
        holds = (t["U"] + t["D"] - t["M"]) == (t["P"] - t["G"])
        out["decomposition"][arm] = {**t, "assigned_fraction": assigned,
                                     "signed_error": e, "identity_holds": holds}
        print(f"{arm:>14} {t['P']:>6} {t['G']:>6} {t['U']:>6} {t['D']:>6} {t['M']:>6} "
              f"{assigned:>8.4f} {e:>+9.4f}  identity {holds}")

    # the released arm read the decoded video here; the published run read the
    # release's own image files. Agreement is a second check on the alignment.
    print("\n=== reproduction against the published contrast ===")
    published = json.loads(Path(
        "runs/bodegas_round2_0809/results/"
        "cadence_contrast_release.json").read_text())
    pub = {r["video"]: r for r in published["sequences"]}
    worst_rel = worst_src = 0.0
    for v in videos:
        r = by[(v, "rel_buf30")]["decomposition"]["1"]["signed_error"]
        s = by[(v, "src_buf30")]["decomposition"]["1"]["signed_error"]
        worst_rel = max(worst_rel, abs(r - pub[v]["released"]))
        worst_src = max(worst_src, abs(s - pub[v]["source_rate"]))
    print(f"largest per-sequence difference: released {worst_rel:.6f}, source {worst_src:.6f}")
    out["reproduction"] = {"max_abs_diff_released": worst_rel, "max_abs_diff_source": worst_src}

    (ROOT / "cadence_decomposition_all28.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nwrote", ROOT / "cadence_decomposition_all28.json")


if __name__ == "__main__":
    main()
