# Changelog

## r10 — 2026-08-14

Fixes what two reviewers hit running r9 from a fresh checkout, and closes the gap
that let a wrong table value survive a round.

- `cbdcom2026_r3/SHA256SUMS` — stale since the r6 CHANGELOG edit, so the older
  smoke test failed on one hash from a clean checkout. Refreshed; it now returns 0.
- `cadence2026_0813/tools/verify_tables_0813.py` — resolved inputs from the
  author's `runs/` layout, which a release checkout does not have, and reported 16
  spurious mismatches there. It now searches the released `results/` layout too,
  so the same script serves an auditor and an author.
- The same verifier now checks `G` and `U` for every row of the external cadence
  table and asserts `P-G = U+D-M` per row. It previously checked only the printed
  subset, which is why a wrong pooled `G` in that table's caption survived: 1,169
  is one video's `G` (MOT20-05), not the four-sequence total of 2,215. The
  manuscript is corrected, and the check now covers 136 cells rather than 112.
- The manuscript's claim that annotated boxes leave `U` empty was wrong and is
  corrected there: oracle boxes remove detector false positives, but a track can
  still drift off every trajectory. MOT20 has `U=83` at `k=4` and `U=12` at
  `k=15`; MOT17 stays at 3 or fewer.

## r6 — 2026-08-13, later the same day

r5 fixed what a reviewer hit running r4, but it carried no new results: a later
reviewer checked the manuscript's claim that every table and figure is auditable
here and found the round of experiments the manuscript had just added missing
from the tag. r6 adds them, in `cadence2026_0813/`.

- `cadence2026_0813/` — new. The external cadence contrast on MOT17/MOT20
  (Table IV), the thinned geometry of all four corpora and the sign-crossing $r$
  behind Fig. 3, the adaptive-sampling arms (Table VI), the low-score
  second-stage audit, the flight-clustered bootstrap, the on-board and link cost
  benchmark (Table VII), and Panel B of the configuration table. 26 frozen result
  files, 13 tools, `SHA256SUMS`, and a README carrying a claim-to-file table.
- `cadence2026_0813/tools/smoke_test.py` — new. Hashes every file, then
  recomputes each of the 28 table entries this round contributes from the frozen
  results and compares them with `results/expected_tables.json`, which holds
  those values as the manuscript typesets them. Fails in both directions. No
  GPU, no imagery, no weights; stock Python 3.9+, no third-party packages.
- `cadence2026_0813/tools/verify_tables_0813.py` — the other half of the loop,
  matching the same numbers against the manuscript source as typeset.
- The README now states the four layers of support separately — frozen-output
  audit, released-output re-analysis, tracker replay, end-to-end inference —
  so "auditable" is not read as covering all four.

One number changed in the manuscript as a result of building this bundle: the
low-score second-stage audit was reported over four sequences and 365 candidates,
but five sequences were run. It is 1,155 candidates over five sequences, still
none accepted. `verify_tables_0813.py` now checks that claim so it cannot drift
again.

Nothing under `cadence2026/`, `results/` or `cbdcom2026_r3/` changed.

## r5 — 2026-08-13

Fixes what a reviewer hit running r4 from a fresh checkout, and adds the test that
would have caught it.

- `tools/aggregate_decomp.py`, `tools/aggregate_decomp_all28.py` — line 19 built a
  path from an undefined `ROOT`, so both raised `NameError` before doing anything.
  They now resolve from `GRAPEMOTS_ROOT`, falling back to the release's own
  `results/`, and locate cross-directory inputs instead of assuming one layout.
- `tools/hota_panelA.py`, `tools/hota_assoc_rows.py` — read the training machine's
  `runs/` layout, which a release checkout does not have. They now search the
  release layout first and, when an input is genuinely not carried here, name the
  file and the directories searched rather than raising a path error.
- `tools/smoke_test.py` — new. Verifies every SHA-256 in the manifest, re-runs the
  analyses that work off the frozen per-frame dumps, and checks their output is
  byte-identical to what ships. No GPU, no imagery, no weights. It also prints the
  four layers of support this package actually provides, so the boundary between
  auditable and rebuildable is stated rather than implied.
- `SHA256SUMS` — refreshed for the files above.

Nothing in `results/` changed: the two analyses re-run here reproduce their frozen
outputs byte for byte, which is what the smoke test asserts.

## r4 — 2026-08-12, later the same day

Nothing in r3 changed. r4 adds the material an auditor needs to go from *checking*
the frozen outputs to *re-running* them, which r3 did not carry:

- `cfg/trackers/*.yaml` — the ten tracker configurations the arms name, including
  `botsort_gmc_bufinf.yaml` (buffer 3,000) and `botsort_gmc_buf1.yaml`, which the
  r3 README referenced without shipping.
- `results/scale_invariance2.json` — the 45,819 same-identity pairs behind Table V,
  by size bin, and `scale_invariance.json` beside it.
- `results/calibration_2023/` — the 28 per-video calibrations whose group median is
  the $c=6.38$ of Table IV; the 2024 per-video calibrations are already in
  `cadence2026/results/`.
- `results/pilot_holdout*.json`, `holdout_transfer.json` — the pilot-versus-tail
  hold-out behind the last sentence of Section III-F.
- `results/cached_conf/` — the per-frame outputs of the confidence 0.70 and 0.85
  rows of Table II, on all six videos, so those two rows can be rebuilt and not
  only checked.
- `results/input_manifest.json` — SHA-256 for the two 2023 checkpoints and for all
  28 source videos.
- `tools/` — the same tools with the training machine's absolute paths removed;
  they now resolve from `GRAPEMOTS_ROOT` or the working directory.

## r3 — 2026-08-12

First release of the cadence decomposition. **Table III changed between the
submitted draft and this one, and the change was a correction, not a re-run.**

The buffer-3,000 block had been computed with a single checkpoint applied to all
17 sequences said to be model-unseen. That checkpoint had trained on 11 of them.
The block is now computed with the same per-fold out-of-fold checkpoints as the
buffer-30 block, from `decomp_fold1_six.json` and `decomp_fold2_eleven.json`.

| $\tau=1$, buffer 3,000 | before | after |
| --- | ---: | ---: |
| released median | $+0.000$ | $-0.316$ |
| source-rate median | $+1.079$ | $+0.781$ |
| paired median | $+1.114$ | $+1.056$ |
| up / down / tie | 15 / 1 / 1 | 17 / 0 / 0 |

The corrected block is the one consistent with the paper's own argument: no 2023
sequence carries more than 28 annotated frames, so the released arm cannot time a
track out at a buffer of 30 either, and raising the buffer to 3,000 must leave
that arm identical. It now does, at every $\tau$.

Two smaller corrections travelled with it. Confidence 0.70, not 0.85, has the
smallest $|e|$ in Table II (0.436). Apparent size moves by 4.0 times across the
Table V bins, not 2.9.

The superseded numbers came from `runs/rebuttal_0811/results/bodegas_*_bufinf_all28.json`
on the training machine, which are unchanged and still carry their own checksums;
they are simply the wrong checkpoint for the group they were reported under.
