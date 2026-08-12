# Changelog

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
