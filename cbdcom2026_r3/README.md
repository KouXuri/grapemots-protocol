# Evidence for *Two Ways to Reverse the Sign of a Video Count Error*

Tags `cbdcom2026-r3` and `cbdcom2026-r4`, both 2026-08-12. r4 adds the inputs and
configurations needed to re-run rather than only check r3, and changes nothing in
it. `CHANGELOG.md` records that, and why Table III changed between the submitted
draft and r3.

Checksums in `SHA256SUMS`; verify with

    shasum -a 256 -c SHA256SUMS

Tools resolve paths from `GRAPEMOTS_ROOT`, defaulting to the working directory:

    export GRAPEMOTS_ROOT=$PWD

## What this round adds

Both arms of the cadence intervention are re-run from **one decode and one tiled
detection per frame**, so they share their pixels and their detections and differ
only in which frames reach the tracker and how long it keeps a lost track. Every
arm keeps its per-frame predicted and reference boxes at the annotated instants,
so the ownership analysis behind `P − G = U + D − M` applies to the causal
experiment and not only to the thinning ladder.

| file | what it holds |
| --- | --- |
| `results/decomp_fold1_six.json` | six model-unseen sequences, Piazolo-split checkpoint, four arms, per-frame dumps |
| `results/decomp_fold2_eleven.json` | eleven model-unseen sequences, complementary-fold checkpoint, four arms |
| `results/decomp_seen_a.json`, `_b.json` | the eleven seen-in-training sequences, four arms |
| `results/decomp_buf1_fold1.json`, `_fold2.json` | released arm at one processed frame of retention |
| `results/cadence_decomposition.json` | Table III and the decomposition, 17 model-unseen |
| `results/cadence_decomposition_all28.json` | the same over all 28 |
| `results/hota_panelA.json`, `hota_assoc_rows.json` | HOTA, DetA, AssA and re-derived U/D/M for all eleven rows of Table II |
| `results/scale_invariance2.json` | the 45,819 size-binned pairs behind Table V |
| `results/calibration_2023/` | the 28 per-video calibrations behind c = 6.38 |
| `results/cached_conf/` | per-frame outputs of the confidence 0.70 and 0.85 rows |
| `results/input_manifest.json` | SHA-256 of the two checkpoints and 28 source videos |
| `cfg/trackers/` | the tracker configurations every arm names |

Arms are `rel_buf30`, `rel_buf3000`, `src_buf30`, `src_buf3000`; `rel` sees only
the aligned annotated frames, `src` every source frame.

## Reproducing

Start here, from a clean checkout, with nothing but Python and numpy:

    python tools/smoke_test.py

It verifies every SHA-256 in the manifest, re-runs the analyses that work off the
frozen per-frame dumps, and checks that their output is byte-identical to what
ships. No GPU, no imagery, no weights. What it prints at the end is the boundary
between what this package lets you audit and what it lets you rebuild.

The rest needs the corpora, which are not redistributed:

    python tools/fullrate_decompose.py \
      --root datasets/protocol_ext/bodegas2023 \
      --video-root datasets/protocol_ext/bodegas2023/videos \
      --frame-map <alignment>.json --weights-map <fold>.json --weights <ckpt> \
      --videos <sequences> \
      --arm src_buf30:cfg/trackers/botsort_gmc.yaml:source \
      --arm src_buf3000:cfg/trackers/botsort_gmc_bufinf.yaml:source \
      --arm rel_buf30:cfg/trackers/botsort_gmc.yaml:released \
      --arm rel_buf3000:cfg/trackers/botsort_gmc_bufinf.yaml:released \
      --out results/decomp_<group>.json

    python tools/aggregate_decomp.py        # Table III + decomposition, 17 sequences
    python tools/aggregate_decomp_all28.py  # the same over 28
    python tools/hota_panelA.py             # HOTA / DetA / AssA, confidence rows
    python tools/hota_assoc_rows.py         # the same for the association rows

`P − G = U + D − M` is checked on every sequence and every arm before any number
is used; the aggregate scripts abort if it fails.

## Two checks worth reading

**Reproduction.** The source-rate arm reproduces the published contrast exactly:
the largest per-sequence difference over the 17 model-unseen sequences is
0.000000 in `e`. The released arm differs on 6 of 17 by at most 0.238, because it
now reads the aligned frames of the decoded video instead of the release's own
image files; no sequence changes direction, and the group medians are unchanged
at τ=1.

**A correction.** The buffer-3,000 block published earlier had scored all 17
"model-unseen" sequences with a single checkpoint that had trained on 11 of them.
With the correct out-of-fold checkpoints the block reads 17/0/0 at τ=1, 3 and 5
(it read 15/1/1 at τ=1), the released arm is identical at both buffers, and the
reversal at τ=1 no longer depends on the buffer.
