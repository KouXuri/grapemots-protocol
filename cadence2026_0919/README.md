# Camera-ready evidence, 2026-09-19/21 (releases `cbdcom2026-r24` to `-r27`)

This directory supersedes every earlier directory in this repository for the
camera-ready version of *Same Footage, Opposite Sign: Cadence, Coverage and
Cancellation in UAV Video Counting* (CBDCom 2026). The camera-ready paper uses one
corpus, GrapeMOTS (Ariza-Sentís et al., *Data in Brief* 54 (2024) 110432), read
full frame, plus AppleMOT as the external check. Earlier directories hold the
submitted version's evidence (the 2023 vineyard release, MOT17/MOT20) and are kept
unchanged.

`cbdcom2026-r27` adds the robustness round of 2026-09-21: 540 arms from 27
retrained detectors (three seeds; for the seven multi-view sequences trained on
multi-view videos only, so no vine is shared with the test video), 200 arms with
ByteTrack and 200 with StrongSORT on the same detections, the AppleMOT grid read
with the detector validated on sequence 0000, and the analyses that turned five
of the paper's limitations into measurements (mask-cutoff sensitivity, ownership
threshold, common-timeline HOTA, the ownerless-track audit, the phase mean).
`tools/robustness_0921.py --check` recomputes every row of the paper's Table I
from the stubs. Note `results/hota_applemot_calibration.json`: the 0.5497 stored
in August was the pooled convention this project retracted on 2026-08-30, and the
per-sequence value the paper cites is 0.5243.

`cbdcom2026-r26` adds the tracker configs, the strict/tie split of the direction
counts and the phase-combination check, and states what the check covers.
`cbdcom2026-r25` (same day) changes no number: the figure scripts draw 8-pt text
and label sequences by their release names, and the check adds the AppleMOT
duplicate-pair comparison.

## What can be checked, and how

**From the frozen outputs (no imagery, weights or GPU).**

    pip install numpy scipy
    python3 tools/paper_numbers_0919.py --check

The script decomposes all 1,307 frozen arms in `raw/` from their per-frame ids and
boxes and recomputes the principal count and timing summaries into
`results/paper_numbers.json`: the direction counts (strict, ties, reversals), the
sign-change map, the controls, the phase extremes and phase combinations, coverage
at sign changes, the held-out count terms, cost per annotated frame and the
AppleMOT grid. It then compares them with `results/expected_numbers.json`, the
archived values the paper was written from, and fails on any drift. It does
**not** read the manuscript, and it does not recompute HOTA, the calibration of
`c`, the size slopes or the pilot-clip statistic; those come from the files named
in the claim table below (HOTA with `tools/compute_hota.py`, which needs
TrackEval). `sha256sum -c SHA256SUMS` checks the files.

**From video (not a download-and-run).** Rerunning an arm needs the GrapeMOTS or
AppleMOT imagery from their own archives, our detector weights (not
redistributed), and the paths in the `tools/run_*.sh` scripts adjusted to the
local layout. The tracker settings are in `cfg/`: `botsort_gmc.yaml` for every
arm, and `botsort_gmc_buf{15,8,4}.yaml`, which differ from it only in
`track_buffer`, for the retention control.

## What is in `raw/`

One tarball per family; each member is one arm cut to what the numbers need
(config, overall metrics, per-frame predicted and reference ids and boxes,
timings), values untouched. `raw/MANIFEST.json` gives each stub's source path on
the analysis server and the SHA-256 of both the source and the stub.

| family | arms | what it is |
|---|---:|---|
| `retrain` | 540 | 27 retrained detectors (seeds 0-2; multi-view folds trained on multi-view videos only), each read on its own test video over the same grid |
| `bytetrack`, `strongsort` | 200 + 200 | the ten-sequence surface with two other trackers on the same detections |
| `apple_val0000` | 12 | the AppleMOT grid with the detector validated on sequence 0000 |
| `lovo_surface` | 200 | ten out-of-fold sequences × σ ∈ {1536, 2048, 2560, 3072, 3840} × Δ ∈ {1, 2, 4, 8}, phase 0, 30-frame buffer; each sequence read by a leave-one-video-out checkpoint |
| `phase` | 550 | the same cells at every other phase k = 1..Δ−1 |
| `ret` | 150 | Δ ≥ 2 with the buffer scaled to 15/8/4 processed frames (retention ≈ 2 s) |
| `cu` | 150 | Δ ≥ 2 with Δ−1 extra Kalman predictions before each update (elapsed-time clock) |
| `rt` | 150 | both |
| `heldout` | 23 | held-out pair (C2 + C4) with the fixed-split checkpoint: full-frame grid and 8-tile arms |
| `timing` | 72 | held-out pair, every full-frame scale and the tiled arm at Δ ∈ {1, 2, 4, 8}, one session, three repeats, per-frame timings |
| `apple` | 12 | AppleMOT, σ ∈ {640, 960, 1280} × Δ ∈ {1, 2, 4, 8} |

## Claim → key in `results/paper_numbers.json`

| Paper | Claim | Key |
|---|---|---|
| Abstract, III-A | denser cadence raises the count in 147 of 150 steps, one tie, two reversals | `surface.strict_steps`, `surface.tie_steps`, `surface.reversed_steps` |
| III-A | U falls / D falls / M rises from Δ=1 to Δ=8 in 50/49/50 rows | `U_falls_rows`, `D_falls_rows`, `M_rises_rows` |
| III-A | larger scale raises the count in 132 of 160 steps; pooled per flight pattern, error and coverage rise with σ at every interval | `sigma_steps_raising`, `pooled_sigma_monotone` |
| III-A | error at Δ=1 ranges from −0.86 (F2) to +4.75 (C6) | `delta1_error_range` |
| III-A | ByteTrack on annotated boxes over-counts the eight multi-view sequences, median +2.49 | `../cadence2026/results/regime_analysis.json`, `sequences[*].whole_sequence_error` for `PathPlanning_*` |
| Fig. 1, Fig. 4, III-B | sign-change brackets, 16 rows in six sequences, ordered in 10/10 (sequences labelled PP2–PP8 and NP1–NP3 after the release's `PathPlanning_*` and `NoPathPlanning_*`) | `crossing_map`, `surface` |
| III-C, Table I | control rows, steps down with ties | `phase_low`, `phase_high`, `ret`, `cu`, `rt` |
| III-C | PP2 at phase 2 mod Δ reverses the scale ordering; 32 of 640 sequence–phase combinations do, all on PP2 | `phase_combinations` |
| III-C | phase spans 0.07/0.11/0.20, 17 cells flip sign, G on processed frames loses up to 10 | `phase` |
| Abstract, III-D | coverage at every sign change at most 57.5 % (50/87 on NP3, σ 2560, Δ 4), median 42 % | `coverage_at_sign_change` |
| III-D | 152 of 1,200 arms within ±0.1 reach at most 63.2 % | `near_zero_arms` |
| Table II | held-out count terms and processed-frame HOTA (typeset from the raw values, e.g. 0.26347 → 0.263) | `heldout`; HOTA in `results/hota_timing_session.json` (same session) and `results/hota_heldout.json` (earlier runs, identical) |
| Table II, IV-B | cost per annotated frame | `cost_ms_per_annotated_frame`, `cost_repeat_spread` |
| Table III (AppleMOT) | errors on the six published test sequences; scoring one of each identically annotated pair (0006/0010, 0007/0011, 0008/0012) changes no sign and no ordering | `apple_error`, `apple_error_unique3`, `apple_structure` |
| Table IV | c = 2.65 (multi-view), 5.54 (frontal) | `../cadence2026/results/calibration_*.json` |
| IV-B | size slopes −0.82 / −0.69; 197 pilots, median error 41 % | `../cbdcom2026_r3/results/scale_invariance.json`, `pilot_holdout.json` |
| III-E | AppleMOT HOTA 0.550 against 0.455 published | `results/hota_applemot_calibration.json` (`applemot_test_asPublished`) |

## How the arms were produced

`tools/track_grapemots_mot.py` (the only tracker driver; `--frame-offset`,
`--catchup` and per-frame timing added 2026-09-19, and with defaults it
reproduces the earlier outputs bit for bit), driven by
`tools/run_lovo_surface_c.sh` and `tools/run_ctl0919_{phase,clock,timing,apple}.sh`.
`tools/build_release_0919.py` cut the stubs. Tracker: BoT-SORT in Ultralytics
8.4.46, whose lost-track buffer counts processed frames and ignores the frame
rate it is given; this is why the retention and clock controls exist.

Figures: `tools/make_fig_samefootage.py`, `make_fig_crossmap.py`,
`make_fig_protocol.py`, reading `results/lovo_surface_terms.json`.

## Not carried

GrapeMOTS and AppleMOT imagery (redistributed by their own archives), detector
checkpoints, and TrackEval (install it to recompute HOTA with `tools/compute_hota.py`).
