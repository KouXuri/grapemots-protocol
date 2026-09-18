# Camera-ready evidence, 2026-09-19 (release `cbdcom2026-r24`)

This directory supersedes every earlier directory in this repository for the
camera-ready version of *Same Footage, Opposite Sign: Cadence, Coverage and
Cancellation in UAV Video Counting* (CBDCom 2026). The camera-ready paper uses one
corpus, GrapeMOTS (Ariza-Sentís et al., *Data in Brief* 54 (2024) 110432), read
full frame, plus AppleMOT as the external check. Earlier directories hold the
submitted version's evidence (the 2023 vineyard release, MOT17/MOT20) and are kept
unchanged.

## Check it

    pip install numpy scipy
    python3 tools/paper_numbers_0919.py --check

It decomposes all 1,307 frozen arms in `raw/` from their per-frame ids and boxes,
recomputes every number below into `results/paper_numbers.json`, and fails if any
differs from `results/expected_numbers.json` (the values as the paper typesets
them). `sha256sum -c SHA256SUMS` checks the files. No imagery, weights or GPU.

## What is in `raw/`

One tarball per family; each member is one arm cut to what the numbers need
(config, overall metrics, per-frame predicted and reference ids and boxes,
timings), values untouched. `raw/MANIFEST.json` gives each stub's source path on
the analysis server and the SHA-256 of both the source and the stub.

| family | arms | what it is |
|---|---:|---|
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
| Abstract, III-A | denser cadence raises the count in 148 of 150 steps | `surface.monotone_steps` |
| III-A | U falls / D falls / M rises from Δ=1 to Δ=8 in 50/49/50 rows | `U_falls_rows`, `D_falls_rows`, `M_rises_rows` |
| III-A | larger scale raises the count in 132 of 160 steps; pooled per flight pattern, error and coverage rise with σ at every interval | `sigma_steps_raising`, `pooled_sigma_monotone` |
| III-A | error at Δ=1 ranges from −0.86 (F2) to +4.75 (C6) | `delta1_error_range` |
| III-A | ByteTrack on annotated boxes over-counts the eight circling sequences, median +2.49 | `../cadence2026/results/regime_analysis.json`, `sequences[*].whole_sequence_error` for `PathPlanning_*` |
| Fig. 1, Fig. 4, III-B | sign-change brackets, 16 rows in six sequences, ordered in 10/10 | `crossing_map`, `surface` |
| III-C, Table II | control rows | `phase_low`, `phase_high`, `ret`, `cu`, `rt` |
| III-C | phase spans 0.07/0.11/0.20, 17 cells flip sign, G on processed frames loses up to 10 | `phase` |
| Abstract, III-D | coverage at every sign change ≤ 0.57, median 0.42 | `coverage_at_sign_change` |
| III-D | 152 of 1,200 arms within ±0.1 reach at most 63 % | `near_zero_arms` |
| Table III | held-out count terms and HOTA | `heldout`; HOTA in `results/hota_timing_session.json` (same session) and `results/hota_heldout.json` (earlier runs, identical) |
| Table III, IV-B | cost per annotated frame | `cost_ms_per_annotated_frame`, `cost_repeat_spread` |
| Table IV | AppleMOT errors | `apple_error` |
| Table V | c = 2.65 (circling), 5.54 (frontal) | `../cadence2026/results/calibration_*.json` |
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
