# Camera-ready evidence, 2026-09-19/21 (releases `cbdcom2026-r24` to `-r28`)

This directory supersedes every earlier directory in this repository for the
camera-ready version of *Same Footage, Opposite Sign: Cadence, Coverage and
Cancellation in UAV Video Counting* (CBDCom 2026). The camera-ready paper uses one
corpus, GrapeMOTS (Ariza-Sentís et al., *Data in Brief* 54 (2024) 110432), read
full frame, plus AppleMOT as the external check. Earlier directories hold the
submitted version's evidence (the 2023 vineyard release, MOT17/MOT20) and are kept
unchanged.

`cbdcom2026-r28` changes no arm and no frozen output. It answers the co-author
re-review of 2026-09-21 (v3.0): the claim table below now points every claim at the
experiment the paper currently reports (Table III at the `apple_val0000` grid, the
AppleMOT HOTA at 0.524, the common-timeline HOTA, the cutoff rows, the ownership
threshold, the cost frontiers), and `tools/robustness_0921.py --check` now also
recomputes the two mask-cutoff rows of Table I, the direction in which brackets
move, the exact arm, numerator and denominator of every coverage maximum the
paper states, and both cost frontiers. Two corrections came out of it. Table I's
400-px row had its moved brackets as 0 sparser / 2 denser; they are 3 sparser /
0 denser (PP3 loses ten reference trajectories, so its error rises). And the
phase rows of `robustness_numbers.json` had taken coverage from the phase-0 arm
instead of the phase arm they pick; no paper number used those two fields. The
retrained-surface fallback to the published checkpoint is now limited to the 60
frontal seed-0 arms the design reuses (`tools/build_rev_lovo_0920.py`); any other
missing arm is an error.

`cbdcom2026-r27` added the robustness round of 2026-09-21: 540 arms from 27
retrained detectors (three seeds; for the seven multi-view sequences trained on
multi-view videos only, so no vine is shared with the test video), 200 arms with
ByteTrack and 200 with StrongSORT on the same detections, the AppleMOT grid read
with the detector validated on sequence 0000, and the analyses that turned five
of the paper's limitations into measurements (mask-cutoff sensitivity, ownership
threshold, common-timeline HOTA, the ownerless-track audit, the phase mean).
Note `results/hota_applemot_calibration.json`: the 0.5497 stored in August was the
pooled convention this project retracted on 2026-08-30, and the per-sequence value
the paper cites is 0.5243.

`cbdcom2026-r26` added the tracker configs, the strict/tie split of the direction
counts and the phase-combination check, and states what the check covers.
`cbdcom2026-r25` (same day) changes no number: the figure scripts draw 8-pt text
and label sequences by their release names, and the check adds the AppleMOT
duplicate-pair comparison.

## What can be checked, and how

**From the frozen outputs (no imagery, weights or GPU).**

    pip install numpy scipy
    python3 tools/paper_numbers_0919.py --check
    python3 tools/robustness_0921.py --check

`paper_numbers_0919.py` decomposes the 1,307 arms of the first camera-ready
freeze from their per-frame ids and boxes and recomputes the principal count and
timing summaries into `results/paper_numbers.json`: the direction counts (strict,
ties, reversals), the sign-change map, the retention/clock controls and phase
extremes, the phase combinations, coverage at sign changes and over the near-zero
arms, the held-out count terms and cost per annotated frame. It compares them with
`results/expected_numbers.json` (22 entries). Its `apple_error` keys are the
superseded AppleMOT grid (family `apple`, below), kept so the earlier releases
still check; the paper's Table III is `apple_val0000`.

`robustness_0921.py` reads the robustness families and writes
`results/robustness_numbers.json`, compared with `results/expected_robustness.json`
(25 entries): every row of Table I, including the sparser/denser split of the
moved brackets and the coverage maximum with its arm and fraction; the two
mask-cutoff rows; the retrained seed spread; coverage at sign changes under
ownership IoU 0.3 and 0.2, under any combination of phases, and over the 152
near-zero arms of the surface and its controls; the AppleMOT `apple_val0000` grid
of Table III with each arm's coverage and the unique-three-sequence check; and the
held-out frontiers of HOTA against cost, on processed frames and on the common
timeline.

Neither script reads the manuscript. Neither recomputes HOTA (that needs
TrackEval; `tools/compute_hota.py` and `tools/common_timeline_hota_0920.py`), the
calibration of `c`, the size slopes, the pilot-clip statistic or the ownerless
audit; those come from the files named in the claim table. **The cutoff rows are
recomputed from `results/cutoff_sensitivity.json`,** the per-arm terms of the
frozen predictions scored against references rebuilt at 1/20/100/400 mask pixels.
Rebuilding those references (`tools/cutoff_sensitivity_0920.py`) needs the
GrapeMOTS instance masks and runs on the analysis server's layout; it is not a
download-and-run. `python3 tools/cutoff_stats_0920.py [file]` summarises the
archived file (or one you rebuilt). `sha256sum -c SHA256SUMS` checks the files.

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
| `lovo_surface` | 200 | ten out-of-fold sequences × σ ∈ {1536, 2048, 2560, 3072, 3840} × Δ ∈ {1, 2, 4, 8}, phase 0, 30-frame buffer; each sequence read by a leave-one-video-out checkpoint |
| `phase` | 550 | the same cells at every other phase k = 1..Δ−1 |
| `ret` | 150 | Δ ≥ 2 with the buffer scaled to 15/8/4 processed frames, holding the retention at its Δ = 1 value: 2 s on multi-view and 1 s on frontal sequences (2.13 s and 1.07 s at Δ ≥ 4, since buffers are whole frames) |
| `cu` | 150 | Δ ≥ 2 with Δ−1 extra Kalman predictions before each update (elapsed-time clock) |
| `rt` | 150 | both |
| `retrain` | 540 | 27 retrained detectors (seeds 0-2; multi-view folds trained on multi-view videos only), each read on its own test video over the same grid. The three frontal folds keep their published seed-0 checkpoint, so their seed 0 is the 60 `lovo_surface` arms |
| `bytetrack`, `strongsort` | 200 + 200 | the ten-sequence surface with two other trackers on the same detections |
| `heldout` | 23 | held-out pair (PP2 + PP4) with the fixed-split checkpoint: full-frame grid and 8-tile arms |
| `timing` | 72 | held-out pair, every full-frame scale and the tiled arm at Δ ∈ {1, 2, 4, 8}, one session, three repeats, per-frame timings |
| `apple_val0000` | 12 | **Table III.** AppleMOT, σ ∈ {640, 960, 1280} × Δ ∈ {1, 2, 4, 8}, detector trained on 0001–0005 and validated on 0000, scored on the six published test sequences |
| `apple` | 12 | superseded: the same grid read by a detector selected on the test sequences (`ours_theirsplit`). Table III still printed this grid in the r27 manuscript while its text described `apple_val0000` (re-review item R-15, corrected with r28); kept for the record |

## Claim → source

Section, table and figure numbers are those of the camera-ready PDF (7 pages).
`PN` is `results/paper_numbers.json`, `RN` is `results/robustness_numbers.json`.

| Paper | Claim | Source |
|---|---|---|
| Abstract, III-A, IV-A | denser cadence lowers the error in 147 of 150 steps, one tie, two reversals | `PN` `surface.strict_steps`, `tie_steps`, `reversed_steps`; `RN` `surface` |
| III-A | U falls / D falls / M rises from Δ=1 to Δ=8 in 50/49/50 rows | `PN` `U_falls_rows`, `D_falls_rows`, `M_rises_rows` |
| III-A, IV-A | larger scale raises the count in 132 of 160 steps; pooled per flight pattern, error and coverage rise with σ at every interval | `PN` `sigma_steps_raising`, `pooled_sigma_monotone` |
| III-A | error at Δ=1 ranges from −0.86 (NP2) to +4.75 (PP6) | `PN` `delta1_error_range` |
| III-A | ByteTrack on annotated boxes over-counts the eight multi-view sequences, median +2.49 | `../cadence2026/results/regime_analysis.json`, `sequences[*].whole_sequence_error` for `PathPlanning_*` |
| Fig. 1, Fig. 4, III-B | sign-change brackets, 16 rows in six sequences, ordered in 10/10 | `PN` `crossing_map`, `surface`; figures from `results/lovo_surface_terms.json` |
| Table I, III-C | every row: steps down (ties), sign-change rows (sequences), ordered sequences, brackets moved sparser/denser | `RN` `surface`, `phase_low`, `phase_high`, `phase_mean`, `ret`, `cu`, `rt`, `cutoff_{1,20,100,400}px`, `bytetrack`, `strongsort`, `retrain_seed{0,1,2}` |
| Table I, III-C | cutoff rows: 1/20/100 px byte-identical, 728 trajectories; 400 px drops ten of PP3's trajectories, three brackets move sparser, two out of range | `RN` `cutoff_*px`; per-arm terms `results/cutoff_sensitivity.json` (`tools/cutoff_sensitivity_0920.py`, `tools/cutoff_stats_0920.py`) |
| III-C | PP2 at phase 2 mod Δ reverses the scale ordering; 32 of 640 sequence–phase combinations do, all on PP2 | `PN` `phase_combinations` |
| III-C | phase spans 0.07/0.11/0.20, 17 cells flip sign, G on processed frames loses up to 10 | `PN` `phase` |
| III-C | phase mean: 148 steps down, ordering in nine sequences | `RN` `phase_mean`; `tools/phase_mean_0920.py` |
| III-C | StrongSORT: 34 reversals (18 from Δ 1→2, 16 from 2→4, none from 4→8); PP2 σ3072 counts 43/57/34/6 | `RN` `strongsort`; per-arm P from `raw/strongsort.tar.gz`; `results/surface_stats.json`; `tools/strongsort_track_0920.py` |
| III-C, IV-A | retrained: 147/141/145 steps down, 15–25 brackets move, 54 cells change sign with the seed; frontal-free training moves 15 of 35 multi-view brackets | `RN` `retrain_seed*`, `retrain_seed_spread`; `results/retrain_stats.json` (`tools/retrain_stats_0920.py`) |
| III-C | coverage at a sign change on the retrained surfaces at most 62.1 % (54/87, NP3 seed 2 σ2048 Δ4) | `RN` `retrain_seed2.coverage_max_at` |
| Abstract, contribution 2, III-D | on the default-phase surface (Fig. 4) coverage at every sign change at most 57.5 % (50/87, NP3 σ2560 Δ4), median 42 % over 32 arms | `PN` `coverage_at_sign_change`; `RN` `surface.coverage_max_at` |
| Contribution 2, III-D | other rows of Table I at most 62.1 %; StrongSORT maximum 67.8 % (59/87, NP3 σ3840 Δ4); any phase combination at most 64.5 % (60/93) | `RN` `*.coverage_max_at`, `phase_any_combination` |
| III-D | 152 of the 1,200 surface and control arms within ±0.1; maximum 63.2 % (55/87, NP3 σ3072 Δ4), median 45 % | `RN` `near_zero_arms`; `PN` `near_zero_arms` |
| III-D | ownership at IoU 0.2: maximum 65.5 % (57/87), median 48 %; U moves to D, e unchanged | `RN` `surface_ownership_iou0.2`; `results/ownership_iou_coverage.json` (`tools/ownership_iou_coverage_0920.py`) |
| III-D | ownerless audit on PP6 (28/6/1/5 of 40); 53 % of 506 reach IoU 0.2–0.5 | `results/ownerless_PP6_s3072_d1.json`, `results/ownerless_max_iou_s3072_d1.json` (`tools/ownerless_audit_0920.py`, `tools/ownerless_max_iou_0920.py`) |
| Table II, III-D | held-out count terms and processed-frame HOTA (typeset from the raw values, e.g. 0.26347 → 0.263) | `PN` `heldout`, `timing_r1_terms`; HOTA `results/hota_timing_session.json` (same session) and `results/hota_heldout.json` (earlier runs, identical) |
| III-D | common-timeline HOTA along σ3072: 0.263/0.229/0.204/0.158; losses 18.6 % (processed) and 22.7 % (common) | `results/common_timeline_hota.json` (`tools/common_timeline_hota_0920.py`, needs TrackEval) |
| Table II, III-E | cost per annotated frame; 7.4-fold saving from Δ1 to Δ8 at σ3072 | `PN` `cost_ms_per_annotated_frame`, `cost_repeat_spread`; `results/timing_summary.json` |
| III-E | 11 of 24 timed arms non-dominated in processed-frame HOTA against cost, tiled Δ=8 among them; the common-timeline frontier also 11, all full-frame | `RN` `frontier_processed`, `frontier_common_timeline` |
| Table III, III-F | AppleMOT errors on the six test sequences with the val-0000 detector; no sign change at σ640/960, between Δ1 and Δ2 at σ1280, with 74 % and 58 % coverage (1027/1396, 805/1396); scoring one of each identically annotated pair (0006/0010, 0007/0011, 0008/0012) changes no sign and no ordering | `RN` `apple_val0000`, `apple_val0000_detail`; `results/apple_val0000_surface.json` (`tools/apple_val0000_surface_0920.py`); `raw/apple_val0000.tar.gz` |
| III-F | AppleMOT HOTA 0.524 against 0.455 published | `results/hota_apple_val0000.json` (`apple_s1280_d1`), same value as `results/hota_applemot_calibration.json` `per_sequence` |
| IV-B | c = 2.65 (multi-view), 5.54 (frontal) | `../cadence2026/results/calibration_*.json` |
| IV-B | size slopes −0.82 / −0.69; 197 pilots, median error 41 % | `../cbdcom2026_r3/results/scale_invariance.json`, `pilot_holdout.json` |

## How the arms were produced

`tools/track_grapemots_mot.py` (the only tracker driver; `--frame-offset`,
`--catchup` and per-frame timing added 2026-09-19, and with defaults it
reproduces the earlier outputs bit for bit), driven by
`tools/run_lovo_surface_c.sh`, `tools/run_ctl0919_{phase,clock,timing,apple}.sh`
and, for the robustness round, `tools/rev0920_jobs.py`, `tools/run_rev0920_worker3.sh`,
`tools/build_rev_lovo_0920.py` (retraining splits), `tools/dump_dets_0920.py`
(one detection pass shared by the three trackers) and
`tools/strongsort_track_0920.py`. `tools/build_release_0919.py` and
`tools/build_release_0921.py` cut the stubs. Tracker: BoT-SORT in Ultralytics
8.4.46, whose lost-track buffer counts processed frames and ignores the frame
rate it is given; this is why the retention and clock controls exist.

Figures: `tools/make_fig_samefootage.py`, `make_fig_crossmap.py`,
`make_fig_protocol.py`, reading `results/lovo_surface_terms.json`.

## Not carried

GrapeMOTS and AppleMOT imagery (redistributed by their own archives), detector
checkpoints, and TrackEval (install it to recompute HOTA with `tools/compute_hota.py`).
