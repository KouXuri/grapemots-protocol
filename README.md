# Protocol-defined counting in aerial vineyard video

Manifests, frozen results and analysis scripts for our work on how a video-derived
count is defined by the conditions under which it was observed.

**`cadence2026/` is the release for the CBDCom 2026 paper *Cadence, Not the Tracker,
Sets the Sign of Video Count Error*** (tag `cbdcom2026-cadence`). It adds three corpora
beyond GrapeMOTS, the frame-level alignment of a public release to its source video, the
cadence contrast over 28 sequences, the annotation-thinning ladder, and the calibration
tool behind the annotation-interval criterion. It carries its own README.

**`cadence2026_0813/` carries the round added after that** (tag `cbdcom2026-r6`): the
external contrast on MOT17/MOT20, the adaptive-sampling arms, the low-score second-stage
audit, the flight-clustered bootstrap, the on-board and link cost benchmark, and Panel B
of the configuration table. Its README carries a claim-to-file table, and
`cadence2026_0813/tools/smoke_test.py` rebuilds every table value it contributes from the
frozen results with stock Python and no GPU. Start there for anything the manuscript
reports from Tables IV, VI or VII.

The rest of this repository is the earlier release, described below.

The paper argues that a predicted-track count is only interpretable together with the
protocol that produced it, and that per-video manifests and per-video results have to
be released for one paper's number to be comparable with another's. This repository is
that release for our own numbers.

The imagery itself is not redistributed here. GrapeMOTS is published under CC-BY by
Ariza-Sentís et al., *Data in Brief* **54** (2024) 110432.

## Release tags

`cbdcom2026` is the original submission evidence. `cbdcom2026-r2` adds the six-video
out-of-fold (OOF) analysis and the matched 15/30 Hz source-video control used by the
final eight-page manuscript, together with the auxiliary paired sampling-policy runs.
`cbdcom2026-cadence` (2026-08-10) adds `cadence2026/`: four corpora and 51 sequences,
the zero-residual alignment of the 2023 vineyard release to its source video, the
28-sequence cadence contrast, the thinning ladder and the annotation-interval criterion.
`cbdcom2026-r5` (2026-08-13) makes the earlier release runnable from a fresh checkout and
adds the smoke test that proves it. `cbdcom2026-r6` (2026-08-13) adds `cadence2026_0813/`:
the external MOT17/MOT20 contrast, the adaptive-sampling arms, the second-stage audit, the
flight-clustered bootstrap, the edge and link benchmark, and Panel B --- the experiments
the manuscript reports in Tables IV, VI and VII, which no earlier tag carried.
Release tags are treated as immutable snapshots: an existing tag is never moved, and any
later correction receives a new tag.

## Layout

```
splits/     the five video-level assignments A–E, at video granularity
results/    the frozen JSON every number in the paper is read from
tools/      the scripts that produce those JSON files
configs/    run arguments, checkpoint hashes and source-video hashes
tests/      focused regression tests for the released analyses
```

### splits/

`split_{A..E}.json` assign all eleven videos to train / validation / test. Structure is
held fixed across the five and only identity rotates: seven train, two validation (one
long frontal plus one multi-view), two test (both multi-view). The two 1920×1080 videos
(`PathPlanning_1`, `PathPlanning_3`) stay in training in every assignment so evaluation
is purely 3840×2160.

### results/

| file | what it holds |
|---|---|
| `paper_numbers.json` | oracle dropout tables, coverage sensitivity, count-ratio range, per-source-frame drift rates, blocked-window validation, the five split results |
| `count_decomposition.json` | the exact `P − G = U + D − M` decomposition per arm and per video |
| `arm_*.json` | the eight real-detection tracking configurations: IDF1, MOTA, recall, IDSW, fragmentations, and the count surfaces |
| `merge05_*.json` | full-frame COCO box AP on validation and test |
| `eff_unified_*.json` | end-to-end timings split into read / inference / merge |
| `gt_control_turnover.txt` | the COUNT DISTINCT control, per-video turnover and visibility, symmetric-τ sensitivity |
| `drift_model_validation.txt` | held-out mean absolute error against the no-drift baseline |
| `per_video_ledger.tex` | Table I of the paper: per-video frames, tracks, cadence, resolution, split roles |
| `realised_loss_rates.json` | what each observation-loss mode actually removes, per video and per nominal `p` |
| `coverage_rules.json` | trajectories recovered under dominant-overlap ownership against any-overlap, per arm and per video |
| `cadence_control_all_taus.json` | the cadence control refitted at every τ, with per-fit R² |
| `oracle/oracle_master_{bernoulli,block,identity,size}.json.gz` | the four observation-loss sweeps behind Fig. 3 and the decomposition: 143 runs each, 572 rows in total, over the 28-length window grid, each row carrying its `count_error_surface`, `blocked_surface`, `sliding_surface` and `drift_fit` |
| `oracle/oracle_count_surface_v2.json.gz` | a different sweep, kept because the buffer sensitivity is quoted in the paper: four tracker variants (ByteTrack, BoT-SORT without GMC, `track_buffer` 10 and 60) over a 16-length grid, Bernoulli loss only. It also happens to hold 572 rows; it is **not** the four-mode surface |
| `oracle/oracle_frames_*.json.gz` | per-frame predicted and true ids for one seed of each mode |
| `oof/raw/oof_{A..E}_{arm}.json.gz` | 30 compressed split-arm outputs with per-frame IDs and boxes; A--C are six unique model-unseen videos and D--E repeat four videos under different checkpoints |
| `oof/oof_analysis.{json,md}` | pooled and paired OOF decomposition, including the frozen confidence-0.55/ReID direction |
| `oof/oof_protocol_surface_analysis.{json,md}` | recomputation of 1,680 prefix cells and the post hoc ranking-sensitivity audit |
| `fullrate/fullrate_oof_buffer{30,60}.json` | six-video 15/30 Hz controls scored on the same 1,738 annotated times |
| `fullrate/fullrate_summary.{json,md}` | pooled and per-video rate effects, including the decoded-MP4/released-PNG check |
| `sampling/` | six validation evaluations, six training curves and the paired-seed sampling-policy summary |

### tools/

| script | role |
|---|---|
| `oracle_count_surface.py` | feeds ground-truth boxes to the tracker and emits the (L, τ) count-error surface, with the four observation-loss modes |
| `oracle_cmc_check.py` | the camera-motion-compensation control: same oracle run with GMC off and with GMC on real frames |
| `track_grapemots_mot.py` | the real-detection pipeline, tiled or resized detector into ByteTrack / BoT-SORT |
| `evaluate_grapemots_fullframe.py` | COCO box AP on merged full-frame predictions |
| `decompose_count_error.py` | the `U + D − M` accounting |
| `freeze_paper_numbers.py` | applies one frozen definition of a retained cell and writes `paper_numbers.json` |
| `make_split_manifests.py` | turns a `splits/*.json` assignment into Ultralytics manifests |
| `oracle_master.py` | the full oracle sweep: all four observation-loss modes, 28 window lengths, prefix / sliding / blocked windows |
| `measure_actual_loss.py` | replays the thinning without the tracker and reports the realised loss rate of each mode |
| `measure_coverage_rules.py` | ownership-free object recovery, to check what `M` actually counts |
| `cadence_all_taus.py` | the cadence control at every τ rather than τ = 1 |
| `create_grapemots_detection_dataset.py` | MOTS instance maps to the box / track sidecars everything else reads |
| `rebuild_cbdcom2026_figures.py` | builds the paper's figures from the frozen JSON. It emits five panels; the submitted paper uses four of them (see the note on the AssA scatter below) |

## The frozen definition of a retained cell

Every summary statistic in the paper uses one rule, implemented in
`freeze_paper_numbers.py`:

- a cell is retained when `τ ≤ L/2`, `G(L) > 0`, and coverage `G(L)/G(full) ≥ 0.8`;
- per video, take the median over that video's retained cells;
- the headline is the median over the eleven per-video values, so a video with many
  retained cells does not outweigh one with few;
- seeds are pooled within a video before the per-video median is taken.

Reading the same JSON under a different rule gives different numbers. That is the
point of the paper, so the rule is stated rather than assumed.

## Protocol vector used here

| parameter | value |
|---|---|
| window length `L` | 28 values, 5–900 annotated frames, plus each sequence's full length |
| minimum track length `τ` | {1, 2, 3, 5, 8}, applied to predictions; symmetric variant reported as a sensitivity |
| `track_buffer` | {10, 30, 60} |
| detector operating point | confidence {0.10, 0.25, 0.40, 0.55} |
| window enumeration | prefix, sliding, and non-overlapping blocks, all from one continuous tracker pass |
| annotation cadence | per video, in `per_video_ledger.tex` |
| split manifest | `splits/split_{A..E}.json`, video granularity |

## Frame counts

The data article states 5,958 annotated frames and its own per-video table sums to
5,758. Counting the released files gives 5,772 images, of which 5,755 carry an
identically named instance map. Seventeen images have no paired mask: nine consecutive
frames in `NoPathPlanning_1`, seven in `PathPlanning_1`, and the first frame of
`PathPlanning_3`. All numbers in the paper use the 5,755 pairs.

## Software

Ultralytics 8.4.46, PyTorch 2.11, single NVIDIA RTX 2000 Ada (16 GB).

## Reproducing the numbers

Scripts read and write `results/` relative to the working directory, which is the layout
of this repository, so they run from a fresh checkout without editing. Anything needing
the imagery resolves it from `GRAPEMOTS_ROOT`, which should point at a workspace holding
`datasets/grapemots_det_721/` — the box and track sidecars, built from the GrapeMOTS
instance maps by `create_grapemots_detection_dataset.py`. No script contains an absolute
path to an author machine.

Frozen inference outputs retain the original `weights`, `root` and `video_path` strings
as provenance. The released analysis scripts do not dereference those fields; their
portable inputs are the frame-level arrays and count surfaces stored in the same files.

A smoke test that needs neither imagery nor GPU:

```bash
python tools/freeze_paper_numbers.py --results results --out /tmp/check.json
python - <<'EOF'
import json
released = json.load(open('results/paper_numbers.json'))
rebuilt  = json.load(open('/tmp/check.json'))
for key in released:
    if key == 'blocked_validation':          # float tails differ by ~1e-16
        assert released[key]['wins'] == rebuilt[key]['wins']
        continue
    assert released[key] == rebuilt[key], key
print('paper_numbers.json reproduces from the released results')
EOF
```

That rebuilds every oracle, dropout, coverage, drift and split number the paper quotes,
from this repository alone, with no imagery, no weights and no GPU.

The revision-specific analyses also run without imagery, weights or a GPU:

```bash
# Six configurations on A--E held-out videos. TrackEval 1.3.0 is needed to
# reproduce HOTA; the count decomposition runs when TrackEval is absent.
python tools/analyse_grapemots_oof.py \
  --results results/oof/raw --splits A B C D E --primary-splits A B C \
  --out /tmp/oof_analysis.json

# Exact audit of the A--C prefix surfaces and stored JSON-content hashes.
python tools/analyse_oof_protocol_surface.py \
  --results results/oof/raw --splits A B C \
  --out /tmp/oof_protocol_surface_analysis.json \
  --markdown /tmp/oof_protocol_surface_analysis.md

# Rebuild the 15/30 Hz and time-matched-buffer summary from frozen inference.
python tools/summarize_fullrate_controls.py \
  --default results/fullrate/fullrate_oof_buffer30.json \
  --buffer60 results/fullrate/fullrate_oof_buffer60.json \
  --png-baseline results/oof/raw/oof_A_botsort.json.gz \
  --png-baseline results/oof/raw/oof_B_botsort.json.gz \
  --png-baseline results/oof/raw/oof_C_botsort.json.gz \
  --out /tmp/fullrate_summary.json

# Rebuild the paired sampling-policy summary.
python tools/summarize_sampling_replicates.py \
  --runs-root results/sampling/runs \
  --results results/sampling/current \
  --legacy-results results/sampling/legacy \
  --out /tmp/sampling_summary.json

pytest -q tests
```

`results/SHA256SUMS` covers all revision-specific raw and derived artefacts. The OOF
analysis scripts accept either `.json` or `.json.gz`; provenance hashes are calculated
over the decompressed JSON bytes, so lossless compression does not change the audit.

```bash
shasum -a 256 -c results/SHA256SUMS
```

```bash
export GRAPEMOTS_ROOT=/path/to/your/workspace
```

Three groups of results have different requirements:

| group | needs | scripts |
|---|---|---|
| oracle counting, all eleven sequences | the sidecars only, no model, no GPU | `oracle_master.py`, `measure_actual_loss.py`, `cadence_control.py`, `cadence_all_taus.py` |
| the decomposition and coverage rules | the frozen `arm_*.json` in this repository | `decompose_count_error.py`, `measure_coverage_rules.py` |
| real-detection arms and AP | detector weights, which are not redistributed | `track_grapemots_mot.py`, `evaluate_grapemots_fullframe.py` |

The first two groups reproduce every oracle number, the U/D/M decomposition, the
coverage comparison and the cadence result from what is in this repository plus the
public GrapeMOTS release. The third needs weights we do not publish; the frozen
`arm_*.json` and `oracle/oracle_frames_*.json.gz` are released so those results can be
checked without rerunning inference.

```bash
# oracle surfaces over all eleven sequences, all four observation-loss modes
for mode in bernoulli block identity size; do
  python tools/oracle_master.py --miss-mode $mode \
      --out $GRAPEMOTS_ROOT/results/oracle_master_$mode.json
done

# what each mode actually removes, which is what makes the four comparable
python tools/measure_actual_loss.py

# the camera-motion-compensation control (BoT-SORT with GMC on real frames)
python tools/oracle_cmc_check.py

# does halving the annotation rate move the drift rate on the same footage?
python tools/cadence_control.py

# tracking every source frame at 30 Hz, scored only on annotated frames
python tools/fullrate_tracking.py

# the U + D - M decomposition, at three matching thresholds
for t in 0.3 0.5 0.7; do
  python tools/decompose_count_error.py --match-iou $t \
    --out results/count_decomposition_iou$t.json
done

# HOTA / DetA / AssA from the stored per-frame boxes (TrackEval 1.3.0)
python tools/compute_hota.py

# apply the frozen retained-cell definition and write paper_numbers.json
python tools/freeze_paper_numbers.py
```

## Tracker configurations

`cfg/trackers/` holds every non-default tracker YAML used in the paper. The two
baseline arms use Ultralytics 8.4.46's stock `botsort.yaml` and `bytetrack.yaml`;
those files are reproduced verbatim in `cfg/trackers/ULTRALYTICS_DEFAULTS.txt` so the
comparison does not rest on a library default that may change. The differences from
stock are:

| file | differs from stock by |
|---|---|
| `botsort_nogmc.yaml` | `gmc_method: none` |
| `botsort_gmc.yaml` | `gmc_method: sparseOptFlow`, real frames supplied |
| `botsort_reid_enc.yaml` | `with_reid: True`, `model: yolo11n-cls.pt` (ImageNet, never trained on vineyard data) |
| `botsort_buf{10,60}.yaml`, `bytetrack_buf{10,60}.yaml` | `track_buffer` only |

## Checkpoints and environment

`configs/train_args_splitA.yaml` is the full Ultralytics training configuration for the
detector behind every real-detection number. Its `project` and `save_dir` entries still
name the machine the run happened on; they are left as written because the file is the
run's own record, and nothing reads them. `configs/checkpoints_sha256_and_env.txt`
carries the SHA-256 of the three checkpoints and the complete `pip freeze` of the
environment every result was produced in, with the directly relevant versions
(torch 2.11.0, ultralytics 8.4.46, scipy 1.17.1, trackeval 1.3.0, motmetrics 1.4.0,
numpy 2.5.1, opencv-python 4.13.0.92) called out at the top. The weights
themselves are not in this repository for size reasons; request them from the
corresponding author, or retrain from `train_args_splitA.yaml` and the manifests in
`splits/`.

## Later additions

| file | what it is |
|---|---|
| `results/oracle_decomposition.json` | `U`, `D`, `M` over the full oracle grid: 11 sequences x 4 miss modes x 5 rates x 3 seeds = 572 rows, and the identity `P-G = U+D-M` closes exactly in all of them. Those 572 rows are not 572 independent runs: the `p = 0` run is one shared baseline recorded once per mode, and the size mode is deterministic so its three seeds repeat. Removing both kinds of duplication leaves **451 distinct runs**, 41 per sequence over 17 mode-rate conditions, which is the figure the paper quotes. Produced by `tools/oracle_decomp.py`. |
| `results/oracle_cmc_check.json` | the camera-motion-compensation control, three arms on all eleven sequences |
| `results/cadence_control.json` | thinning the four cadence-1 sequences on the same footage, steps 1/2/3 |
| `results/fullrate_tracking.json` | tracking every source frame at 30 Hz against the 15 Hz annotated subsequence |
| `results/fullrate_buf60.json` | the 30 Hz arm with `track_buffer` doubled, so state retention matches 15 Hz in seconds |
| `results/hota_arms.json` | HOTA / DetA / AssA for the eight arms, TrackEval 1.3.0, from the stored per-frame boxes |
| `results/count_decomposition_iou0.{3,5,7}.json` | the decomposition at three matching thresholds |
| `results/arm_conf0.70.json` | an under-counting real-detection arm (recall 0.071, e = -0.529). Note: run without `--save-frame-tracks`, so it stores boxes but not per-frame identities; its `U`/`D`/`M` are therefore not computed and are not quoted in the paper. |

`tools/gen_cadence_figure.py`, `tools/gen_identity_break_figure.py` and
`tools/gen_missmode_and_alignment_figures.py` regenerate the paper figures from these
files. The last script also emits an AssA-against-count-error scatter that the
submitted version of the paper does not use: the eight arms share two videos and most
of the pipeline, so they are nested variants rather than eight replicates, and the
numbers behind that panel are in Table IV instead.

## Licence

Code in this repository is MIT licensed; see `LICENSE`. The GrapeMOTS imagery it
analyses is CC BY 4.0 and is distributed by Ariza-Sentís et al., *Data in Brief* **54**
(2024) 110432. It is not redistributed here.

Absolute paths inside `results/*.json` record the machine each run was executed on and
are provenance, not inputs; nothing reads them back.
