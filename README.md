# Protocol Sensitivity in Predicted-Track Counting — manifests, results and scripts

Accompanies the CBDCom 2026 paper *Protocol Sensitivity in Predicted-Track Counting:
How Reported Grape-Bunch Counts Depend on Undeclared Parameters in UAV Video*.

The paper argues that a predicted-track count is only interpretable together with the
protocol that produced it, and that per-video manifests and per-video results have to
be released for one paper's number to be comparable with another's. This repository is
that release for our own numbers.

The imagery itself is not redistributed here. GrapeMOTS is published under CC-BY by
Ariza-Sentís et al., *Data in Brief* **54** (2024) 110432.

## Layout

```
splits/     the five video-level assignments A–E, at video granularity
results/    the frozen JSON every number in the paper is read from
tools/      the scripts that produce those JSON files
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
