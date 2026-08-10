# Cadence, Not the Tracker, Sets the Sign of Video Count Error — release

Artefacts for the IEEE CBDCom 2026 submission of the same title.
Frozen 2026-08-10. **No imagery is redistributed**: the two vineyard corpora are
public under their own licences (Data in Brief 54:110432 and 46:108848), MOT17
and MOT20 labels come from the official TrackEval mirror.

Every number quoted in the paper is read from a file in `results/`. `SHA256SUMS`
inside `results/` fixes their contents.

## Layout

```
results/   frozen analysis outputs, one per experiment (JSON, some Markdown)
tools/     the analysis programs that produced them
figures/   the three figures of the paper, as submitted (PDF)
```

## Which file backs which claim

| Paper | Claim | File |
| --- | --- | --- |
| Table I | Oracle boxes: over-count on 11/11 vineyard 2024, under-count on 26/29 vineyard 2023, pedestrian direction not significant | `bootstrap_headlines.json`, `regime_analysis.json` |
| Table II | Fifteen configurations, none negative | `controlled_association.json`, `lovo_oof_summary.json` |
| Table III, Fig. 2b | Displacement / target size against consecutive reference IoU, all 51 sequences | `sequence_structure.json` |
| Sec. IV-B, Fig. 2a | 28/28 aligned at residual exactly 0.000; 642 of 25,864 frames; median 1.67 Hz; 6/28 below 1 Hz | `bodegas_alignment_all28.json`, `fig_measured_values.json` |
| Sec. IV-C, Fig. 4 | 28/28 counts rise; model-unseen median −0.316 → +0.812; seen −0.600 → +0.903 | `cadence_contrast_release.json`, and its two arms `bodegas_{released,sourcerate}_{all28,fold2}.json` |
| Sec. IV-D, Fig. 1 | Thinning ladder: e +2.665 → −0.299 while assigned 0.790 → 0.368 | `density_realpipeline.json`, `figure_values.json`, `structure_k*.json` |
| Sec. V | c per sequence, spread across lags ≤ 1.07× | `calibration_*.json` |
| Sec. VI-A | Coverage standardisation removes 12% | `effort_standardised_corpora.json`, `effort_standardised_density.json` |

## Reproducing the design rule on your own footage

The rule of Section V needs one densely labelled pilot clip of about a second
(thirty consecutive frames is plenty), in the sidecar layout these tools use:

```bash
python tools/calibrate_annotation_interval.py \
  --root <track_root> --video <name> --size <W> <H> \
  --source-fps <fps> --pilot-step 1
```

It prints `c` at each lag, the spread across lags (a clip whose spread exceeds
1.5x is not a steady pass and the rule does not transfer), and the frame step
for each threshold band. `--quantile 0.9` computes `c` from the fastest and
smallest targets instead of the typical one; the tool reports both regardless.

## Verifying the decomposition

`P - G = U + D - M` holds by construction, and every result was checked against
it before use. `tools/decompose_count_error.py` recomputes it from any tracking
output in `results/`.

## Conventions that matter when reading the files

- Sequences scored by a checkpoint that trained on them are never pooled with
  model-unseen sequences. `cadence_contrast_release.json` carries the grouping in
  its `groups` key and `analyse_cadence_contrast.py` refuses to merge them.
- Protocol cells are retained only if `G(L) > 0`, `tau <= L/2` and
  `G(L)/G(full) >= 0.8`.
- Identities are video-local. `G` counts annotated trajectories within a video,
  not physical objects across views.

## Citation

Pending. Please cite the CBDCom 2026 paper once the proceedings appear.
