# cadence2026_0813 — the 08-13 round

`cadence2026/` (tag `cbdcom2026-cadence`) carries the cadence contrast, the
thinning ladder and the annotation-interval criterion. This directory carries
what was added afterwards, and what `cbdcom2026-r5` did **not** contain: the
external contrast on MOT17/MOT20, the adaptive-sampling arms, the low-score
second-stage audit, the flight-clustered bootstrap, the on-board and link cost
benchmark, and Panel B of the configuration table.

Everything here is frozen output plus the script that produced it. The imagery,
the checkpoints and the detection caches are not redistributed; `tools/smoke_test.py`
names them rather than implying they are present.

## Check it

```
python3 tools/smoke_test.py
```

Needs Python 3.9+ and nothing else — no GPU, no imagery, no weights. It hashes
every file in `SHA256SUMS`, then **recomputes** every table value this round
contributes, straight from the frozen results, and compares each against
`results/expected_tables.json`, which holds those values as the manuscript
typesets them. Both directions fail: a result file that changed, and a paper
value that no longer follows from it.

`tools/verify_tables_0813.py path/to/paper.tex` closes the other half of the
loop, matching the same numbers against the manuscript source as typeset, so a
changed rounding shows up.

## Claim to file

| Manuscript | Claim | File | Tool |
| --- | --- | --- | --- |
| Table IV | cadence contrast on densely annotated corpora, $e$ and $1{-}M/G$ per $k$, both arms | `results/ext_cadence_0813/cadence_mot17.json`, `cadence_mot20.json` | `tools/cadence_intervention_mot.py` |
| Table IV, $r$ column | sequence-median $r$ at each $k$ | `results/ext_cadence_0813/geometry_mot17.json`, `geometry_mot20.json` | `tools/thinned_geometry.py` |
| Fig. 1 | three paths to a zero count error, each reaching 0.37--0.48 of the reference; crossings interpolated between the bracketing measured points | `results/fig_cancellation_data.json` (points and crossings), rebuilt from `results/final_analyses_0809`, `cbdcom2026_r3/hota_panelA.json` and `results/adaptive_0813` | `tools/make_fig_cancellation.py` |
| Fig. 2 | one bunch, two tracker identities 7.2 s apart on a frontal 2024 pass, found by the ownership rule rather than by eye | `results/fig_identity_break_data.json`; imagery not redistributed | `tools/make_fig_identity_break.py` |
| §III-A, confidence bracket | two replays of the same caches at 0.75 and 0.80, so the sign change between 0.70 and 0.85 is bracketed 0.12 of coverage apart instead of 0.42 | `results/conf_fill_0815/conf_fill.json` | `tools/conf_fill_summary_0815.py` |
| Fig. 3(a) geometry, 3(b) sign, §III-B and §III-D | sign-crossing $r$ for four corpora; aspect ratios; share of pairs above (2) | `results/ext_cadence_0813/geometry_*.json` | `tools/thinned_geometry.py`, `tools/make_fig_geometry_and_sign.py` |
| §III-C, adaptive arms | one frame budget spent uniformly or by frame difference, 17 model-unseen sequences | `results/adaptive_0813/arms_fold1_six.json`, `arms_fold2_eleven.json` | `tools/cadence_arms_0813.py`, `tools/adaptive_frame_selection.py` |
| §III-C, frame sets | which frames each arm processed | `results/adaptive_0813/frame_sets_17.json` | `tools/adaptive_frame_selection.py` |
| §III-C, low-score arms | +1,415 and +41,443 candidates at extraction floor 0.10, $U,D,M,e$ unchanged | `results/adaptive_0813/arms_*.json` (`rel_low`, `src_low`) | `tools/cadence_arms_0813.py` |
| §III-C, second stage | five sequences, 1,155 low-score candidates offered, 0 accepted | `results/adaptive_0813/second_stage_*.json` | `tools/second_stage_audit.py` |
| Table VI, on board | fps, stage latency, J/frame per configuration | `results/adaptive_0813/edge_*.json` | `tools/edge_cost_bench.py` |
| Table VI, on the link | Mbit/s for the three architectures, all through one x264 encoder at CRF 23 | `results/link_allintra_0814/link_allintra.json` | `tools/link_allintra.py` |
| Table II, IDF1 column | IDF1 for every tabulated row of both panels, from the same per-frame boxes as the U/D/M columns | `results/definition_0815/idf1_table2.json` | `tools/idf1_table2_0815.py` |
| §II-D, ownership gate | $e$ unchanged at ownership IoU 0.2/0.3/0.5/0.7 while coverage reads 0.42/0.41/0.35/0.25 against 0.75/0.74/0.69/0.53; 0.2 is the threshold Piazolo et al. (2026) match at | `results/definition_0815/definition_sensitivity_iou.json` (0.3/0.5/0.7 also in `definition_sensitivity.json`, which differs by one track between $U$ and $D$ on the source arm; $U+D$, $M$, $P$, $G$, $e$ and coverage agree) | `tools/definition_sensitivity_0815.py --match-ious 0.2 0.3 0.5 0.7` |
| §II-D, identity family | 19 duplicates against 18 switches and 86 against 101; $M$ 219/106 against mostly-lost 259/139; IDF1 0.284 against 0.421 | `results/definition_0815/identity_metrics.json` | `tools/identity_metrics_0815.py` |
| §III-C, symmetric $\tau$ | the reference filtered by the same $\tau$: $G$ 339 to 299 at $\tau=3$, both arms still negative | `results/definition_0815/definition_sensitivity.json` | `tools/definition_sensitivity_0815.py` |
| Table III, controls | elapsed-time prediction on the sparse arm, and the association gate at IoU 0.3 and 0.5 in both arms | `results/timescale_0815/timescale_fold1.json`, `timescale_fold2.json`, `timescale_summary.json` | `tools/cadence_timescale_0815.py`, `tools/timescale_summary_0815.py` |
| Table II, Panel B | eleven sequences, each read by a checkpoint blind to it | `results/decomp_0812/hota_panelB.json` | `tools/hota_panelB.py` |
| §III-C, flights | flight-clustered bootstrap, $\tau=1$ interval $[+0.84,+1.58]$, 4/4 flights positive | `results/decomp_0812/cluster_bootstrap.json` | `tools/cluster_bootstrap.py` |
| §IV-B, estimating $c$ | the trajectories a tracker loses move 1.9 and 2.4 times faster than those it keeps, so its own tracks return about half the true $c$ | `results/c_bias_0814/c_estimator_bias.json` | `tools/c_estimator_bias.py` |
| §II-B, detector quality | out-of-fold full-frame AP after merge, by source resolution: median AP$_{50}$ 0.377 over nine 4K videos and 0.086 over two 1080p ones | `results/ap_lovo_0814/lovo_ap_summary.json` | `tools/summarise_lovo_ap.py` |

## What each layer of support means

The manuscript's claims rest on four different things, and they are not
interchangeable:

1. **frozen-output audit** — hash a file here and read the number off it. Every
   row of Tables II-B, IV and VI is at this level.
2. **released-output re-analysis** — re-run a script here over the frozen
   per-frame output and get the same summary. This is what `smoke_test.py` does.
3. **tracker replay** — re-run association over a released detection cache.
   Needs the caches, which are in `cadence2026/` for the vineyard arms and are
   not carried for MOT17/MOT20.
4. **end-to-end detector inference or training** — needs the imagery and the
   checkpoints. Neither is redistributed here.

## On the on-board numbers

`edge_*.json` was timed on a desktop RTX 2000 Ada, not an airborne payload, and
`joules_per_frame` is `nvidia-smi` board power divided by throughput: it excludes
CPU, memory, storage and camera, and the sparse-optical-flow motion compensation
has a substantial CPU path. The link rows are not encoded alike either — the
full-rate figure is the released MP4's own bitrate, the sparse figure is JPEG
q90. Both limits are stated in the manuscript; neither number should be carried
to another platform.
