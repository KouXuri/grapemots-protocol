# journal2026_pa — evidence for the Precision Agriculture article

*When finding more bunches makes the count worse: identity errors in tracking-based grape-bunch counting
from multi-view UAV video*, by Xuri Kou, Wen Qiu, Musen Cheng, Taihui Liu and Hiroshi Masui (submitted to
*Precision Agriculture*, 2026). The article extends the CBDCom 2026 paper whose evidence is in
`cadence2026_1001/` and the directories it lists; nothing in those directories is changed here.

## What is here

| Path | Content |
|---|---|
| `raw/*.tar.xz*` | Per-frame outputs of every run the article uses, as written by the runs (`runs/...` paths; each file under 40 MB, 79 MB in all): the main grid of the whole-frame detectors with its controls (`fullframe_pponly_0922`), BoT-SORT and ByteTrack on the reference boxes (`oracle_cadence_1119`), StrongSORT on the real detections (`wholeframe_strongsort_1119`), and the error substitution, the OC-SORT, Deep OC-SORT and StrongSORT oracle and real-detection runs and the two BoT-SORT interventions (`error_sources_1120`). The crop-trained arms are `cadence2026_0919/raw/retrain.tar.gz`. |
| `results/` | Every frozen result the article cites, and the claim-to-file table `claims_pa_1121.md` / `.csv` (49 rows: printed value, frozen value, file and key, tool). |
| `tools/` | The analysis scripts, run from a workspace laid out by `setup_workspace.sh`; `claims_pa_1121.py` checks every number printed in the article; `make_figures_pa119_1121.py` draws the figures from `results/` alone. |
| `runner/` | The scripts that produced the per-frame outputs on the GPU server (Ultralytics 8.4.46, BoxMOT 23.0.0): detection dumps, replay and substitution (`error_sources_1120.py`), the BoxMOT associators, the queues. They need the GrapeMOTS imagery and the detector weights. |
| `cfg/` | The two BoT-SORT configurations of the interventions, each one value changed from `cfg/trackers/botsort_gmc.yaml`. |
| `figures/` | Figures 1-4 and S1-S2 as submitted. |

## Rebuilding the results

    bash journal2026_pa/setup_workspace.sh /tmp/ws      # about 2 GB once extracted
    python3 journal2026_pa/smoke_test.py /tmp/ws         # Python 3.10+, SciPy, NumPy; no GPU, imagery or weights

The smoke test reruns every analysis script on the archived outputs, compares each rewritten result with its
frozen copy, runs `journal_numbers.py --check`, and finally checks every number printed in the article against
the frozen results (`claims_pa_1121.py`). Values that a frozen file rounds to three decimals are checked on
the unrounded value (`supplementary_checks_1121.py`), with half-up rounding as printed. `identity_metrics.json`
(TrackEval on the GrapeMOTS annotation files) is not rebuilt here.

## Not redistributed

GrapeMOTS imagery and annotations (CC BY 4.0, Ariza-Sentís et al., *Data in Brief* 54 (2024) 110432,
https://doi.org/10.5281/zenodo.10625595; the per-frame outputs carry the reference identities derived from
its masks), the detector weights (available from the corresponding author on request) and the per-frame
detection dumps used for replay.
