# cadence2026_1001 — camera-ready revision of 1 October 2026

Evidence added for the camera-ready version of *Same Footage, Opposite Sign:
Cadence, Coverage and Cancellation in UAV Video Counting* (CBDCom 2026), the
version built on the 2021 vineyard campaign (the earlier release) that the two
Accept reviews read. The 2021 intervention itself and everything not listed below
are in `cadence2026_0813/`, `cbdcom2026_r3/` and `cadence2026/`, unchanged.

    python3 cadence2026_1001/tools/smoke_test.py     # from the archive root; needs SciPy

rebuilds every result below from archived per-frame outputs, with no imagery,
weights or GPU, compares each with its frozen file, and re-checks every number the
manuscript prints outside its tables (`tools/claims_audit.py`, 40 claims).

| Manuscript claim | Frozen result | Tool and input |
|---|---|---|
| §III-C and Table III: ten GrapeMOTS sequences, both arms scored at every kth labelled frame against the same reference; the count rises in 148 of 150 comparisons, one falls by four tracks, one ties; opposite signs in 20, on six sequences; arms within ±0.10 reach a median 44% of the reference, at most 64% | `results/gm_matched.json` | `tools/gm_matched.py` on `cadence2026_0919/raw/lovo_surface.tar.gz` |
| §III-C: ByteTrack on the same detections, 149 of 150 | `results/gm_matched_bytetrack.json` | same tool, `bytetrack.tar.gz` |
| §III-C: retrained detectors, 148 / 139 / 147; ten of the sixteen exceptions on `NoPathPlanning_1`, seed 1; circling sequences (plant-disjoint training) 309 of 315 | `results/gm_matched_retrain_s{0,1,2}.json` | same tool, `retrain.tar.gz` split by seed; the frontal folds' seed 0 is the `lovo_surface` arms |
| §III-C: StrongSORT, 100 of 150, 58 of 105 on circling videos | `results/gm_matched_strongsort.json` | same tool, `strongsort.tar.gz` |
| §III-C: on the held-out pair, 8 tiles and the native whole frame both rise in all eight comparisons; at k=8 both give `PathPlanning_2` opposite signs | `results/heldout_readmode.json` | `tools/heldout_readmode.py`, `heldout.tar.gz` |
| §II-A and Table I "Alignment out": 19 labelled images off their assigned source frame, 13 in the evaluated set; without them every sequence still rises, Δ = +1.063 (exactly 1.0625), [+0.78, +1.67] | `results/align_sensitivity.json`, `results/bodegas_alignment_audit/` | `tools/align_sensitivity.py`; the audit is `tools/audit_bodegas_0922.py` |
| §III-B and Table I "Whole frame": the 2021 intervention with each frame read in one letterboxed pass at imgsz 4096 instead of eight 1280 tiles; the count rises on all 17 held-out sequences, Δ = +0.824 [+0.68, +1.15]; pooled e −0.351 → +0.552 (U 94→273, D 16→44, M 229→130); a tiled re-run of `row_4.3_2` on the same machine gives 40 and 18 tracks against the server's 42 and 19 | `results/wholeframe_2021.json`, `results/wholeframe_2021/repro_tiled_row_4.3_2_{mps,cpu}.json` | `tools/wholeframe_table.py` on `raw/wholeframe_2021.tar.gz`; produced by `runner/` |
| Fig. 1 | `cadence2026_0813/results/decomp_0812/cadence_decomposition.json` | `tools/make_fig_overview_1001.py` |
| Fig. 3: 2021 arms cross zero at r = 1.85, the GrapeMOTS ladder at 3.89 | `cadence2026_0813/results/ext_cadence_0813/geometry_*.json` | `tools/make_fig_geometry_and_sign_1001.py` |

**Whole-frame read of the 2021 intervention.** The 2021 checkpoints
(SHA-256 as `cbdcom2026_r3/results/input_manifest.json`) and the 28 source videos
(same manifest) were re-run on a second machine (Apple M4, MPS; Ultralytics 8.4.46,
as on the server) with `runner/fullrate_decompose_1001.py`, the archived
`tools/fullrate_decompose.py` plus `--detector-mode resize` (one whole-frame pass
through `track_grapemots_mot.resize_raw`, then the same IoU-0.5 merge) and
`--device`. Frame map, tracker file, arms and scoring are the frozen run's
(`runner/inputs/`, `runner/run_*.sh`). Both arms of a sequence come from one decode
and one detection pass, so the row's contrast changes the cadence alone. The
machine is not the server: on `row_4.3_2` the tiled read there gives 40 and 18
tracks against 42 and 19 (CPU and MPS agree with each other), so the row is read
within itself, not digit by digit against the tiled rows. The MPS allocator ran
with watermarks 0.6/1.4: at imgsz 4096 one attention product needs a 4 GB buffer;
the setting governs memory, not arithmetic.

**Design of the GrapeMOTS replication.** The archived surface scores each arm at
the frames it processed. The 2021 intervention reads both arms at the same
instants, so here the dense arm (every labelled frame) is scored only at frames
0, k, 2k, … — exactly the frames the sparse arm processed — against the same
reference. `PathPlanning_1` has no row: it is a 1080p video on which its held-out
checkpoint reaches AP50 0.004 (`cadence2026_0813/results/ap_lovo_0814/`), and the
surface was run on the other ten. The video-disjoint checkpoints share vines with
their test video through the frontal passes; for the seven circling sequences the
retrained detectors are trained on circling videos only, each of which films a
different vine. No plant-disjoint training set exists for the three frontal
sequences. The held-out pair is read by the split-A checkpoint, whose training set
includes two frontal videos, so it too is video-disjoint, not plant-disjoint.

**Matching.** `decompose_count_error.py` assigns boxes by maximum total IoU when
SciPy is present and greedily otherwise. Counts, signs and pooled errors depend
only on P and G and are identical either way; U and D can move by one track. All
frozen files here were built with SciPy and reproduce every decomposition the
server embedded in `cbdcom2026_r3/results/decomp_fold*.json`.

**Claims withdrawn from the accepted text** (each with the value that ruled it
out, printed by `tools/claims_audit.py`): three sequences within 0.10 of zero at
the released cadence (there is none); a tracker's own tracks returning "about
half" the true c (0.82 and 0.41 of it); the frame-difference budget "crossing
zero at half the frames" (an interpolation, replaced by the measured pair at 715
frames, +0.271 against −0.094); "association rather than detection limits" a
cut-last platform (8-tile YOLO26s detects in 197 ms and tracks in 79 ms; the
tracking step alone caps it near 12 fps).

**AppleMOT** (`apple_matched.json`, `geometry_applemot.json`, `tools/apple_*.py`)
was the replication in r29–r31. It is no longer in the manuscript, which uses
GrapeMOTS instead; the files stay so those tags remain reproducible.

**Why the alignment was re-checked.** The accepted text said every labelled frame
aligned "at a median residual of exactly 0.000". The aligner's monotone repair
replaces a duplicate or decreasing match with the previous index plus one without
recomputing its residual, so that figure did not certify the stored mapping. A
full-resolution re-decode puts 660 of 679 images within a mean absolute RGB
difference of 5 of their assigned frame; 19, all but one the second image of a
sequence, differ by 14–42. The manuscript says so and reports the intervention
without them.
