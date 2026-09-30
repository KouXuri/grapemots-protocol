# cadence2026_1001 — camera-ready revision of 1 October 2026

Archived snapshot: tag `cbdcom2026-r30`. The identical content was first released
as `cbdcom2026-r29`, whose Zenodo archiving stalled after the webhook was accepted;
r30 re-triggers it and adds only this note.

Evidence added for the camera-ready version of *Same Footage, Opposite Sign:
Cadence, Coverage and Cancellation in UAV Video Counting* (CBDCom 2026), the
version built on the 2021 vineyard campaign (the earlier release) that the two
Accept reviews read. Everything else the manuscript reports is in
`cadence2026_0813/` and `cbdcom2026_r3/`, unchanged.

    python3 cadence2026_1001/tools/smoke_test.py     # from the archive root

rebuilds every result below from archived inputs with stock Python and no GPU.

| Manuscript claim | Frozen result | Tool |
|---|---|---|
| Table III and §III-D: AppleMOT with both arms scored at every kth frame against the same reference; 53 of 54 comparisons rise (27 of 27 on the three independently annotated sequences); the pooled error stays negative in every cell; +0.001 on the unique three at σ=1280, k=2 | `results/apple_matched.json` | `tools/apple_matched.py` on `cadence2026_0919/raw/apple_val0000.tar.gz` |
| Fig. 3 AppleMOT points and curve; r = 0.59 at full rate; (U+D)/G = 0.29; crossing at r = 0.66 | `results/geometry_applemot.json` | `tools/apple_geometry.py`, same input |
| §II-A and Table I "Alignment out": 19 labelled images off their assigned source frame, 13 in the evaluated set; without them every sequence still rises, Δ = +1.063, [+0.78, +1.67] | `results/align_sensitivity.json`, `results/bodegas_alignment_audit/` | `tools/align_sensitivity.py`; the audit is `tools/audit_bodegas_0922.py` |
| Fig. 1 | `cadence2026_0813/results/decomp_0812/cadence_decomposition.json` | `tools/make_fig_overview_1001.py` |

**Why AppleMOT was re-scored.** The archived AppleMOT surface
(`cadence2026_0919/results/apple_val0000_surface.json`) scores every arm at the
frames it processed, so its dense arm is read at more instants than its sparse
arm. The vineyard intervention reads both arms at the same instants. Re-scored
that way, the dense arm's pooled error at σ=1280 falls from +0.030 to −0.025 at
k=2 and the sign change the archived surface shows disappears; the rise remains.

**Why the alignment was re-checked.** The accepted text said every labelled frame
aligned "at a median residual of exactly 0.000". The aligner's monotone repair
replaces a duplicate or decreasing match with the previous index plus one without
recomputing its residual, so that figure did not certify the stored mapping. A
full-resolution re-decode puts 660 of 679 images within a mean absolute RGB
difference of 5 of their assigned frame; 19, nearly all the second label of a
sequence, differ by 14–42. The manuscript now says so and reports the
intervention without them.
