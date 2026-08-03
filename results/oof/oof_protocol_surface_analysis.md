# OOF real-configuration protocol-surface audit

This is a read-only analysis of existing A--C OOF outputs. No inference or training was run.

## Verification

- 18 input JSON files, 6 distinct held-out videos.
- Recomputed 1680 stored prefix cells from frame-level IDs; all matched P, G and e.
- Ground truth and protocol grids matched across all six arms within every split-video cell.

## Retained prefix surface

Rule: `tau <= L/2, G(L) > 0, and G(L)/G(full) >= 0.8`. Winner means minimum `absolute signed relative count error |(P-G)/G|`.

There are 100 video-protocol cells and 600 arm measurements. The winner is unique in 99 cells and tied in 1.

| Video | Cells | Winner changes from full, tau=1 | Pair reversals (of 15) | Winning arms |
|---|---:|---:|---:|---|
| PathPlanning_2 | 5 | 2 | 7 | conf055, conf040, bytetrack |
| PathPlanning_4 | 10 | 2 | 3 | conf055, conf040 |
| PathPlanning_5 | 10 | 0 | 2 | conf055 |
| PathPlanning_7 | 20 | 0 | 2 | conf055 |
| PathPlanning_6 | 25 | 0 | 2 | conf055 |
| PathPlanning_8 | 30 | 10 | 15 | conf055, conf040, ios, botsort, bytetrack, reid |

Confidence 0.55 is the full-sequence, tau=1 reference winner on all six videos. The cell-level winner changes from that reference on 3/6 videos and in 14/100 retained cells. At least one pair reverses on 6/6 videos. Overall, 31/90 arm-pair/video cases reverse, and 15/15 arm pairs reverse on at least one video.

Unique winner counts are descriptive cell counts (videos contribute different numbers of retained L values):

| Arm | Unique wins | Co-wins |
|---|---:|---:|
| conf055 | 86 | 87 |
| conf040 | 5 | 5 |
| ios | 2 | 3 |
| botsort | 2 | 2 |
| bytetrack | 2 | 2 |
| reid | 2 | 2 |

| Pair | Videos with reversal |
|---|---:|
| bytetrack vs reid | 5 |
| conf040 vs ios | 4 |
| conf055 vs conf040 | 3 |
| conf055 vs ios | 3 |
| botsort vs bytetrack | 2 |
| conf040 vs bytetrack | 2 |
| conf055 vs botsort | 2 |
| conf055 vs bytetrack | 2 |
| ios vs bytetrack | 2 |
| botsort vs reid | 1 |
| conf040 vs botsort | 1 |
| conf040 vs reid | 1 |
| conf055 vs reid | 1 |
| ios vs botsort | 1 |
| ios vs reid | 1 |

## Common-grid sensitivity

No literal `(L, tau)` cell survives the retained-cell rule on all six videos (0 common retained cells). The coverage-unfiltered intersection has 39 cells. Across their six-video macro-median absolute errors, all six arms win at least one cell and 15/15 arm pairs reverse.

This common-grid result is sensitivity evidence, not a replacement for the retained analysis, because low-coverage windows are included.

## Interpretation boundary

The existing outputs fully support a within-video retained-surface ranking analysis without new inference. A single six-video retained ranking at every literal L is not identifiable because the retained-L intersection is empty. Producing one would require a declared additional analysis choice, such as normalised sequence fractions or interpolation, not additional model computation.
