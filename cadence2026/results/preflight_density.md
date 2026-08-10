# Countability preflight

Computed from reference annotations only. No detector, no tracker, no training.

Association gate for the 'below gate' column: IoU < 0.3. Lifetimes are compared against a 5-frame track confirmation delay.

## Corpora

| corpus | seq | frames | traj | consec IoU | frac below gate | step/size | lifetime | life/confirm | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| k01 | 11 | 5755 | 757 | 0.635 | 0.161 | 0.185 | 158.0 | 31.6 | associable |
| k02 | 11 | 2880 | 754 | 0.396 | 0.381 | 0.370 | 79.0 | 15.8 | marginal |
| k04 | 11 | 1443 | 752 | 0.104 | 0.656 | 0.737 | 40.0 | 8.0 | degenerate |
| k08 | 11 | 724 | 752 | 0.000 | 0.840 | 1.476 | 20.0 | 4.0 | degenerate |
| k16 | 11 | 365 | 735 | 0.000 | 0.938 | 2.894 | 10.0 | 2.0 | degenerate |
| k32 | 11 | 185 | 715 | 0.000 | 0.998 | 5.933 | 5.0 | 1.0 | degenerate |
| k64 | 11 | 96 | 689 | 0.000 | 1.000 | 11.117 | 3.0 | 0.6 | degenerate |

## Measured count error, for comparison

(no regime file supplied)
