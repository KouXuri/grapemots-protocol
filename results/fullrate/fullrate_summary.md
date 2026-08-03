# Full-rate out-of-fold control

The 30 Hz arm processes 3,470 source frames; all conditions are scored at the same 1,738 annotated times.

| video | G | 15 Hz, buffer 30 | 30 Hz, buffer 30 | 30 Hz, buffer 60 |
| --- | ---: | ---: | ---: | ---: |
| PathPlanning_2 | 45 | +1.867 | +2.711 | +2.667 |
| PathPlanning_4 | 93 | +1.495 | +2.215 | +2.151 |
| PathPlanning_5 | 84 | +2.310 | +3.060 | +2.976 |
| PathPlanning_6 | 51 | +6.627 | +7.529 | +7.392 |
| PathPlanning_7 | 65 | +2.538 | +3.462 | +3.446 |
| PathPlanning_8 | 77 | +2.364 | +3.312 | +3.247 |
| pooled | 415 | +2.655 | +3.492 | +3.424 |

Pooled 30 Hz minus 15 Hz error: +0.836.
Time-matching the buffer changes that difference by 8.1%.

The decoded 15 Hz and released-PNG errors differ by 0.022--0.369 across videos (median 0.064); PathPlanning_7 has the largest difference.
