# GrapeMOTS out-of-fold configuration analysis

Counts use minimum track length 1 and matching IoU 0.50. The Hungarian assignment is used for the per-frame count decomposition.

Pooled values weight trajectories; macro values are medians across paired split-video cells.

## Primary cohort (splits A/B/C)

| Arm | Cells | P | G | U | D | M | Net error | Recovered | Median cell error | Median cell recovery | IDF1 | MOTA | Recall | HOTA | DetA | AssA | LocA | Identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Confidence 0.55 | 6 | 919 | 415 | 323 | 331 | 150 | +1.214 | 0.639 | +1.078 | 0.652 | 0.303 | 0.135 | 0.329 | 0.2578 | 0.2277 | 0.3046 | 0.7481 | 6/6 |
| Confidence 0.40 | 6 | 1241 | 415 | 528 | 411 | 113 | +1.990 | 0.728 | +1.792 | 0.755 | 0.346 | 0.135 | 0.389 | 0.2866 | 0.2694 | 0.3194 | 0.7407 | 6/6 |
| IoS merge | 6 | 1257 | 415 | 544 | 388 | 90 | +2.029 | 0.783 | +1.567 | 0.782 | 0.389 | 0.141 | 0.406 | 0.3071 | 0.2919 | 0.3381 | 0.7370 | 6/6 |
| BoT-SORT | 6 | 1521 | 415 | 713 | 480 | 87 | +2.665 | 0.790 | +2.299 | 0.799 | 0.361 | 0.106 | 0.417 | 0.2980 | 0.2897 | 0.3222 | 0.7364 | 6/6 |
| ByteTrack | 6 | 1916 | 415 | 785 | 813 | 97 | +3.617 | 0.766 | +3.388 | 0.771 | 0.239 | 0.070 | 0.355 | 0.2123 | 0.2498 | 0.1930 | 0.7307 | 6/6 |
| BoT-SORT + ReID | 6 | 1910 | 415 | 993 | 580 | 78 | +3.602 | 0.812 | +3.048 | 0.827 | 0.359 | 0.086 | 0.431 | 0.2998 | 0.2951 | 0.3197 | 0.7337 | 6/6 |

Count identity checks: 36/36 per-cell checks passed.

Confidence 0.55 versus ReID: 6 of 6 paired cells were strict inversions. Of these, 6 favoured confidence 0.55 for absolute count error and ReID for recovery; 0 had the reverse direction.

| Split | Video | conf055 error | ReID error | conf055 recovery | ReID recovery | Lower absolute error | Higher recovery | Inversion |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| A | PathPlanning_2 | +0.311 | +2.889 | 0.378 | 0.622 | conf055 | reid | yes |
| A | PathPlanning_4 | +0.161 | +2.333 | 0.505 | 0.785 | conf055 | reid | yes |
| B | PathPlanning_5 | +1.000 | +2.857 | 0.667 | 0.869 | conf055 | reid | yes |
| B | PathPlanning_7 | +1.662 | +4.062 | 0.815 | 0.923 | conf055 | reid | yes |
| C | PathPlanning_6 | +3.804 | +7.784 | 0.843 | 0.922 | conf055 | reid | yes |
| C | PathPlanning_8 | +1.156 | +3.208 | 0.636 | 0.727 | conf055 | reid | yes |

## Sensitivity cohort (splits A/B/C/D/E)

| Arm | Cells | P | G | U | D | M | Net error | Recovered | Median cell error | Median cell recovery | IDF1 | MOTA | Recall | HOTA | DetA | AssA | LocA | Identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Confidence 0.55 | 10 | 1490 | 701 | 488 | 564 | 263 | +1.126 | 0.625 | +1.078 | 0.652 | 0.261 | 0.135 | 0.305 | 0.2417 | 0.2078 | 0.2923 | 0.7505 | 10/10 |
| Confidence 0.40 | 10 | 2028 | 701 | 799 | 725 | 197 | +1.893 | 0.719 | +1.678 | 0.755 | 0.331 | 0.135 | 0.368 | 0.2790 | 0.2606 | 0.3114 | 0.7428 | 10/10 |
| IoS merge | 10 | 2115 | 701 | 867 | 714 | 167 | +2.017 | 0.762 | +1.567 | 0.779 | 0.373 | 0.141 | 0.391 | 0.3004 | 0.2869 | 0.3276 | 0.7392 | 10/10 |
| BoT-SORT | 10 | 2530 | 701 | 1125 | 863 | 159 | +2.609 | 0.773 | +2.260 | 0.784 | 0.357 | 0.108 | 0.403 | 0.2921 | 0.2856 | 0.3128 | 0.7387 | 10/10 |
| ByteTrack | 10 | 3196 | 701 | 1288 | 1380 | 173 | +3.559 | 0.753 | +3.122 | 0.753 | 0.230 | 0.082 | 0.346 | 0.2096 | 0.2437 | 0.1920 | 0.7338 | 10/10 |
| BoT-SORT + ReID | 10 | 3200 | 701 | 1597 | 1043 | 141 | +3.565 | 0.799 | +3.048 | 0.800 | 0.359 | 0.090 | 0.421 | 0.2986 | 0.2924 | 0.3187 | 0.7359 | 10/10 |

Count identity checks: 60/60 per-cell checks passed.

Confidence 0.55 versus ReID: 10 of 10 paired cells were strict inversions. Of these, 10 favoured confidence 0.55 for absolute count error and ReID for recovery; 0 had the reverse direction.

| Split | Video | conf055 error | ReID error | conf055 recovery | ReID recovery | Lower absolute error | Higher recovery | Inversion |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| A | PathPlanning_2 | +0.311 | +2.889 | 0.378 | 0.622 | conf055 | reid | yes |
| A | PathPlanning_4 | +0.161 | +2.333 | 0.505 | 0.785 | conf055 | reid | yes |
| B | PathPlanning_5 | +1.000 | +2.857 | 0.667 | 0.869 | conf055 | reid | yes |
| B | PathPlanning_7 | +1.662 | +4.062 | 0.815 | 0.923 | conf055 | reid | yes |
| C | PathPlanning_6 | +3.804 | +7.784 | 0.843 | 0.922 | conf055 | reid | yes |
| C | PathPlanning_8 | +1.156 | +3.208 | 0.636 | 0.727 | conf055 | reid | yes |
| D | PathPlanning_4 | -0.022 | +2.118 | 0.484 | 0.763 | conf055 | reid | yes |
| D | PathPlanning_8 | +0.610 | +2.727 | 0.519 | 0.675 | conf055 | reid | yes |
| E | PathPlanning_6 | +3.020 | +7.549 | 0.765 | 0.922 | conf055 | reid | yes |
| E | PathPlanning_7 | +1.323 | +3.262 | 0.754 | 0.815 | conf055 | reid | yes |
