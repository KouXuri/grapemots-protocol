# Leave-one-video-out real-pipeline summary

6 model-unseen videos, 6 configurations, matching IoU 0.5.

All decompositions satisfy `P - G = U + D - M`.

| Configuration | U | D | M | Assigned | e |
| --- | ---: | ---: | ---: | ---: | ---: |
| k32 | 81 | 49 | 247 | 0.368 | -0.299 |
| k16 | 158 | 138 | 218 | 0.467 | +0.191 |
| k8 | 251 | 281 | 165 | 0.600 | +0.891 |
| k4 | 359 | 379 | 133 | 0.677 | +1.468 |
| k2 | 478 | 420 | 104 | 0.748 | +1.927 |
| k1 | 713 | 480 | 87 | 0.790 | +2.665 |

Focal pair conf055 vs reid: both directions hold in 0/0 videos.

