# Leave-one-video-out real-pipeline summary

6 model-unseen videos, 9 configurations, matching IoU 0.5.

All decompositions satisfy `P - G = U + D - M`.

| Configuration | U | D | M | Assigned | e |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiles, conf. 0.55 | 323 | 331 | 150 | 0.639 | +1.214 |
| Tiles, conf. 0.40 | 528 | 411 | 113 | 0.728 | +1.990 |
| Tiles + IoS | 544 | 388 | 90 | 0.783 | +2.029 |
| Tiles + BoT-SORT | 713 | 480 | 87 | 0.790 | +2.665 |
| assoc_buf60 | 719 | 822 | 96 | 0.769 | +3.482 |
| assoc_nogmc | 714 | 859 | 96 | 0.769 | +3.559 |
| Tiles + ReID | 993 | 580 | 78 | 0.812 | +3.602 |
| Tiles + ByteTrack | 785 | 813 | 97 | 0.766 | +3.617 |
| assoc_buf10 | 848 | 821 | 99 | 0.761 | +3.783 |

Focal pair conf055 vs reid: both directions hold in 6/6 videos.

