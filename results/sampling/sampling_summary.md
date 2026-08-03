# Sampling replicate summary

| representation | seed | best epoch | val AP50 | val AP50--95 | AR@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| step3 | 0 | 4 | 0.2571 | 0.0827 | 0.2305 |
| step3 | 1 | 4 | 0.2363 | 0.0730 | 0.2027 |
| step3 | 2 | 3 | 0.3093 | 0.0999 | 0.2412 |
| step3 mean (population SD) | | | 0.2675 (0.0307) | 0.0852 | |
| step1 | 0 | 1 | 0.2457 | 0.0774 | 0.2253 |
| step1 | 1 | 1 | 0.2636 | 0.0807 | 0.2281 |
| step1 | 2 | 1 | 0.2318 | 0.0721 | 0.2224 |
| step1 mean (population SD) | | | 0.2470 (0.0130) | 0.0767 | |

Paired AP50 differences (`step1 - step3`): seed 0: -0.0114, seed 1: +0.0273, seed 2: -0.0774; mean -0.0205.
