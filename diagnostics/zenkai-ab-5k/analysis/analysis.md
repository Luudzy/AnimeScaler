# Zenkai A/B: quality and latency

Decision: REVIEW REQUIRED. No model promoted.

Measured Torch model latency plus resize; not player FPS. Quality uses native x2 validation crops.

LPIPS/latency dominance is a numeric proxy; every metric regression remains in analysis.json.

## 640x360-to-1280x720

| Model | Mean LPIPS | Median ms | P95 ms | Proxy dominated by |
|---|---:|---:|---:|---|
| zenkai-a | 0.13151 | 16.86 | 17.21 | adore-lite-v3 |
| zenkai-b | 0.12806 | 19.54 | 19.87 | adore-lite-v3 |
| adore-lite-v3 | 0.12261 | 7.32 | 7.66 | none |
| realesr-anime | 0.10575 | 31.13 | 31.44 | none |
| hfa2k-cugan | 0.12366 | 63.45 | 63.85 | adore-lite-v3, realesr-anime |

## 1280x720-to-1920x1080

| Model | Mean LPIPS | Median ms | P95 ms | Proxy dominated by |
|---|---:|---:|---:|---|
| zenkai-a | 0.13151 | 61.05 | 61.45 | adore-lite-v3 |
| zenkai-b | 0.12806 | 70.78 | 73.16 | adore-lite-v3 |
| adore-lite-v3 | 0.12261 | 25.15 | 25.45 | none |
| realesr-anime | 0.10575 | 119.36 | 119.72 | none |
| hfa2k-cugan | 0.12366 | 237.14 | 237.64 | adore-lite-v3, realesr-anime |

## Regressions against V3

| Model | Case | Metric | Value | V3 |
|---|---|---|---:|---:|
| zenkai-a | clean | mae | 0.009015 | 0.008381 |
| zenkai-a | clean | edgeMae | 0.009844 | 0.009114 |
| zenkai-a | clean | laplacianMae | 0.027459 | 0.025586 |
| zenkai-a | clean | hfError | 0.007090 | 0.006472 |
| zenkai-a | clean | ssim | 0.971347 | 0.974300 |
| zenkai-a | heavy | mae | 0.017381 | 0.016699 |
| zenkai-a | heavy | edgeMae | 0.013876 | 0.013207 |
| zenkai-a | heavy | laplacianMae | 0.032898 | 0.031600 |
| zenkai-a | heavy | hfError | 0.009516 | 0.009020 |
| zenkai-a | heavy | lpipsSqueeze | 0.147541 | 0.132721 |
| zenkai-a | heavy | ssim | 0.899347 | 0.904583 |
| zenkai-a | heavy | msSsim | 0.964176 | 0.965579 |
| zenkai-a | codec-intra-h264 | mae | 0.021404 | 0.020943 |
| zenkai-a | codec-intra-h264 | edgeMae | 0.014529 | 0.013996 |
| zenkai-a | codec-intra-h264 | laplacianMae | 0.033467 | 0.032511 |
| zenkai-a | codec-intra-h264 | hfError | 0.009735 | 0.009345 |
| zenkai-a | codec-intra-h264 | lpipsSqueeze | 0.165340 | 0.155418 |
| zenkai-a | codec-intra-h264 | ssim | 0.869820 | 0.874131 |
| zenkai-a | codec-intra-h264 | msSsim | 0.944066 | 0.945184 |
| zenkai-a | codec-intra-h265 | mae | 0.020918 | 0.020191 |
| zenkai-a | codec-intra-h265 | edgeMae | 0.014781 | 0.013987 |
| zenkai-a | codec-intra-h265 | laplacianMae | 0.033496 | 0.031991 |
| zenkai-a | codec-intra-h265 | hfError | 0.009822 | 0.009247 |
| zenkai-a | codec-intra-h265 | lpipsSqueeze | 0.199729 | 0.177642 |
| zenkai-a | codec-intra-h265 | ssim | 0.863520 | 0.870957 |
| zenkai-a | codec-intra-h265 | msSsim | 0.942436 | 0.944785 |
| zenkai-a | codec-intra-av1 | mae | 0.015497 | 0.014837 |
| zenkai-a | codec-intra-av1 | edgeMae | 0.012362 | 0.011620 |
| zenkai-a | codec-intra-av1 | laplacianMae | 0.030441 | 0.028786 |
| zenkai-a | codec-intra-av1 | hfError | 0.008464 | 0.007884 |
| zenkai-a | codec-intra-av1 | lpipsSqueeze | 0.107739 | 0.102919 |
| zenkai-a | codec-intra-av1 | ssim | 0.919183 | 0.923785 |
| zenkai-a | codec-intra-av1 | msSsim | 0.971465 | 0.972266 |
| zenkai-b | clean | mae | 0.008643 | 0.008381 |
| zenkai-b | clean | edgeMae | 0.009510 | 0.009114 |
| zenkai-b | clean | laplacianMae | 0.026652 | 0.025586 |
| zenkai-b | clean | hfError | 0.006798 | 0.006472 |
| zenkai-b | clean | ssim | 0.972868 | 0.974300 |
| zenkai-b | heavy | mae | 0.017158 | 0.016699 |
| zenkai-b | heavy | edgeMae | 0.013630 | 0.013207 |
| zenkai-b | heavy | laplacianMae | 0.032336 | 0.031600 |
| zenkai-b | heavy | hfError | 0.009311 | 0.009020 |
| zenkai-b | heavy | lpipsSqueeze | 0.143297 | 0.132721 |
| zenkai-b | heavy | ssim | 0.900760 | 0.904583 |
| zenkai-b | heavy | msSsim | 0.964310 | 0.965579 |
| zenkai-b | codec-intra-h264 | mae | 0.021259 | 0.020943 |
| zenkai-b | codec-intra-h264 | edgeMae | 0.014318 | 0.013996 |
| zenkai-b | codec-intra-h264 | laplacianMae | 0.033001 | 0.032511 |
| zenkai-b | codec-intra-h264 | hfError | 0.009563 | 0.009345 |
| zenkai-b | codec-intra-h264 | lpipsSqueeze | 0.161664 | 0.155418 |
| zenkai-b | codec-intra-h264 | ssim | 0.870818 | 0.874131 |
| zenkai-b | codec-intra-h264 | msSsim | 0.944116 | 0.945184 |
| zenkai-b | codec-intra-h265 | mae | 0.020715 | 0.020191 |
| zenkai-b | codec-intra-h265 | edgeMae | 0.014516 | 0.013987 |
| zenkai-b | codec-intra-h265 | laplacianMae | 0.032886 | 0.031991 |
| zenkai-b | codec-intra-h265 | hfError | 0.009602 | 0.009247 |
| zenkai-b | codec-intra-h265 | lpipsSqueeze | 0.194472 | 0.177642 |
| zenkai-b | codec-intra-h265 | ssim | 0.865043 | 0.870957 |
| zenkai-b | codec-intra-h265 | msSsim | 0.942650 | 0.944785 |
| zenkai-b | codec-intra-av1 | mae | 0.015283 | 0.014837 |
| zenkai-b | codec-intra-av1 | edgeMae | 0.012082 | 0.011620 |
| zenkai-b | codec-intra-av1 | laplacianMae | 0.029753 | 0.028786 |
| zenkai-b | codec-intra-av1 | hfError | 0.008222 | 0.007884 |
| zenkai-b | codec-intra-av1 | lpipsSqueeze | 0.105025 | 0.102919 |
| zenkai-b | codec-intra-av1 | ssim | 0.920339 | 0.923785 |
| zenkai-b | codec-intra-av1 | msSsim | 0.971478 | 0.972266 |
| realesr-anime | clean | mae | 0.016004 | 0.008381 |
| realesr-anime | clean | edgeMae | 0.012151 | 0.009114 |
| realesr-anime | clean | laplacianMae | 0.031744 | 0.025586 |
| realesr-anime | clean | hfError | 0.008340 | 0.006472 |
| realesr-anime | clean | lpipsSqueeze | 0.066711 | 0.044333 |
| realesr-anime | clean | ssim | 0.944273 | 0.974300 |
| realesr-anime | clean | msSsim | 0.983434 | 0.996280 |
| realesr-anime | heavy | mae | 0.018893 | 0.016699 |
| realesr-anime | heavy | edgeMae | 0.013809 | 0.013207 |
| realesr-anime | heavy | laplacianMae | 0.033628 | 0.031600 |
| realesr-anime | heavy | hfError | 0.009374 | 0.009020 |
| realesr-anime | heavy | msSsim | 0.965287 | 0.965579 |
| realesr-anime | codec-intra-h264 | mae | 0.021242 | 0.020943 |
| realesr-anime | codec-intra-h264 | edgeMae | 0.014248 | 0.013996 |
| realesr-anime | codec-intra-h264 | laplacianMae | 0.034120 | 0.032511 |
| realesr-anime | codec-intra-h264 | hfError | 0.009562 | 0.009345 |
| realesr-anime | codec-intra-h265 | edgeMae | 0.014076 | 0.013987 |
| realesr-anime | codec-intra-h265 | laplacianMae | 0.033479 | 0.031991 |
| realesr-anime | codec-intra-h265 | hfError | 0.009388 | 0.009247 |
| realesr-anime | codec-intra-av1 | mae | 0.017554 | 0.014837 |
| realesr-anime | codec-intra-av1 | edgeMae | 0.013200 | 0.011620 |
| realesr-anime | codec-intra-av1 | laplacianMae | 0.032975 | 0.028786 |
| realesr-anime | codec-intra-av1 | hfError | 0.008963 | 0.007884 |
| realesr-anime | codec-intra-av1 | ssim | 0.913404 | 0.923785 |
| realesr-anime | codec-intra-av1 | msSsim | 0.967868 | 0.972266 |
| hfa2k-cugan | clean | mae | 0.021942 | 0.008381 |
| hfa2k-cugan | clean | edgeMae | 0.014855 | 0.009114 |
| hfa2k-cugan | clean | laplacianMae | 0.039669 | 0.025586 |
| hfa2k-cugan | clean | hfError | 0.009928 | 0.006472 |
| hfa2k-cugan | clean | lpipsSqueeze | 0.079269 | 0.044333 |
| hfa2k-cugan | clean | ssim | 0.899134 | 0.974300 |
| hfa2k-cugan | clean | msSsim | 0.955276 | 0.996280 |
| hfa2k-cugan | heavy | mae | 0.027024 | 0.016699 |
| hfa2k-cugan | heavy | edgeMae | 0.017136 | 0.013207 |
| hfa2k-cugan | heavy | laplacianMae | 0.043856 | 0.031600 |
| hfa2k-cugan | heavy | hfError | 0.011478 | 0.009020 |
| hfa2k-cugan | heavy | ssim | 0.858774 | 0.904583 |
| hfa2k-cugan | heavy | msSsim | 0.930787 | 0.965579 |
| hfa2k-cugan | codec-intra-h264 | mae | 0.030122 | 0.020943 |
| hfa2k-cugan | codec-intra-h264 | edgeMae | 0.017568 | 0.013996 |
| hfa2k-cugan | codec-intra-h264 | laplacianMae | 0.044107 | 0.032511 |
| hfa2k-cugan | codec-intra-h264 | hfError | 0.011682 | 0.009345 |
| hfa2k-cugan | codec-intra-h264 | ssim | 0.830052 | 0.874131 |
| hfa2k-cugan | codec-intra-h264 | msSsim | 0.912054 | 0.945184 |
| hfa2k-cugan | codec-intra-h265 | mae | 0.033077 | 0.020191 |
| hfa2k-cugan | codec-intra-h265 | edgeMae | 0.018311 | 0.013987 |
| hfa2k-cugan | codec-intra-h265 | laplacianMae | 0.045820 | 0.031991 |
| hfa2k-cugan | codec-intra-h265 | hfError | 0.012149 | 0.009247 |
| hfa2k-cugan | codec-intra-h265 | ssim | 0.830174 | 0.870957 |
| hfa2k-cugan | codec-intra-h265 | msSsim | 0.911241 | 0.944785 |
| hfa2k-cugan | codec-intra-av1 | mae | 0.029843 | 0.014837 |
| hfa2k-cugan | codec-intra-av1 | edgeMae | 0.017242 | 0.011620 |
| hfa2k-cugan | codec-intra-av1 | laplacianMae | 0.045315 | 0.028786 |
| hfa2k-cugan | codec-intra-av1 | hfError | 0.011646 | 0.007884 |
| hfa2k-cugan | codec-intra-av1 | lpipsSqueeze | 0.117680 | 0.102919 |
| hfa2k-cugan | codec-intra-av1 | ssim | 0.864245 | 0.923785 |
| hfa2k-cugan | codec-intra-av1 | msSsim | 0.930724 | 0.972266 |

REAL VIDEO VALIDATION NOT EXECUTED.
