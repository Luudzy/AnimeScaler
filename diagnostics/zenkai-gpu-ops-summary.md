# RX 5500 XT operator screening

MEASURED: NCHW, eager ROCm; 50 warmup, 500 synchronized iterations. GPU tensors only.
These values do not predict complete browser FPS.

| Operator | FP16 median ms | FP32 median ms |
|---|---:|---:|
| conv3x3 | 2.301 | 3.820 |
| conv1x1 | 1.636 | 2.814 |
| depthwise3x3 | 18.515 | 18.223 |
| prelu | 0.696 | 1.277 |
| silu | 0.694 | 1.271 |
| add | 0.694 | 1.269 |
| mul | 0.688 | 1.271 |
| mul_add | 1.289 | 2.462 |
| nearest | 2.425 | 3.444 |
| bilinear | 2.220 | 3.477 |
| pixelshuffle | 1.268 | 1.291 |
