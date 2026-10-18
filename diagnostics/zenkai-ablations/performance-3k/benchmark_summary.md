# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-frequency-3k | 640x360 | 1280x720 | 17.83 | 18.14 |
| zenkai-frequency-3k | 1280x720 | 1920x1080 | 65.85 | 66.36 |
| zenkai-ablation-spab-identity-skip-3k-continuation | 640x360 | 1280x720 | 19.10 | 19.42 |
| zenkai-ablation-spab-identity-skip-3k-continuation | 1280x720 | 1920x1080 | 70.98 | 71.45 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.70 | 8.04 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 27.00 | 27.32 |
