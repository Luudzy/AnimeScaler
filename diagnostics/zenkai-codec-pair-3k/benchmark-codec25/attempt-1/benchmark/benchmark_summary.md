# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-ablation-spab-identity-skip-3k-continuation | 640x360 | 1280x720 | 17.77 | 18.37 |
| zenkai-ablation-spab-identity-skip-3k-continuation | 1280x720 | 1920x1080 | 66.70 | 67.13 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.11 | 7.58 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 25.28 | 25.60 |
