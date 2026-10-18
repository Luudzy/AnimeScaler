# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-ablation-spab-identity-skip-1k | 640x360 | 1280x720 | 31.97 | 35.63 |
| zenkai-ablation-spab-identity-skip-1k | 1280x720 | 1920x1080 | 66.43 | 112.75 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.57 | 8.02 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 25.49 | 25.86 |
