# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-ablation-skip-v2-dataset-1k | 640x360 | 1280x720 | 18.25 | 18.71 |
| zenkai-ablation-skip-v2-dataset-1k | 1280x720 | 1920x1080 | 67.23 | 67.66 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.40 | 7.75 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 25.66 | 25.93 |
