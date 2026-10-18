# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-ablation-skip-v2-domain10-1k | 640x360 | 1280x720 | 19.08 | 19.85 |
| zenkai-ablation-skip-v2-domain10-1k | 1280x720 | 1920x1080 | 68.41 | 79.27 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.54 | 8.15 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 26.08 | 26.53 |
