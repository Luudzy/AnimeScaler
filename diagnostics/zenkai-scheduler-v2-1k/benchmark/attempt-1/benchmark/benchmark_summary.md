# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-ablation-skip-factor-scheduler-constant-v2-1k | 640x360 | 1280x720 | 19.24 | 20.35 |
| zenkai-ablation-skip-factor-scheduler-constant-v2-1k | 1280x720 | 1920x1080 | 71.20 | 71.85 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.95 | 8.59 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 28.39 | 42.26 |
