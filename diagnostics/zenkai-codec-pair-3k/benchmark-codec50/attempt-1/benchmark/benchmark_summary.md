# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-ablation-skip-factor-codec50-3k-continuation | 640x360 | 1280x720 | 18.04 | 18.34 |
| zenkai-ablation-skip-factor-codec50-3k-continuation | 1280x720 | 1920x1080 | 66.88 | 67.32 |
