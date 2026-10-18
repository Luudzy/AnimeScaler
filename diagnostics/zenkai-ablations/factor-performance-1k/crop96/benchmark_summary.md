# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-ablation-skip-factor-crop96-1k | 640x360 | 1280x720 | 18.65 | 19.67 |
| zenkai-ablation-skip-factor-crop96-1k | 1280x720 | 1920x1080 | 66.13 | 67.41 |
