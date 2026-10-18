# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| adore-lite-v3 | 640x360 | 1280x720 | 8.23 | 8.94 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 27.85 | 28.53 |
| realesr-anime | 640x360 | 1280x720 | 37.20 | 37.97 |
| realesr-anime | 1280x720 | 1920x1080 | 142.03 | 144.08 |
| hfa2k-cugan | 640x360 | 1280x720 | 73.11 | 74.46 |
| hfa2k-cugan | 1280x720 | 1920x1080 | 274.29 | 281.18 |
