# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-a | 640x360 | 1280x720 | 16.86 | 17.21 |
| zenkai-a | 1280x720 | 1920x1080 | 61.05 | 61.45 |
| zenkai-b | 640x360 | 1280x720 | 19.54 | 19.87 |
| zenkai-b | 1280x720 | 1920x1080 | 70.78 | 73.16 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.32 | 7.66 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 25.15 | 25.45 |
| realesr-anime | 640x360 | 1280x720 | 31.13 | 31.44 |
| realesr-anime | 1280x720 | 1920x1080 | 119.36 | 119.72 |
| hfa2k-cugan | 640x360 | 1280x720 | 63.45 | 63.85 |
| hfa2k-cugan | 1280x720 | 1920x1080 | 237.14 | 237.64 |
