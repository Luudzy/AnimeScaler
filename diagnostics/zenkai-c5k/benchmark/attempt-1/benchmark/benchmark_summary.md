# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-lite-v1-c-codec50-5k | 640x360 | 1280x720 | 18.23 | 18.98 |
| zenkai-lite-v1-c-codec50-5k | 1280x720 | 1920x1080 | 67.28 | 67.99 |
| adore-lite-v3 | 640x360 | 1280x720 | 7.59 | 8.50 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 25.80 | 26.76 |
