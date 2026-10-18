# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| zenkai-1-0 | 640x360 | 1280x720 | 30.79 | 35.77 |
| zenkai-1-0 | 1280x720 | 1920x1080 | 100.27 | 117.38 |
| adore-lite-v3 | 640x360 | 1280x720 | 12.30 | 13.76 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 38.60 | 41.63 |
