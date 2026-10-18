# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| adore-lite-v3 | 640x360 | 1280x720 | 7.71 | 8.22 |
| adore-lite-v3 | 1280x720 | 1920x1080 | 26.82 | 27.23 |
| adore-lite-v3 | 1280x808 | 3424x2160 | 30.46 | 30.83 |
