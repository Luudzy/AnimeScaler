# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| adore-v3-retrained-control | 640x360 | 1280x720 | 12.46 | 12.98 |
| adore-v3-retrained-control | 1280x720 | 1920x1080 | 37.26 | 42.35 |
