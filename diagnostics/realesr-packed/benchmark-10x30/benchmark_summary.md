# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| realesr-anime | 640x360 | 1280x720 | 32.85 | 33.07 |
| realesr-anime | 1280x720 | 1920x1080 | 125.53 | 125.94 |
| realesr-anime | 1280x808 | 3424x2160 | 151.71 | 151.95 |
