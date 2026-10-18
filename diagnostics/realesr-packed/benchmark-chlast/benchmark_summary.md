# Unified baseline performance

All values MEASURED. Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.

| Model | Input | Output | Median ms | P95 ms |
|---|---|---|---:|---:|
| realesr-anime | 640x360 | 1280x720 | 33.63 | 33.85 |
| realesr-anime | 1280x720 | 1920x1080 | 127.18 | 127.76 |
| realesr-anime | 1280x808 | 3424x2160 | 154.06 | 154.34 |
