"""Build an FP16 inference graph, retaining float32 I/O for the WebGPU shaders.

Requires onnx==1.20.1 and onnxruntime==1.23.2 (conversion tools only).
Run npm run setup:animejanai first to prepare the corrected FP32 source.
"""
from pathlib import Path
import hashlib
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel

root = Path(__file__).resolve().parents[1]
source = root / 'models/2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu.onnx'
target = source.with_name(source.stem + '_fp16.onnx')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '3eca0dee498cc0bfb311a6ed97e23bf225f7660de033ca9683ffefd155c0b164'
model = convert_float_to_float16(onnx.load(source), keep_io_types=True)
OnnxModel(model).topological_sort()
onnx.checker.check_model(model)
onnx.save(model, target)
print(f'{target.name}: {target.stat().st_size} bytes; SHA-256 {hashlib.sha256(target.read_bytes()).hexdigest()}')
