"""Prepare Adore FP16 weights/operations with FP32 GPU I/O; preserve originals."""
from pathlib import Path
import hashlib
import os
import tempfile

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16, DEFAULT_OP_BLOCK_LIST
from onnxruntime.transformers.onnx_model import OnnxModel

root = Path(__file__).resolve().parents[1]
models = root / 'models'
source = models / '2x_Adore_renarchi_fp32.onnx'
targets = {
    'fp32': models / '2x_Adore_renarchi_webgpu.onnx',
    'fp16': models / '2x_Adore_renarchi_webgpu_fp16.onnx',
}
expected = {
    'source': '532c1b30711c2bb7ea6ddf123fd219df87ca9816fbb8ddaaad5c1f311d5b7c44',
    'fp32': 'b0a76dff7816816dee2c9a19f58bb1a81dd68178547f9518b64bc0257205749d',
    'fp16': 'e503b357fe6d6ce34c3db9980cc2ed37dd1f23d4772b352c8a4f2befb151f2b5',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, digest, label):
    actual = sha256(path)
    if actual != digest:
        raise RuntimeError(f'{label} SHA-256 mismatch: {path.name}; preserved without overwrite')


def promote_without_overwrite(temporary, destination, digest):
    if destination.exists():
        require_hash(destination, digest, 'Existing Adore model')
        return
    try:
        # Same-directory hard-link creation is atomic and fails if destination exists.
        os.link(temporary, destination)
    except FileExistsError:
        require_hash(destination, digest, 'Existing Adore model')


require_hash(source, expected['source'], 'Adore FP32 source')
for name, target in targets.items():
    if target.exists():
        require_hash(target, expected[name], 'Existing Adore model')

with tempfile.TemporaryDirectory(prefix='.adore-webgpu-', dir=models) as temporary_directory:
    temporary_directory = Path(temporary_directory)
    fp32_temp = temporary_directory / targets['fp32'].name
    fp16_temp = temporary_directory / targets['fp16'].name

    model = onnx.load(source)
    # NCHW axes need separate names at 2x output for graph capture.
    for tensor in model.graph.input:
        tensor.type.tensor_type.shape.dim[2].dim_param = 'height'
        tensor.type.tensor_type.shape.dim[3].dim_param = 'width'
    for tensor in model.graph.output:
        tensor.type.tensor_type.shape.dim[2].dim_param = 'output_height'
        tensor.type.tensor_type.shape.dim[3].dim_param = 'output_width'
    onnx.checker.check_model(model)
    onnx.save(model, fp32_temp)

    # Squeeze/excitation reductions aggregate large spatial planes. Keep accumulation
    # in FP32 to avoid overflow in half-precision WebGPU reductions.
    model = convert_float_to_float16(
        model, keep_io_types=True,
        op_block_list=DEFAULT_OP_BLOCK_LIST + ['ReduceMean'],
    )
    OnnxModel(model).topological_sort()
    onnx.checker.check_model(model)
    onnx.save(model, fp16_temp)

    require_hash(fp32_temp, expected['fp32'], 'Generated Adore WebGPU FP32')
    require_hash(fp16_temp, expected['fp16'], 'Generated Adore WebGPU FP16')
    for name, target in targets.items():
        promote_without_overwrite(temporary_directory / target.name, target, expected[name])

for name, target in targets.items():
    print(f'{target.name}: {target.stat().st_size} bytes; SHA-256 {sha256(target)}')
