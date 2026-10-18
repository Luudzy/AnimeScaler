"""Extract verified ONNX weights without pickle and create a shared numerical fixture."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=360)
args = parser.parse_args()
if min(args.width, args.height) < 40 or args.width % 4 or args.height % 4:
    parser.error('Adore requires dimensions >= 40 and multiples of four')
root = Path(__file__).resolve().parents[1]
source = root / 'models/2x_Adore_renarchi_fp32.onnx'
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == '532c1b30711c2bb7ea6ddf123fd219df87ca9816fbb8ddaaad5c1f311d5b7c44'
model = onnx.load(source)
prefix = 'Adore.renarchi.'
weights = {v.name.removeprefix(prefix): onnx.numpy_helper.to_array(v).copy()
           for v in model.graph.initializer if v.name.startswith(prefix)}
assert len(weights) > 0
dest = root / 'diagnostics/native-fixture'
dest.mkdir(exist_ok=True)
np.savez(dest / 'adore-weights.npz', **weights)
np.savez(root / 'models/adore-native.npz', **weights)
(root / 'models/adore-native.sha256').write_text(hashlib.sha256((root / 'models/adore-native.npz').read_bytes()).hexdigest() + '\n')
y, x = np.indices((args.height, args.width))
data = np.stack([((x * 7 + y * 3 + c * 47) % 256) / 255 for c in range(3)])[None].astype(np.float32)
session = ort.InferenceSession(str(source), providers=['CPUExecutionProvider'])
output = session.run(None, {session.get_inputs()[0].name: data})[0]
np.save(dest / 'input.npy', data)
np.save(dest / 'reference.npy', output)
(dest / 'metadata.json').write_text(json.dumps({'modelSha256': digest,
    'weightsSha256': hashlib.sha256((dest / 'adore-weights.npz').read_bytes()).hexdigest(),
    'inputSha256': hashlib.sha256(data.tobytes()).hexdigest(), 'shape': list(data.shape),
    'reference': f'ONNX Runtime {ort.__version__} CPU FP32'}, indent=2), encoding='utf-8')
print(dest)
