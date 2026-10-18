"""Export trained Lite weights for optional TensorRT runtime."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from lite_model import AdoreLite

root = Path(__file__).resolve().parents[1]
path = root / 'models/adore-lite-experimental.npz'
report = json.loads(path.with_suffix('.json').read_text())
if hashlib.sha256(path.read_bytes()).hexdigest() != report['weightsSha256']:
    raise ValueError('Lite checksum mismatch')
model = AdoreLite().eval()
with np.load(path, allow_pickle=False) as weights:
    model.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files})
torch.onnx.export(model, torch.zeros(1, 3, 40, 40), str(path.with_suffix('.onnx')),
                  input_names=['input'], output_names=['output'], opset_version=17,
                  dynamic_axes={'input': {2:'height',3:'width'}, 'output': {2:'out_height',3:'out_width'}}, dynamo=False)
print(path.with_suffix('.onnx'))
