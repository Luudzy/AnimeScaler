"""Convert only the pinned bundled model and verify against ONNX Runtime CPU."""
from pathlib import Path
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort
import torch
from animejanai_model import AnimeJaNai

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'models/2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu.onnx'
    if hashlib.sha256(source.read_bytes()).hexdigest() != '3eca0dee498cc0bfb311a6ed97e23bf225f7660de033ca9683ffefd155c0b164':
        raise ValueError('Unknown ONNX topology; conversion requires review')
    graph = onnx.load(source).graph
    weights = {t.name: onnx.numpy_helper.to_array(t).copy() for t in graph.initializer}
    state = {}
    index = 0
    for node in graph.node:
        if node.op_type == 'Conv':
            state[f'body.{index}.weight'] = weights[node.input[1]]
            state[f'body.{index}.bias'] = weights[node.input[2]]
            index += 1
        elif node.op_type == 'PRelu':
            state[f'body.{index}.weight'] = weights[node.input[1]].reshape(-1)
            index += 1
    model = AnimeJaNai().eval()
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    session = ort.InferenceSession(str(source), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(42)
    errors = []
    with torch.inference_mode():
        for h, w in [(40, 40), (48, 80), (72, 128)]:
            x = rng.random((1, 3, h, w), dtype=np.float32)
            expected = session.run(None, {session.get_inputs()[0].name: x})[0]
            actual = model(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            errors.append(float(np.max(np.abs(expected - actual))))
    path = ROOT / 'models/animejanai-native.npz'
    np.savez_compressed(path, **state)
    report = {'architecture': 'AnimeJaNai24-v1', 'weightsSha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'maxAbsoluteErrors': errors}
    path.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
