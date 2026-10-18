"""Actual versioned-registry parity; optional GPU renderer check, no promotion."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'native'))
import numpy as np
import torch
from export_zenkai import ROOT, TOLERANCES, configure_cache, parity_inputs, parity_metrics, require_parity
from export_zenkai_skip import load_candidate
from model_registry import build_registry
from zenkai_1_0 import MODEL_ID, PACKAGE_RELATIVE, SOURCE_MASTER_SHA, SOURCE_RUN, digest, validate_package


def renderer_cases():
    # Bridge requires minimum40 and input dimensions divisible by four.
    # Odd/tiny dimensions remain in direct model parity, outside the bridge.
    return ((44, 40, 88, 80), (64, 48, 128, 96), (96, 64, 144, 96))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('New workspace validation report required')
    if args.device == 'cuda' and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError('ROCm GPU required; no CPU fallback')
    configure_cache()
    torch.set_num_threads(2)
    package = ROOT / PACKAGE_RELATIVE
    validate_package(package)
    master = ROOT / 'models/zenkai/experiments' / SOURCE_RUN / 'step-5000-master-fp32.pt'
    if digest(master) != SOURCE_MASTER_SHA:
        raise ValueError('C5k source changed')
    reference, _ = load_candidate(master)
    reference = reference.to(args.device).eval()
    registry = build_registry(ROOT)
    result = {'state': 'validating', 'model': MODEL_ID, 'sourceCheckpointSha256': SOURCE_MASTER_SHA,
              'packageManifestSha256': digest(package / 'manifest.json'), 'device': args.device,
              'torch': str(torch.__version__), 'hip': torch.version.hip,
              'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
              'scope': 'New exact versioned registry load versus original C5k training master; no optimizer updates',
              'cases': [], 'renderer': 'NOT EXECUTED ON CPU', 'newTrainingExecuted': False,
              'TensorRT': 'NOT MEASURED — NVIDIA hardware unavailable', 'releaseApproved': False}
    inputs = list(parity_inputs())
    inputs.append(('odd-23x31', torch.rand(1, 3, 23, 31, generator=torch.Generator().manual_seed(929))))
    expected = {}
    with torch.inference_mode():
        for precision, key in (('fp32', 'registeredFp32'), ('fp16', 'registeredFp16')):
            runtime = registry.load(MODEL_ID, device=args.device, precision=precision)
            for index, (name, value) in enumerate(inputs):
                x = value.to(args.device)
                if precision == 'fp32':
                    expected[name] = reference(x).float().cpu()
                    result['cases'].append({'name': name, 'shape': list(value.shape)})
                actual = runtime.infer(x.half() if precision == 'fp16' else x)
                scores = parity_metrics(expected[name], actual)
                require_parity(scores, **TOLERANCES[precision])
                result['cases'][index][key] = scores
            registry.unload(MODEL_ID)
    del reference
    if args.device == 'cuda':
        from server import Renderer
        from protocol import HEADER, MAGIC
        renderer = Renderer()
        runtime = registry.load(MODEL_ID, device='cuda', precision='fp16')
        renderer_rows = []
        for index, (w, h, ow, oh) in enumerate(renderer_cases()):
            pixels = np.random.default_rng(929 + index).integers(0, 256, (h, w, 4), dtype=np.uint8)
            pixels[..., 3] = 255
            data = HEADER.pack(MAGIC, index + 1, w, h, ow, oh) + pixels.tobytes()
            body, measured_ms = renderer.render(data, model_name=MODEL_ID, transport='rgba', allow_repeat=False)
            if HEADER.unpack_from(body) != (MAGIC, index + 1, ow, oh, ow, oh):
                raise ValueError('Renderer frame identity/shape changed')
            actual = np.frombuffer(body, dtype=np.uint8, offset=HEADER.size).reshape(oh, ow, 4)
            with torch.inference_mode():
                x = torch.from_numpy(pixels[..., :3].copy()).permute(2, 0, 1)[None].cuda().half() / 255
                prediction = runtime.infer(x)
                if prediction.shape[-2:] != (oh, ow):
                    prediction = torch.nn.functional.interpolate(prediction, size=(oh, ow), mode='bilinear', align_corners=False)
                expected_rgb = prediction[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
            error = np.abs(actual[..., :3].astype(np.int16) - expected_rgb.astype(np.int16))
            if error.max() > 1 or not np.all(actual[..., 3] == 255) or renderer.last_reused:
                raise ValueError('Renderer conversion/reuse parity failed')
            renderer_rows.append({'input': [w, h], 'output': [ow, oh], 'maxQuantizedErrorLsb': int(error.max()),
                                  'meanQuantizedErrorLsb': float(error.mean()), 'frameReused': renderer.last_reused,
                                  'oneFrameRenderMs': measured_ms, 'scope': 'Parity smoke timing, not a player FPS benchmark'})
        if renderer.graph is not None:
            raise ValueError('Unmeasured graph capture must remain disabled for this candidate')
        for name in list(renderer.models):
            renderer.registry.unload(name)
        registry.unload(MODEL_ID)
        result['renderer'] = {'state': 'PASS', 'cases': renderer_rows,
                              'scope': 'Actual ROCm Renderer RGBA path; not browser/socket/video playback certification'}
    if digest(master) != SOURCE_MASTER_SHA:
        raise ValueError('Runtime verification changed source master')
    validate_package(package)
    result['state'] = 'PASS'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'state': 'PASS', 'model': MODEL_ID, 'device': args.device, 'output': str(output)}))


if __name__ == '__main__':
    main()
