"""Portable ONNX backend benchmark. Never installs drivers or silently substitutes CPU."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROVIDERS = {
    'cpu': 'CPUExecutionProvider', 'rocm': 'ROCMExecutionProvider',
    'migraphx': 'MIGraphXExecutionProvider', 'directml': 'DmlExecutionProvider',
}


def statistics(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {'medianMs': float(np.median(values)), 'p95Ms': float(np.percentile(values, 95)),
            'estimatedInferenceFps': float(1000 / np.median(values))}


def compare(reference, output):
    if output.shape != reference.shape or not np.isfinite(output).all():
        return {'passed': False, 'reason': 'shape mismatch or nonfinite output'}
    delta = np.abs(output.astype(np.float32) - reference)
    mean, maximum = float(delta.mean()), float(delta.max())
    return {'passed': mean <= .005 and maximum <= .05, 'meanAbsoluteError': mean, 'maxAbsoluteError': maximum}


def execute(path, data, provider, warmup, frames):
    if provider not in ort.get_available_providers():
        return {'status': 'unavailable', 'reason': f'{provider} is not installed'}, None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if provider == 'DmlExecutionProvider':
        options.enable_mem_pattern = False
    if provider != 'CPUExecutionProvider':
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    session.disable_fallback()
    if provider not in session.get_providers():
        raise RuntimeError(f'Requested provider {provider} was not activated')
    input_meta = session.get_inputs()[0]
    dtype = np.float16 if input_meta.type == 'tensor(float16)' else np.float32
    feeds = {input_meta.name: np.ascontiguousarray(data, dtype=dtype)}
    times = []
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    output = None
    for index in range(warmup + frames):
        started = time.perf_counter()
        # Returning numpy output synchronizes completion; includes host/device copies.
        output = session.run(None, feeds)[0]
        if index >= warmup:
            times.append((time.perf_counter() - started) * 1000)
    wall = time.perf_counter() - wall_started
    return {'status': 'measured', 'provider': provider, 'activeProviders': session.get_providers(),
            **statistics(times), 'samples': len(times), 'cpuCoreEquivalentPercent':
            (time.process_time() - cpu_started) / wall * 100,
            'gpuUtilizationPercent': None, 'vramBytes': None,
            'scope': 'session.run including CPU input and output transfers; no decode, render or tiling; CPU utilization includes warmup'}, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--model', choices=['adore', 'animejanai'], default='adore')
    parser.add_argument('--output', type=Path, default=Path('diagnostics/native-results.json'))
    args = parser.parse_args()
    if min(args.width, args.height, args.frames) <= 0 or args.warmup < 0 or args.width % 4 or args.height % 4:
        parser.error('dimensions must be positive multiples of four; frames > 0 and warmup >= 0')
    root = Path(__file__).resolve().parents[1]
    stem = '2x_Adore_renarchi_webgpu' if args.model == 'adore' else '2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu'
    models = {'fp32': root / f'models/{stem}.onnx', 'fp16': root / f'models/{stem}_fp16.onnx'}
    # Fixed integer pattern, identical across providers and portable across platforms.
    y, x = np.indices((args.height, args.width))
    data = np.stack([((x * 7 + y * 3 + c * 47) % 256) / 255 for c in range(3)])[None].astype(np.float32)
    report = {'schemaVersion': 1, 'platform': platform.platform(), 'ortVersion': ort.__version__,
              'availableProviders': ort.get_available_providers(), 'model': args.model,
              'inputShape': list(data.shape), 'inputSha256': hashlib.sha256(data.tobytes()).hexdigest(),
              'warmup': args.warmup, 'frames': args.frames, 'results': []}
    try:
        import torch
        report['torch'] = {'version': torch.__version__, 'hip': torch.version.hip,
                           'status': 'not-benchmarked',
                           'reason': 'Needs original PyTorch architecture and matching weights; ONNX is not a PyTorch checkpoint'}
    except ImportError:
        report['torch'] = {'status': 'unavailable'}
    report['webgpu'] = {'status': 'separate-browser-harness',
                        'reason': 'Use tuning.html integer-grid-v1 and match inputSha256; browser includes preprocessing and rendering, so timing scopes still differ'}
    reference = None
    for precision, path in models.items():
        for backend, provider in PROVIDERS.items():
            entry = {'backend': backend, 'precision': precision}
            try:
                entry['modelSha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                measurement, output = execute(path, data, provider, args.warmup, args.frames)
                entry.update(measurement)
                if precision == 'fp32' and backend == 'cpu':
                    reference = output
                if output is not None:
                    entry['quality'] = compare(reference, output) if reference is not None else {'passed': False, 'reason': 'no CPU reference'}
            except Exception as error:
                entry.update(status='failed', reason=str(error))
            report['results'].append(entry)
            print(f"{backend} {precision}: {entry['status']}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
