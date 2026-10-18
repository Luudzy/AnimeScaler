"""Historical HIP graph probe: live playback regressed; disabled by default."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'native'))
from protocol import HEADER, MAGIC


def summary(values):
    return {'meanMs': float(np.mean(values)), 'medianMs': float(np.median(values)), 'p95Ms': float(np.percentile(values, 95)), 'maxMs': float(max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--experimental-unstable-hip-graphs', action='store_true',
                        help='Explicitly enable the rejected experiment; may destabilize this PC')
    args = parser.parse_args()
    if not args.experimental_unstable_hip_graphs:
        parser.error('HIP graph probe disabled after live playback regression and reported PC crash. See docs/stability-report-20261011.md.')
    from server import Renderer
    if args.iterations < 20 or args.iterations > 1000:
        raise ValueError('Use 20–1000 measured iterations')
    if args.output.exists():
        raise FileExistsError(args.output)
    renderer = Renderer()
    name = 'zenkai-animevideo-30k'
    renderer.select_model(name)
    original = renderer.model.metadata
    width, height, ow, oh = 640, 360, 1280, 720
    rng = np.random.default_rng(929)
    frames = [rng.integers(0, 256, (height, width, 4), dtype=np.uint8) for _ in range(20)]
    for frame in frames:
        frame[..., 3] = 255
    results = {}
    outputs = {}
    for mode in ('eager', 'bounded-graph', 'eager-repeat', 'bounded-graph-repeat'):
        graph = 'graph' in mode
        renderer.graph = None; renderer.shape = None; renderer.use_graph = graph
        renderer.model.metadata = dataclasses.replace(original, capture_graph=False, capture_graph_max_pixels=width * height if graph else 0)
        times = []; wall = []; peak = 0
        renderer.torch.cuda.reset_peak_memory_stats()
        for index in range(30 + args.iterations):
            frame = frames[index % len(frames)]
            body = HEADER.pack(MAGIC, index + 1, width, height, ow, oh) + frame.tobytes()
            start = time.perf_counter()
            output, duration = renderer.render(body, name, 'rgb', False)
            if index >= 30:
                times.append(duration); wall.append((time.perf_counter() - start) * 1000)
            if index in (30, 31, 32):
                outputs.setdefault(mode, []).append(np.frombuffer(output, dtype=np.uint8, offset=HEADER.size).copy())
        peak = renderer.torch.cuda.max_memory_allocated() / 1048576
        results[mode] = {'service': summary(times), 'wall': summary(wall), 'peakAllocatedMiB': peak, 'graphActive': renderer.graph is not None, 'measuredFrames': args.iterations, 'warmup': 30}
        print(json.dumps({'mode': mode, **results[mode]}), flush=True)
    parity = {mode: max(int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)))) for a, b in zip(outputs['eager'], values)) for mode, values in outputs.items()}
    report = {'state': 'complete', 'scope': 'Actual Renderer FP16, upload/model/resize/readback/RGB serialization; no browser/socket. Same 20 distinct inputs cycled, no adjacent repeats and no frame reuse.', 'model': name, 'gpu': renderer.info['gpu'], 'torch': str(renderer.torch.__version__), 'cases': results, 'maxOutputByteDifferenceVsEager': parity}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if any(parity.values()) or not results['bounded-graph']['graphActive']:
        raise RuntimeError('Graph probe failed parity or did not activate')


if __name__ == '__main__':
    main()
