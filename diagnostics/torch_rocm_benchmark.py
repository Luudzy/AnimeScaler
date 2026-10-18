"""Run using the user's existing ROCm Python. No dependency installation or ComfyUI changes."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np

_cache = Path(__file__).resolve().parent / 'native-cache'
for _name, _folder in [('MIOPEN_USER_DB_PATH', 'db'), ('MIOPEN_CUSTOM_CACHE_DIR', 'kernels')]:
    (_cache / _folder).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_name, str(_cache / _folder))

import torch
from spandrel.architectures.RealCUGAN.__arch.upcunet_v3 import UpCunet2x_fast

parser = argparse.ArgumentParser()
parser.add_argument('--frames', type=int, default=10)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--output', type=Path, default=Path('diagnostics/torch-rocm-adore-results.json'))
args = parser.parse_args()
if args.frames < 1 or args.warmup < 0:
    parser.error('frames > 0 and warmup >= 0 required')
if not torch.version.hip or not torch.cuda.is_available():
    raise RuntimeError('A working HIP/ROCm PyTorch GPU is required; CPU substitution is disabled')
root = Path(__file__).resolve().parents[1]
fixture = root / 'diagnostics/native-fixture'
data = np.load(fixture / 'input.npy', allow_pickle=False)
reference = np.load(fixture / 'reference.npy', allow_pickle=False)
metadata = json.loads((fixture / 'metadata.json').read_text())
assert metadata['inputSha256'] == hashlib.sha256(data.tobytes()).hexdigest()
if 'weightsSha256' in metadata:
    assert metadata['weightsSha256'] == hashlib.sha256((fixture / 'adore-weights.npz').read_bytes()).hexdigest()
weights = np.load(fixture / 'adore-weights.npz', allow_pickle=False)
state = {name: torch.from_numpy(weights[name].copy()) for name in weights.files}
properties = torch.cuda.get_device_properties(0)
report = {**metadata, 'torch': torch.__version__, 'hip': torch.version.hip,
          'gpu': properties.name, 'architecture': properties.gcnArchName,
          'warmup': args.warmup, 'frames': args.frames, 'results': [],
          'notes': 'Same Adore weights and input as ONNX CPU fixture; whole frame, no tiles. GPU-resident inference and CPU roundtrip are separate measurements. No video decoding or rendering.'}


def summarize(values):
    return {'medianMs': float(np.median(values)), 'p95Ms': float(np.percentile(values, 95))}


for precision, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
    for layout in ['nchw', 'channels_last']:
        result = {'precision': precision, 'layout': layout}
        try:
            model = UpCunet2x_fast()
            model.load_state_dict(state, strict=True)
            model = model.eval().to(device='cuda', dtype=dtype)
            memory_format = torch.channels_last if layout == 'channels_last' else torch.contiguous_format
            model = model.to(memory_format=memory_format)
            x = torch.from_numpy(data).to(device='cuda', dtype=dtype, memory_format=memory_format)

            def infer(value):
                # Reproduce the ONNX export's pro normalization exactly.
                return (model(value.clamp(0, 1) * .7 + .15) - .15) / .7

            with torch.inference_mode():
                for _ in range(args.warmup):
                    infer(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for _ in range(args.frames):
                    started = time.perf_counter()
                    output = infer(x)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - started) * 1000)
                actual = output.float().cpu().numpy()
                valid = actual.shape == reference.shape and bool(np.isfinite(actual).all())
                if valid:
                    delta = np.abs(actual - reference)
                    mean, maximum = float(delta.mean()), float(delta.max())
                    result['quality'] = {'passed': mean <= .005 and maximum <= .05,
                                         'meanAbsoluteError': mean, 'maxAbsoluteError': maximum}
                else:
                    result['quality'] = {'passed': False, 'reason': 'shape mismatch or nonfinite output'}
                result['gpuResident'] = summarize(samples)
                result['peakTorchAllocatedBytes'] = torch.cuda.max_memory_allocated()
                result['peakTorchReservedBytes'] = torch.cuda.max_memory_reserved()
                roundtrip = []
                for _ in range(args.frames):
                    started = time.perf_counter()
                    value = torch.from_numpy(data).to(device='cuda', dtype=dtype, memory_format=memory_format)
                    infer(value).float().cpu().numpy()
                    torch.cuda.synchronize()
                    roundtrip.append((time.perf_counter() - started) * 1000)
                result['cpuRoundtrip'] = summarize(roundtrip)
            result['status'] = 'measured'
            del model, x, output
            torch.cuda.empty_cache()
        except Exception as error:
            result.update(status='failed', reason=str(error))
        report['results'].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(result), flush=True)
