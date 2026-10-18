"""Bounded persistent official FSR3 D3D12 worker. No Torch/ROCm dependency."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

from backend_preflight import resolve_fsr_sdk_path
from protocol import HEADER, MAGIC, decode_frame

MODEL = 'fsr3-native'
REQUEST = struct.Struct('<8I')


class Fsr3Worker:
    def __init__(self, root, sdk=None):
        root = Path(root)
        sdk = resolve_fsr_sdk_path(sdk)
        self.exe = root / 'native/fsr3-probe/build/fsr3-worker.exe'
        self.dll = sdk / 'Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll'
        if not self.exe.is_file() or not self.dll.is_file():
            raise RuntimeError('Compile native/fsr3-probe/build.ps1 and configure the official FSR SDK path')
        log_dir = root / 'diagnostics/fsr3-runtime'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (log_dir / f'worker-{time.time_ns()}.stderr.txt').open('wb')
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fsr3-pipe')
        self.process = subprocess.Popen([str(self.exe), str(self.dll)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            line = self.io.submit(self.process.stdout.readline).result(timeout=15)
            self.info = json.loads(line)
            if self.info.get('backend') != 'd3d12' or not str(self.info.get('provider', '')).startswith('3.1.'):
                raise RuntimeError('Official FSR3.1 provider not confirmed')
        except Exception:
            self.close()
            raise

    @staticmethod
    def read_exact(stream, length):
        body = bytearray()
        while len(body) < length:
            chunk = stream.read(length - len(body))
            if not chunk:
                raise ConnectionError('FSR3 worker stopped; inspect diagnostics/fsr3-runtime')
            body.extend(chunk)
        return bytes(body)

    def _exchange(self, data, reset, frame_delta_ms):
        frame_id, width, height, ow, oh = decode_frame(data)
        if ow < width or oh < height:
            raise ValueError('FSR3 input cannot exceed output dimensions')
        request = REQUEST.pack(MAGIC, frame_id, width, height, ow, oh, int(reset), max(1000,min(250000,round(frame_delta_ms*1000)))) + data[HEADER.size:]
        # Raw pipes may perform short writes; never truncate a frame silently.
        view = memoryview(request)
        while view:
            written = self.process.stdin.write(view)
            if not written:
                raise ConnectionError('FSR3 input pipe closed')
            view = view[written:]
        self.process.stdin.flush()
        header = self.read_exact(self.process.stdout, HEADER.size)
        expected = (MAGIC, frame_id, ow, oh, ow, oh)
        if HEADER.unpack(header) != expected:
            raise ConnectionError('FSR3 response identity or dimensions mismatch')
        return header + self.read_exact(self.process.stdout, ow * oh * 4)

    def infer(self, data, reset=False, frame_delta_ms=1000/24):
        decode_frame(data)
        if self.process is None or self.process.poll() is not None:
            raise ConnectionError('FSR3 worker unavailable; restart Iniciar-FSR3.cmd')
        future = self.io.submit(self._exchange, data, reset, frame_delta_ms)
        try:
            return future.result(timeout=20)
        except Exception:
            # Fresh service restart is explicit; never create a crash loop.
            self.close()
            raise

    def close(self):
        process, self.process = getattr(self, 'process', None), None
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
        if hasattr(self, 'io'):
            self.io.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, 'log'):
            self.log.close()


class Fsr3Renderer:
    backend = 'd3d12'

    def __init__(self, root):
        self.worker = Fsr3Worker(root)
        self.info = {**self.worker.info, 'protocol': 1, 'model': MODEL,
                     'precision': 'native', 'architecture': 'DirectX 12',
                     'models': [MODEL], 'scales': {MODEL: 1},
                     'revisions': {MODEL: hashlib.sha256(self.worker.dll.read_bytes()).hexdigest()},
                     'mode': 'estimated-video-motion',
                     'metadata': {MODEL: {'name': 'AMD FSR 3.1 experimental video',
                                         'status': 'experimental', 'provider': self.worker.info['provider'],
                                         'native_scale': 1, 'family': 'FSR3', 'camera_jitter': 'none-decoded-video',
                                         'depth': 'constant-plane', 'motion': 'GPU-estimated',
                                         'frame_generation': False}}}
        self.shape = None
        self.history_reset_pending = True
        self.last_reused = False
        self.frame_revision = 0
        self.frame_delta_ms = 1000/24

    def reset_history(self):
        self.history_reset_pending = True

    def frame_context(self, reset, delta_ms):
        self.history_reset_pending = self.history_reset_pending or reset
        self.frame_delta_ms = delta_ms

    def render(self, data, model_name=MODEL, transport='rgba', allow_repeat=False):
        if model_name != MODEL or transport != 'rgba':
            raise ValueError('FSR3 service supports only its own model and RGBA transport')
        _, width, height, ow, oh = decode_frame(data)
        self.last_warmup = self.shape != (width, height, ow, oh)
        started = time.perf_counter()
        body = self.worker.infer(data, reset=self.history_reset_pending, frame_delta_ms=self.frame_delta_ms)
        self.history_reset_pending = False
        self.shape = (width, height, ow, oh)
        self.frame_revision += 1
        return body, (time.perf_counter() - started) * 1000

    def close(self):
        self.worker.close()
