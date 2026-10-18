"""Persistent bridge to the native Anime4KCPP OpenCL worker."""
import os
import math
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from anime4kcpp_protocol import HEADER, pack_request, unpack_response_header


MODELS = {
    'anime4kcpp-acnet-f8b4': 'acnet-f8b4',
    'anime4kcpp-acnet-f8b8': 'acnet-f8b8',
    'anime4kcpp-acnet-f8b8-hdn': 'acnet-f8b8-hdn',
    'anime4kcpp-arnet-f8b8': 'arnet-f8b8',
    'anime4kcpp-artcnn-c4f16': 'artcnn-c4f16',
    'anime4kcpp-fsrcnnx-f8b4': 'fsrcnnx-f8b4',
}


class Anime4KCPPWorker:
    def __init__(self, root, model_id, device=0, request_timeout=120):
        if model_id not in MODELS:
            raise ValueError(f'Anime4KCPP model unavailable: {model_id}')
        root = Path(root)
        self.root = root
        self.model_id = model_id
        self.model_name = MODELS[model_id]
        self.device = int(device)
        self.request_timeout=float(request_timeout)
        if not math.isfinite(self.request_timeout) or self.request_timeout<=0:
            raise ValueError('Anime4KCPP request timeout must be finite and positive')
        self.exe = root / 'third_party' / 'Anime4KCPP-master' / 'worker' / 'build' / 'bin' / 'Release' / 'anime4kcpp_worker.exe'
        self.opencl_dir = root / 'third_party' / 'Anime4KCPP-master' / 'build-opencl-md' / 'bin' / 'Release'
        if not self.exe.is_file():
            raise RuntimeError(f'Anime4KCPP worker not built: {self.exe}')
        self.process = None
        self.last_shape = None
        self.start()

    def start(self):
        self.close()
        env = os.environ.copy()
        env['PATH'] = str(self.opencl_dir) + os.pathsep + env.get('PATH', '')
        self.process = subprocess.Popen(
            [str(self.exe), self.model_name, str(self.device)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, env=env, bufsize=0,
        )

    @staticmethod
    def _read_exact(stream, size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = stream.read(size - len(chunks))
            if not chunk:
                raise ConnectionError('Anime4KCPP worker closed its output')
            chunks.extend(chunk)
        return bytes(chunks)

    def _run_with_timeout(self, operation):
        completed=threading.Event();result=[];failure=[]
        def run():
            try: result.append(operation())
            except BaseException as error: failure.append(error)
            finally: completed.set()
        threading.Thread(target=run,name='anime4kcpp-pipe-io',daemon=True).start()
        if not completed.wait(self.request_timeout):
            raise TimeoutError('Anime4KCPP worker timed out')
        if failure: raise failure[0]
        return result[0]

    def infer(self, rgba, frame_id, output_size):
        value = np.asarray(rgba, dtype=np.uint8)
        if value.ndim != 3 or value.shape[2] != 4:
            raise ValueError('Anime4KCPP expects an HWC RGBA frame')
        height, width = value.shape[:2]
        output_width, output_height = output_size
        packet = pack_request(frame_id, width, height, output_width, output_height)
        started = time.perf_counter()
        process=self.process
        try:
            if self.process is None or self.process.poll() is not None:
                self.start()
            process=self.process
            def exchange():
                process.stdin.write(packet)
                process.stdin.write(value.tobytes(order='C'))
                process.stdin.flush()
                response=self._read_exact(process.stdout,HEADER.size)
                _,response_width,response_height=unpack_response_header(
                    response,expected_frame_id=frame_id,expected_input_size=(width,height),
                    expected_output_size=output_size)
                payload=self._read_exact(process.stdout,response_width*response_height*4)
                return payload,response_width,response_height
            payload,response_width,response_height=self._run_with_timeout(exchange)
        except TimeoutError:
            if self.process is process: self.close()
            raise TimeoutError('Anime4KCPP worker timed out; process terminated') from None
        except ValueError as error:
            if self.process is process: self.close()
            raise ConnectionError('Anime4KCPP worker returned an invalid header; process stopped') from error
        except (BrokenPipeError, ConnectionError, OSError):
            self.start()
            raise ConnectionError('Anime4KCPP worker failed; process restarted')
        result = np.frombuffer(payload, dtype=np.uint8).reshape(response_height, response_width, 4).copy()
        return result, (time.perf_counter() - started) * 1000

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: process.kill()
            except Exception: pass
            process.wait(timeout=2)
        for pipe in (getattr(process,'stdin',None),getattr(process,'stdout',None)):
            if pipe is not None:
                try: pipe.close()
                except Exception: pass
