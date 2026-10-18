import os
import threading
import subprocess
import unittest
from pathlib import Path

import numpy as np

from anime4kcpp_worker import Anime4KCPPWorker
from anime4kcpp_protocol import HEADER, RESPONSE_MAGIC


class BlockingStdout:
    def __init__(self,stopped): self.stopped=stopped
    def read(self,size):
        self.stopped.wait()
        return b''


class FakeStdin:
    def write(self,data): return len(data)
    def flush(self): pass
    def close(self): pass


class BlockingWorkerProcess:
    def __init__(self):
        self.stopped=threading.Event();self.stdin=FakeStdin();self.stdout=BlockingStdout(self.stopped)
    def poll(self): return None
    def terminate(self): self.stopped.set()
    def kill(self): self.stopped.set()
    def wait(self,timeout=None): return 0


class InvalidHeaderStdout:
    def __init__(self): self.read_sizes=[]
    def read(self,size):
        self.read_sizes.append(size)
        if len(self.read_sizes)>1: raise AssertionError('payload read before header validation')
        return HEADER.pack(RESPONSE_MAGIC,99,40,40,80,80)


class InvalidHeaderProcess(BlockingWorkerProcess):
    def __init__(self):
        super().__init__();self.stdout=InvalidHeaderStdout()


class SlowExitProcess:
    def __init__(self): self.stdin=FakeStdin();self.wait_calls=0;self.kill_calls=0
    def terminate(self): pass
    def kill(self): self.kill_calls+=1
    def wait(self,timeout=None):
        self.wait_calls+=1
        if self.wait_calls==1: raise subprocess.TimeoutExpired('worker',timeout)
        return 0


class Anime4KCPPWorkerTests(unittest.TestCase):
    def test_close_waits_after_killing_a_worker_that_ignores_terminate(self):
        process=SlowExitProcess();worker=Anime4KCPPWorker.__new__(Anime4KCPPWorker);worker.process=process
        worker.close()
        self.assertEqual(process.kill_calls,1)
        self.assertEqual(process.wait_calls,2)
        self.assertIsNone(worker.process)

    def test_invalid_response_header_is_rejected_before_payload_read(self):
        process=InvalidHeaderProcess()
        worker=Anime4KCPPWorker.__new__(Anime4KCPPWorker)
        worker.process=process;worker.request_timeout=.1
        with self.assertRaisesRegex(ConnectionError,'invalid header'):
            worker.infer(np.zeros((40,40,4),dtype=np.uint8),1,(80,80))
        self.assertEqual(process.stdout.read_sizes,[HEADER.size])
        self.assertTrue(process.stopped.is_set())

    def test_unresponsive_worker_is_terminated_at_deadline(self):
        process=BlockingWorkerProcess()
        worker=Anime4KCPPWorker.__new__(Anime4KCPPWorker)
        worker.process=process;worker.request_timeout=.01
        frame=np.zeros((40,40,4),dtype=np.uint8)
        with self.assertRaisesRegex(TimeoutError,'timed out'):
            worker.infer(frame,1,(80,80))
        self.assertTrue(process.stopped.is_set())
        self.assertIsNone(worker.process)

    @unittest.skipUnless(os.environ.get('ANIMESCALER_RUN_OPENCL_INTEGRATION') == '1',
                         'Set ANIMESCALER_RUN_OPENCL_INTEGRATION=1 for the real OpenCL worker')
    def test_real_opencl_worker_round_trip_preserves_input_and_output_dimensions(self):
        worker = Anime4KCPPWorker(Path(__file__).resolve().parents[1], 'anime4kcpp-acnet-f8b4')
        frame = np.zeros((64, 64, 4), dtype=np.uint8)
        frame[..., 3] = 255
        try:
            output, duration = worker.infer(frame, 1, (128, 128))
            self.assertEqual(output.shape, (128, 128, 4))
            self.assertEqual(output.dtype, np.uint8)
            self.assertGreaterEqual(duration, 0)
        finally:
            worker.close()


if __name__=='__main__': unittest.main()
