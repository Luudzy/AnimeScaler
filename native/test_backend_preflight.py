import unittest

from backend_preflight import anime4kcpp_worker_available,backend_issues,resolve_fsr_sdk_path
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


class BackendPreflightTests(unittest.TestCase):
    def test_fsr_sdk_path_uses_explicit_or_environment_value_and_rejects_missing(self):
        environment={'ANIMESCALER_FSR_SDK':'D:/sdk/from-env'}
        self.assertEqual(resolve_fsr_sdk_path(environ=environment),Path('D:/sdk/from-env'))
        self.assertEqual(resolve_fsr_sdk_path('E:/sdk/explicit',environment),Path('E:/sdk/explicit'))
        with self.assertRaisesRegex(ValueError,'ANIMESCALER_FSR_SDK'):
            resolve_fsr_sdk_path(environ={})

    def test_fsr3_entrypoints_have_no_personal_sdk_fallback(self):
        paths=(ROOT/'native/fsr3_worker.py',ROOT/'native/fsr3-probe/run.py')
        for path in paths:
            source=path.read_text(encoding='utf-8')
            self.assertNotIn(r'C:\Users\example\Downloads\FidelityFX-SDK-main',source)
            self.assertIn('ANIMESCALER_FSR_SDK' if path.name == 'run.py' else 'resolve_fsr_sdk_path',source)

    def test_fsr3_probe_exits_with_configuration_error_when_sdk_is_missing(self):
        environment=dict(os.environ)
        environment.pop('ANIMESCALER_FSR_SDK',None)
        result=subprocess.run([sys.executable,str(ROOT/'native/fsr3-probe/run.py'),
                               '--input','unused.png','--output-dir','unused-output'],
                              env=environment,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2)
        self.assertIn('Set --sdk or configure ANIMESCALER_FSR_SDK',result.stderr)

    def test_rocm_requires_hip_torch_and_shared_packages(self):
        packages = {'torch': '2.12.0+rocm7.15', 'spandrel': '0.4.1', 'numpy': '2.2.6', 'aiohttp': '3.12.0'}
        self.assertEqual(backend_issues('rocm', (3, 10), packages), [])
        errors = backend_issues('rocm', (3, 10), {**packages, 'torch': '2.12.0+cu128', 'numpy': None})
        self.assertTrue(any('ROCm/HIP' in error for error in errors))
        self.assertTrue(any('numpy' in error for error in errors))

    def test_tensorrt_requires_cuda_torch_and_supported_tensorrt_major(self):
        packages = {'torch': '2.7.0+cu128', 'tensorrt': '10.13.0', 'spandrel': '0.4.1', 'numpy': '2.2.6', 'aiohttp': '3.12.0'}
        self.assertEqual(backend_issues('tensorrt', (3, 10), packages), [])
        errors = backend_issues('tensorrt', (3, 10), {**packages, 'tensorrt': '9.3.0', 'torch': '2.7.0+rocm7.0'})
        self.assertTrue(any('TensorRT 10' in error for error in errors))
        self.assertTrue(any('CUDA' in error for error in errors))

    def test_fsr3_requires_python_310_dependencies_and_sdk(self):
        packages = {'aiohttp': '3.12.0', 'numpy': '2.2.6'}
        self.assertEqual(backend_issues('fsr3', (3, 10), packages, sdk_present=True), [])
        errors = backend_issues('fsr3', (3, 11), {'aiohttp': None, 'numpy': '2.2.6'}, sdk_present=False)
        self.assertTrue(any('Python 3.10' in error for error in errors))
        self.assertTrue(any('aiohttp' in error for error in errors))
        self.assertTrue(any('SDK' in error for error in errors))

    def test_unknown_backend_is_rejected(self):
        self.assertTrue(backend_issues('opencl', (3, 10), {}))

    def test_opencl_requires_only_host_packages_and_worker_files(self):
        packages={'numpy':'2.2.6','aiohttp':'3.12.0','idna':'3.10'}
        self.assertEqual(backend_issues('opencl',(3,10),packages,opencl_worker=True),[])
        errors=backend_issues('opencl',(3,10),{'numpy':'2.2.6','aiohttp':'3.12.0'},opencl_worker=True)
        self.assertTrue(any('idna' in error for error in errors))
        errors=backend_issues('opencl',(3,10),{'numpy':'2.2.6'},opencl_worker=True)
        self.assertTrue(any('aiohttp' in error for error in errors))
        errors=backend_issues('opencl',(3,10),packages,opencl_worker=False)
        self.assertTrue(any('Anime4KCPP OpenCL worker' in error for error in errors))

    def test_opencl_worker_preflight_checks_both_executable_and_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertFalse(anime4kcpp_worker_available(root))
            worker=root/'third_party/Anime4KCPP-master/worker/build/bin/Release/anime4kcpp_worker.exe'
            runtime=root/'third_party/Anime4KCPP-master/build-opencl-md/bin/Release'
            worker.parent.mkdir(parents=True);worker.write_bytes(b'worker')
            self.assertFalse(anime4kcpp_worker_available(root))
            runtime.mkdir(parents=True)
            self.assertTrue(anime4kcpp_worker_available(root))


if __name__ == '__main__':
    unittest.main()
