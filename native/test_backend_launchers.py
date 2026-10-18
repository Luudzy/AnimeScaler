"""Exercise Windows launchers with CPU stubs; never load a backend or start a server."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import venv


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {'ROCm': 'rocm', 'TensorRT': 'tensorrt', 'OpenCL': 'opencl', 'FSR3': 'fsr3'}


@unittest.skipUnless(os.name == 'nt', 'Windows cmd.exe launchers')
class BackendLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='launchers & (CPU) ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        native = self.root / 'native'
        native.mkdir()
        scripts = self.root / 'scripts'
        scripts.mkdir()
        (scripts / 'setup-opencl.mjs').write_text(
            '''import { cpSync, existsSync } from 'node:fs';
if (process.env.TEST_BOOTSTRAP_EXIT !== '0') {
  console.log('STUB_bootstrap'); process.exitCode = 5;
} else {
  if (!existsSync('native/.venv-opencl')) cpSync('native/.venv-nvidia', 'native/.venv-opencl', { recursive: true });
  console.log('STUB_bootstrap');
}
''', encoding='utf-8')
        (scripts / 'setup-fsr3.mjs').write_text(
            '''import { cpSync, existsSync } from 'node:fs';
if (process.env.TEST_BOOTSTRAP_EXIT === '5') {
  console.log('STUB_fsr3_bootstrap'); process.exitCode = 5;
} else {
  if (!existsSync('native/.venv-fsr3')) cpSync('native/.venv-nvidia', 'native/.venv-fsr3', { recursive: true });
  console.log('STUB_fsr3_bootstrap');
}
''', encoding='utf-8')
        venv.EnvBuilder(with_pip=False).create(native / '.venv-nvidia')
        self.python = native / '.venv-nvidia/Scripts/python.exe'
        for name in BACKENDS:
            script = (ROOT / f'Iniciar-{name}.cmd').read_text(encoding='utf-8')
            # Simulate dismissing PAUSE, including a changed errorlevel. This proves
            # the launcher saves the original failure before user interaction.
            script = re.sub(r'(?m)^(\s*)pause\s*$', r'\1echo LAUNCHER_PAUSE\n\1cmd /d /c exit 0', script)
            (self.root / f'Iniciar-{name}.cmd').write_text(script, encoding='utf-8')
        for name in ('AMD', 'NVIDIA'):
            (self.root / f'Iniciar-{name}.cmd').write_text(
                (ROOT / f'Iniciar-{name}.cmd').read_text(encoding='utf-8'), encoding='utf-8')
        stub = '''import os, pathlib, sys
stage = "preflight" if "preflight" in __file__ else "prepare" if "prepare_tensorrt" in __file__ else "service"
with pathlib.Path("calls.txt").open("a") as log:
    log.write(f"{stage}|{os.environ.get('ANIMESCALER_BACKEND', '')}|{sys.flags.isolated}|{sys.argv[1:]}\\n")
print("STUB_" + stage)
sys.exit(int(os.environ.get("TEST_" + stage.upper() + "_EXIT", "0")))
'''
        for file in ['backend_preflight.py', 'prepare_tensorrt.py', 'server.py', 'fsr3_server.py']:
            (native / file).write_text(stub, encoding='utf-8')
        build = native / 'fsr3-probe/build'
        build.mkdir(parents=True)
        (build / 'fsr3-worker.exe').touch()
        (build.parent / 'build.ps1').write_text('Write-Output "STUB_build"; exit 13', encoding='utf-8')
        self.sdk = self.root / 'SDK & (CPU)'
        self.sdk.mkdir()

    def run_launcher(self, name, **changes):
        environment = {key: value for key, value in os.environ.items()
                       if not key.upper().startswith(('ANIMESCALER_', 'TEST_'))}
        environment.update({
            'ANIMESCALER_ROCM_PYTHON': str(self.python),
            'ANIMESCALER_OPENCL_PYTHON': str(self.python),
            'ANIMESCALER_FSR3_PYTHON': str(self.python),
            'ANIMESCALER_FSR_SDK': str(self.sdk),
            'LAUNCHER_TEST_SCRIPT': str(self.root / f'Iniciar-{name}.cmd'),
        })
        for key, value in changes.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = str(value)
        calls = self.root / 'calls.txt'
        calls.unlink(missing_ok=True)
        result = subprocess.run(
            f'"{os.environ["COMSPEC"]}" /d /s /c "call "%LAUNCHER_TEST_SCRIPT%""',
            cwd=self.root.parent, env=environment, capture_output=True, text=True, timeout=20,
        )
        return result, calls.read_text() if calls.exists() else ''

    def test_success_runs_preflight_then_isolated_service_from_project_directory(self):
        for name, backend in BACKENDS.items():
            with self.subTest(backend=name):
                result, calls = self.run_launcher(name)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                expected_stages = ['preflight', 'prepare', 'service'] if backend == 'tensorrt' else ['preflight', 'service']
                self.assertEqual([line.split('|')[0] for line in calls.splitlines()], expected_stages)
                self.assertIn('--backend', calls)
                self.assertIn(backend, calls)
                self.assertTrue(all(line.split('|')[2] == '1' for line in calls.splitlines()))
                if name != 'FSR3':
                    self.assertTrue(all(line.split('|')[1] == backend for line in calls.splitlines()))
                self.assertEqual(result.stdout.count('LAUNCHER_PAUSE'), 1)

    def test_two_entrypoints_route_to_vendor_primary_backend(self):
        for entry, expected in [('AMD', 'rocm'), ('NVIDIA', 'tensorrt')]:
            with self.subTest(entry=entry):
                environment = {key: value for key, value in os.environ.items()
                               if not key.upper().startswith(('ANIMESCALER_', 'TEST_'))}
                environment.update({
                    'ANIMESCALER_ROCM_PYTHON': str(self.python),
                    'ANIMESCALER_OPENCL_PYTHON': str(self.python),
                    'ANIMESCALER_FSR3_PYTHON': str(self.python),
                    'ANIMESCALER_FSR_SDK': str(self.sdk),
                })
                entry_script = self.root / f'Iniciar-{entry}.cmd'
                command = f'"{os.environ["COMSPEC"]}" /d /s /c "call "%ENTRY_SCRIPT%""'
                environment['ENTRY_SCRIPT'] = str(entry_script)
                result = subprocess.run(
                    command,
                    cwd=self.root.parent, env=environment, input='1\n',
                    capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = (self.root / 'calls.txt').read_text()
                self.assertIn(f'preflight|{expected}|1|', calls)
                self.assertIn('service|', calls)

    def test_failed_preflight_pauses_preserves_code_and_never_starts_service(self):
        for name in BACKENDS:
            with self.subTest(backend=name):
                result, calls = self.run_launcher(name, TEST_PREFLIGHT_EXIT=7)
                self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
                self.assertNotIn('service|', calls)
                self.assertEqual(result.stdout.count('LAUNCHER_PAUSE'), 1)

    def test_service_failure_survives_pause(self):
        for name in BACKENDS:
            with self.subTest(backend=name):
                result, calls = self.run_launcher(name, TEST_SERVICE_EXIT=9)
                self.assertIn('service|', calls)
                self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
                self.assertEqual(result.stdout.count('LAUNCHER_PAUSE'), 1)

    def test_opencl_missing_explicit_python_is_reported_before_preflight(self):
        result, calls = self.run_launcher('OpenCL', ANIMESCALER_OPENCL_PYTHON=self.root / 'missing.exe')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(calls, '')
        self.assertIn('Python OpenCL nao encontrado', result.stdout)
        self.assertEqual(result.stdout.count('LAUNCHER_PAUSE'), 1)

    def test_opencl_bootstraps_isolated_environment_from_one_cmd(self):
        result, calls = self.run_launcher('OpenCL', ANIMESCALER_OPENCL_PYTHON=None,
                                          TEST_BOOTSTRAP_EXIT='0')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('STUB_bootstrap', result.stdout)
        self.assertEqual([line.split('|')[0] for line in calls.splitlines()], ['preflight', 'service'])

    def test_fsr3_build_failure_pauses_and_prevents_service(self):
        (self.root / 'native/fsr3-probe/build/fsr3-worker.exe').unlink()
        result, calls = self.run_launcher('FSR3')
        self.assertIn('STUB_build', result.stdout)
        self.assertEqual(result.returncode, 13, result.stdout + result.stderr)
        self.assertNotIn('service|', calls)
        self.assertEqual(result.stdout.count('LAUNCHER_PAUSE'), 1)

    def test_missing_configuration_stops_before_preflight_without_global_fallback(self):
        for name, key in [('ROCm', 'ANIMESCALER_ROCM_PYTHON'),
                          ('FSR3', 'ANIMESCALER_FSR3_PYTHON')]:
            with self.subTest(backend=name):
                result, calls = self.run_launcher(name, **{key: None})
                self.assertEqual(result.returncode, 1)
                self.assertEqual(calls, '')
                self.assertIn('Set ' + key, result.stdout)

        result, calls = self.run_launcher('OpenCL', ANIMESCALER_OPENCL_PYTHON=None,
                                          TEST_BOOTSTRAP_EXIT='5')
        self.assertEqual(result.returncode, 5)
        self.assertEqual(calls, '')
        self.assertIn('STUB_bootstrap', result.stdout)

    def test_project_venv_fallback_preserves_backend_selection(self):
        for name, directory, key in [('ROCm', 'rocm', 'ANIMESCALER_ROCM_PYTHON'),
                                      ('OpenCL', 'opencl', 'ANIMESCALER_OPENCL_PYTHON'),
                                      ('FSR3', 'fsr3', 'ANIMESCALER_FSR3_PYTHON')]:
            with self.subTest(backend=name):
                target = self.root / f'native/.venv-{directory}'
                (self.python.parents[1]).rename(target)
                try:
                    changes = {key: None}
                    if name == 'OpenCL':
                        changes['TEST_BOOTSTRAP_EXIT'] = '0'
                    result, calls = self.run_launcher(name, **changes)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('service|', calls)
                finally:
                    target.rename(self.python.parents[1])

    def write_local_paths(self, python=None):
        variables = {'ANIMESCALER_ROCM_PYTHON': python or self.python,
                     'ANIMESCALER_FSR3_PYTHON': python or self.python,
                     'ANIMESCALER_FSR_SDK': self.sdk}
        script = '@echo off\n' + ''.join(
            f'if not defined {key} set "{key}={value}"\n' for key, value in variables.items())
        (self.root / 'native/launcher-paths.cmd').write_text(script, encoding='utf-8')

    def test_local_configuration_resolves_existing_python_and_sdk_without_environment(self):
        self.write_local_paths()
        for name, key in [('ROCm', 'ANIMESCALER_ROCM_PYTHON'),
                          ('FSR3', 'ANIMESCALER_FSR3_PYTHON')]:
            with self.subTest(backend=name):
                result, calls = self.run_launcher(name, **{key: None, 'ANIMESCALER_FSR_SDK': None})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('service|', calls)

    def test_local_configuration_preserves_explicit_interpreter(self):
        self.write_local_paths(self.root / 'missing.exe')
        result, calls = self.run_launcher('ROCm')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('service|', calls)

    def test_project_venv_takes_priority_over_machine_configuration(self):
        self.write_local_paths(self.root / 'missing.exe')
        target = self.root / 'native/.venv-rocm'
        self.python.parents[1].rename(target)
        result, calls = self.run_launcher('ROCm', ANIMESCALER_ROCM_PYTHON=None)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('service|', calls)


if __name__ == '__main__':
    unittest.main()
