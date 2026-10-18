import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const readLauncher = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('AMD and NVIDIA entrypoints expose their supported native backends', async () => {
  const [amd, nvidia] = await Promise.all([
    readLauncher('Iniciar-AMD.cmd'),
    readLauncher('Iniciar-NVIDIA.cmd'),
  ]);
  for (const [script, backends] of [[amd, ['ROCm', 'OpenCL', 'FSR3']], [nvidia, ['TensorRT', 'OpenCL', 'FSR3']]]) {
    for (const backend of backends) assert.match(script, new RegExp(`Iniciar-${backend}\\.cmd`));
    assert.match(script, /choice \/c 123/);
    assert.match(script, /if errorlevel 3 goto :fsr3/);
  }
});

test('ROCm launcher requires explicit interpreter or project-local venv', async () => {
  const script = await readLauncher('Iniciar-ROCm.cmd');
  assert.match(script, /ANIMESCALER_ROCM_PYTHON/);
  assert.match(script, /native\\.venv-rocm\\Scripts\\python\.exe/);
  assert.doesNotMatch(script, /[A-Z]:\\(?:Users|comfyui)/i);
  assert.ok(script.indexOf('--backend rocm') < script.indexOf('native\\server.py'));
});

test('FSR3 launcher requires explicit interpreter and SDK or project-local Python', async () => {
  const script = await readLauncher('Iniciar-FSR3.cmd');
  assert.match(script, /ANIMESCALER_FSR3_PYTHON/);
  assert.match(script, /native\\.venv-fsr3\\Scripts\\python\.exe/);
  assert.match(script, /ANIMESCALER_FSR_SDK/);
  assert.doesNotMatch(script, /[A-Z]:\\(?:Users|Program Files)/i);
  assert.ok(script.indexOf('--backend fsr3') < script.indexOf('native\\fsr3_server.py'));
});

test('TensorRT launcher stays isolated in its project-local environment', async () => {
  const script = await readLauncher('Iniciar-TensorRT.cmd');
  assert.match(script, /native\\.venv-nvidia\\Scripts\\python\.exe/);
  assert.match(script, /GPU NVIDIA/);
  assert.match(script, /TensorRT 10\.x/);
  assert.ok(script.indexOf('native\\prepare_tensorrt.py') < script.indexOf('native\\server.py'));
  assert.doesNotMatch(script, /ANIMESCALER_TENSORRT_PYTHON/);
  assert.doesNotMatch(script, /ANIMESCALER_ROCM_PYTHON|ANIMESCALER_FSR3_PYTHON|comfyui/i);
  assert.ok(script.indexOf('--backend tensorrt') < script.indexOf('native\\server.py'));
});

test('OpenCL launcher bootstraps its pinned environment, then uses package-only preflight and service',async()=>{
  const script=await readLauncher('Iniciar-OpenCL.cmd');
  assert.match(script,/ANIMESCALER_OPENCL_PYTHON/);
  assert.match(script,/node scripts\\setup-opencl\.mjs/);
  assert.ok(script.indexOf('setup-opencl.mjs') < script.indexOf('--backend opencl'));
  assert.ok(script.indexOf('--backend opencl')<script.indexOf('native\\server.py'));
  assert.doesNotMatch(script,/ANIMESCALER_ROCM_PYTHON|ANIMESCALER_TENSORRT/);
  const instructions=await readFile(new URL('../native/README.md',import.meta.url),'utf8');
  assert.match(instructions,/Anime4KCPP OpenCL isolado/);
  assert.match(instructions,/cria `native\/.venv-opencl` e instala os pins/i);
  assert.doesNotMatch(instructions,/depois `python` do PATH/i);
  const ignore=await readFile(new URL('../.gitignore',import.meta.url),'utf8');
  assert.match(ignore,/^native\/\.venv-\*$/m);
  assert.match(instructions,/n[aã]o importa PyTorch, ROCm ou TensorRT/i);
});

test('OpenCL launcher provisions its isolated environment when no interpreter is configured', async () => {
  const script = await readLauncher('Iniciar-OpenCL.cmd');
  assert.match(script, /if not defined ANIMESCALER_OPENCL_PYTHON \([\s\S]*?node scripts\\setup-opencl\.mjs[\s\S]*?set "ANIMESCALER_OPENCL_PYTHON=native\\.venv-opencl\\Scripts\\python\.exe"/);
  assert.doesNotMatch(script, /ANIMESCALER_OPENCL_PYTHON=python/i);
});

test('OpenCL launcher preserves explicit interpreter and project-local venv selection', async () => {
  const script = await readLauncher('Iniciar-OpenCL.cmd');
  const bootstrap = script.indexOf('node scripts\\setup-opencl.mjs');
  const localVenv = script.indexOf('native\\.venv-opencl\\Scripts\\python.exe');
  const missingInterpreter = script.indexOf('if not defined ANIMESCALER_OPENCL_PYTHON (');
  const preflight = script.indexOf('--backend opencl');
  assert.ok(missingInterpreter >= 0 && bootstrap > missingInterpreter);
  assert.ok(localVenv > bootstrap && preflight > localVenv);
});

test('backend setup instructions use environment-specific paths, not machine-local paths', async () => {
  const instructions = await readFile(new URL('../native/README.md', import.meta.url), 'utf8');
  assert.match(instructions, /ANIMESCALER_ROCM_PYTHON/);
  assert.match(instructions, /ANIMESCALER_FSR3_PYTHON/);
  assert.match(instructions, /ANIMESCALER_FSR_SDK/);
  assert.doesNotMatch(instructions, /ANIMESCALER_PYTHON(?!_)/);
  assert.doesNotMatch(instructions, /F:\\comfyui-rocm|C:\\Users\\mybad|C:\\Program Files\\Python310/i);
});

test('FSR3 build script requires SDK path instead of a machine-specific default', async () => {
  const script = await readFile(new URL('../native/fsr3-probe/build.ps1', import.meta.url), 'utf8');
  assert.match(script, /ANIMESCALER_FSR_SDK/);
  assert.match(script, /SDK path.*required/i);
  assert.doesNotMatch(script, /C:\\Users\\mybad/i);
});
