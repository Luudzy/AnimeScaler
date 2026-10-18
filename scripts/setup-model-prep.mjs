import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const environment = join(root, 'native', '.venv-model-prep');
const python = process.platform === 'win32'
  ? join(environment, 'Scripts', 'python.exe')
  : join(environment, 'bin', 'python');
const basePython = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const lock = join(root, 'native', 'requirements-model-prep.lock');
const packages = {
  onnx: '1.20.1', onnxruntime: '1.23.2', numpy: '2.2.6', ml_dtypes: '0.5.4',
  protobuf: '6.33.5', typing_extensions: '4.14.1', coloredlogs: '15.0.1',
  flatbuffers: '25.12.19', packaging: '25.0', sympy: '1.14.0',
  humanfriendly: '10.0', mpmath: '1.3.0',
};

function run(executable, args, label) {
  const result = spawnSync(executable, args, { cwd: root, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${label} failed with exit code ${result.status}`);
}

function packageCheck(pythonPath) {
  const expected = JSON.stringify(packages);
  const code = `import json; from importlib.metadata import PackageNotFoundError, version; expected=json.loads(${JSON.stringify(expected)}); actual={};\nfor name in expected:\n try: actual[name]=version(name)\n except PackageNotFoundError: actual[name]=None\nraise SystemExit(0 if actual==expected else 1)`;
  return spawnSync(pythonPath, ['-I', '-c', code], { cwd: root, stdio: 'ignore' });
}

if (!existsSync(python)) {
  run(basePython, ['-c', 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'], 'Python 3.10 check');
  run(basePython, ['-m', 'venv', environment], 'Model-preparation virtual environment creation');
}

run(python, ['-c', 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'], 'Model-preparation Python version check');
if (packageCheck(python).status !== 0) {
  run(python, ['-m', 'pip', 'install', '--disable-pip-version-check', '--requirement', lock], 'Pinned model-preparation dependency install');
}
if (packageCheck(python).status !== 0) throw new Error('Model-preparation environment does not match requirements-model-prep.lock');
console.log('[setup] model-preparation environment verified (Python 3.10, pinned CPU packages)');
