import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';

const root = resolve('.');
const environment = join(root, 'native', '.venv-opencl');
const python = process.platform === 'win32'
  ? join(environment, 'Scripts', 'python.exe')
  : join(environment, 'bin', 'python');
const basePython = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const lock = join(root, 'native', 'requirements-native-host.lock');
const requirements = (await readFile(lock, 'utf8')).split(/\r?\n/)
  .map(line => line.trim()).filter(line => line && !line.startsWith('#'))
  .map(line => {
    const separator = line.indexOf('==');
    if (separator < 1) throw new Error(`Invalid OpenCL dependency pin: ${line}`);
    return [line.slice(0, separator), line.slice(separator + 2)];
  });

function run(executable, args, label, stdio = 'inherit') {
  const result = spawnSync(executable, args, { cwd: root, stdio });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${label} failed with exit code ${result.status}`);
  return result;
}

function packageCheck() {
  const expected = JSON.stringify(Object.fromEntries(requirements));
  const code = `import json; from importlib.metadata import PackageNotFoundError, version; expected=json.loads(${JSON.stringify(expected)}); actual={};\nfor name in expected:\n try: actual[name]=version(name)\n except PackageNotFoundError: actual[name]=None\nraise SystemExit(0 if actual==expected else 1)`;
  return spawnSync(python, ['-I', '-c', code], { cwd: root, stdio: 'ignore' });
}

if (!existsSync(python)) {
  const version = run(basePython, ['-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}"); raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'], 'Python 3.10+ check', 'pipe');
  process.stdout.write(`Using Python ${version.stdout.toString().trim()} for the OpenCL environment.\n`);
  run(basePython, ['-m', 'venv', environment], 'OpenCL virtual environment creation');
}

run(python, ['-c', 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'], 'OpenCL Python version check');
if (packageCheck().status !== 0) {
  run(python, ['-m', 'pip', 'install', '--disable-pip-version-check', '--requirement', lock], 'Pinned OpenCL dependency install');
}
if (packageCheck().status !== 0) throw new Error('OpenCL environment does not match requirements-native-host.lock');
run(python, ['-I', '-c', 'import aiohttp, idna, numpy'], 'OpenCL isolated import check');
console.log('[setup] OpenCL Python environment verified (isolated, pinned dependencies)');
