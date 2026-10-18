import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';

const root = resolve('.');
const environment = join(root, 'native', '.venv-fsr3');
const python = join(environment, 'Scripts', 'python.exe');
const basePython = process.env.ANIMESCALER_FSR3_PYTHON || process.env.PYTHON || 'python';
const lock = join(root, 'native', 'requirements-native-host.lock');
const requirements = (await readFile(lock, 'utf8')).split(/\r?\n/)
  .map(line => line.trim()).filter(line => line && !line.startsWith('#'))
  .map(line => {
    const separator = line.indexOf('==');
    if (separator < 1) throw new Error(`Invalid FSR3 dependency pin: ${line}`);
    return [line.slice(0, separator), line.slice(separator + 2)];
  });

function run(executable, args, label, stdio = 'inherit') {
  const result = spawnSync(executable, args, { cwd: root, stdio });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${label} failed with exit code ${result.status}`);
  return result;
}

function checkPython(executable) {
  return spawnSync(executable, ['-c', 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'],
    { cwd: root, stdio: 'ignore' }).status === 0;
}

const expected = JSON.stringify(Object.fromEntries(requirements));
const packageCheckCode = `import json; from importlib.metadata import PackageNotFoundError, version; expected=json.loads(${JSON.stringify(expected)}); actual={};\nfor name in expected:\n try: actual[name]=version(name)\n except PackageNotFoundError: actual[name]=None\nraise SystemExit(0 if actual==expected else 1)`;
function packageCheck() {
  return spawnSync(python, ['-I', '-c', packageCheckCode], { cwd: root, stdio: 'ignore' });
}

if (!existsSync(python)) {
  if (!checkPython(basePython)) throw new Error('FSR3 requires Python 3.10 to create native/.venv-fsr3');
  run(basePython, ['-m', 'venv', environment], 'FSR3 virtual environment creation');
}
if (!checkPython(python)) throw new Error('FSR3 virtual environment must use Python 3.10');
if (packageCheck().status !== 0) {
  run(python, ['-m', 'pip', 'install', '--disable-pip-version-check', '--requirement', lock], 'Pinned FSR3 dependency install');
}
if (packageCheck().status !== 0) throw new Error('FSR3 environment does not match requirements-native-host.lock');
run(python, ['-I', '-c', 'import aiohttp, idna, numpy'], 'FSR3 isolated import check');
console.log('[setup] FSR3 Python 3.10 environment verified (isolated, pinned dependencies)');
