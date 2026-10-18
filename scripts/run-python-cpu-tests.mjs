import { spawnSync } from 'node:child_process';

const executable = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const result = spawnSync(executable, ['-m', 'unittest', 'discover', '-s', 'native', '-p', 'test_*.py'], {
  stdio: 'inherit',
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '', HIP_VISIBLE_DEVICES: '' },
});
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
