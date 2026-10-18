import { readdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

const testDirectory = resolve('tests');
const files = (await readdir(testDirectory))
  .filter((name) => name.endsWith('.test.js') && name !== 'diagnostic-server.test.js')
  .sort()
  .map((name) => resolve(testDirectory, name));

if (files.length === 0) throw new Error('No CPU JavaScript tests found');
const result = spawnSync(process.execPath, ['--test', ...files], { stdio: 'inherit' });
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
