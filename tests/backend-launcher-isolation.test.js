import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const readLauncher = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('ROCm, TensorRT, and FSR3 services start with isolated Python mode', async () => {
  const launchers = await Promise.all([
    readLauncher('Iniciar-ROCm.cmd'),
    readLauncher('Iniciar-TensorRT.cmd'),
    readLauncher('Iniciar-FSR3.cmd'),
  ]);

  for (const script of launchers) {
    const serviceLine = script.split(/\r?\n/).find((line) => /native\\(?:fsr3_)?server\.py/i.test(line));
    assert.ok(serviceLine, 'expected a native service launch command');
    assert.match(serviceLine, /-u\b/i);
    assert.match(serviceLine, /-I\b/i);
  }
});

test('OpenCL service launch keeps isolated Python mode', async () => {
  const script = await readLauncher('Iniciar-OpenCL.cmd');
  const serviceLine = script.split(/\r?\n/).find((line) => /native\\server\.py/i.test(line));
  assert.ok(serviceLine, 'expected a native service launch command');
  assert.match(serviceLine, /-u\b/i);
  assert.match(serviceLine, /-I\b/i);
});
