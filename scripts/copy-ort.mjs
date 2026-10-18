import { copyFile, mkdir, readFile, readdir, rename, rm } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { writeRuntimeManifest } from './ort-runtime-integrity.mjs';

const source = join('node_modules', 'onnxruntime-web', 'dist');
const destination = 'vendor';
const rootPackage = JSON.parse(await readFile('package.json', 'utf8'));
const runtimePackage = JSON.parse(await readFile(join('node_modules', 'onnxruntime-web', 'package.json'), 'utf8'));
const expectedVersion = rootPackage.dependencies?.['onnxruntime-web'];
if (runtimePackage.version !== expectedVersion) {
  throw new Error(`onnxruntime-web version mismatch: expected ${expectedVersion}, got ${runtimePackage.version}`);
}
await mkdir(destination, { recursive: true });
const files = await readdir(source);
const wanted = files.filter((name) =>
  name === 'ort.webgpu.bundle.min.mjs' || /^ort-wasm.*\.(wasm|mjs)$/.test(name)
);
if (!wanted.includes('ort.webgpu.bundle.min.mjs')) {
  throw new Error('onnxruntime-web dist/ort.webgpu.bundle.min.mjs was not found. Run npm install first.');
}
for (const name of wanted.sort()) {
  const temp = join(destination, `${name}.${randomUUID()}.tmp`);
  try {
    await copyFile(join(source, name), temp);
    await rename(temp, join(destination, name));
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
  console.log(`[runtime] copied ${name}`);
}
await writeRuntimeManifest(destination, runtimePackage.version, wanted);
