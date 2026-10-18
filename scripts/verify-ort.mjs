import { readFile } from 'node:fs/promises';
import { verifyRuntimeManifest } from './ort-runtime-integrity.mjs';

const packageJson = JSON.parse(await readFile('package.json', 'utf8'));
const version = packageJson.dependencies?.['onnxruntime-web'];
await verifyRuntimeManifest('vendor', version);
console.log(`[runtime] verified onnxruntime-web ${version}`);
