import { createReadStream, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { dirname, isAbsolute, relative, resolve, sep, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_OUTPUTS = {
  fp32: 'models/2x_Adore_renarchi_webgpu.onnx',
  fp16: 'models/2x_Adore_renarchi_webgpu_fp16.onnx',
};
const DEFAULT_HASHES = {
  fp32: 'b0a76dff7816816dee2c9a19f58bb1a81dd68178547f9518b64bc0257205749d',
  fp16: 'e503b357fe6d6ce34c3db9980cc2ed37dd1f23d4772b352c8a4f2befb151f2b5',
};

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function resolveOutput(root, path) {
  const absolute = resolve(root, path);
  const relativePath = relative(root, absolute);
  if (relativePath === '..' || relativePath.startsWith(`..${sep}`) || isAbsolute(relativePath)) {
    throw new Error(`Adore output outside project root: ${path}`);
  }
  return absolute;
}

function run(executable, args, root, label) {
  const result = spawnSync(executable, args, { cwd: root, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${label} failed with exit code ${result.status}`);
}

async function generateAdore(root) {
  const environment = join(root, 'native', '.venv-model-prep');
  const python = process.platform === 'win32'
    ? join(environment, 'Scripts', 'python.exe')
    : join(environment, 'bin', 'python');
  if (!existsSync(python)) run(process.execPath, ['scripts/setup-model-prep.mjs'], root, 'Model-preparation environment setup');
  if (!existsSync(python)) throw new Error('Pinned model-preparation Python was not created');
  run(python, ['scripts/convert-adore-fp16.py'], root, 'Adore WebGPU conversion');
}

export async function prepareAdoreWebgpu({
  root = projectRoot,
  outputs = DEFAULT_OUTPUTS,
  expectedHashes = DEFAULT_HASHES,
  generate = generateAdore,
} = {}) {
  root = resolve(root);
  const paths = Object.fromEntries(Object.entries(outputs).map(([name, path]) => [name, resolveOutput(root, path)]));
  const missing = [];
  for (const [name, path] of Object.entries(paths)) {
    if (!existsSync(path)) {
      missing.push(name);
      continue;
    }
    if (await sha256(path) !== expectedHashes[name]) throw new Error(`Existing Adore model hash mismatch: ${outputs[name]}; preserved without overwrite`);
  }
  if (missing.length === 0) {
    console.log('[models] verified Adore WebGPU models already present');
    return { prepared: false, outputs: paths };
  }
  await generate(root, missing);
  for (const [name, path] of Object.entries(paths)) {
    if (!existsSync(path) || await sha256(path) !== expectedHashes[name]) {
      throw new Error(`Generated Adore model failed SHA-256 verification: ${outputs[name]}`);
    }
  }
  console.log('[models] prepared and verified Adore WebGPU models');
  return { prepared: true, outputs: paths };
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  await prepareAdoreWebgpu();
}
