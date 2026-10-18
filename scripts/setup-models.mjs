import { createHash, randomUUID } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { Readable, Transform } from 'node:stream';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MAX_MODEL_BYTES = 512 * 1024 * 1024;
const DOWNLOAD_TIMEOUT_MS = 120_000;
const ANIMEJANAI_SOURCE = '2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx';
const ANIMEJANAI_SOURCE_SHA256 = 'f6bbfbed502b8d300ee9d678d2c277d4b56235c36d053c3a1a41be060c54c0e6';

const models = [
  {
    name: '2x_Adore_renarchi_fp16.onnx',
    url: 'https://github.com/renarchi/Re-SISR/releases/download/Adore/2x_Adore_renarchi_fp16.onnx',
    sha256: '79e007eda5721cfa596945d5849320e2fad67f8596f9ac31ed06cc8e3ddaf57d',
  },
  {
    name: '2x_Adore_renarchi_fp32.onnx',
    url: 'https://github.com/renarchi/Re-SISR/releases/download/Adore/2x_Adore_renarchi_fp32.onnx',
    sha256: '532c1b30711c2bb7ea6ddf123fd219df87ca9816fbb8ddaaad5c1f311d5b7c44',
  },
];

async function digestFile(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

async function valid(path, expected) {
  try { return await digestFile(path) === expected; } catch { return false; }
}

export async function setupModels(modelList = models, {
  directory = 'models', fetchImpl = fetch, timeoutMs = DOWNLOAD_TIMEOUT_MS,
  maxBytes = MAX_MODEL_BYTES, renameImpl = rename,
} = {}) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new RangeError('Model download size limit must be positive');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError('Model download timeout must be positive');
  await mkdir(directory, { recursive: true });
  for (const model of modelList) {
    const target = join(directory, model.name);
    if (await valid(target, model.sha256)) {
      console.log(`[models] OK ${model.name}`);
      continue;
    }
    const temp = `${target}.${randomUUID()}.download`;
    console.log(`[models] downloading ${model.name}`);
    try {
      const signal = AbortSignal.timeout(timeoutMs);
      let response;
      try {
        response = await fetchImpl(model.url, { redirect: 'follow', signal });
      } catch (error) {
        if (signal.aborted) throw new Error(`Timed out downloading ${model.name}`, { cause: error });
        throw error;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status} downloading ${model.name}`);
      const declaredBytes = Number(response.headers.get('content-length'));
      if (Number.isFinite(declaredBytes) && declaredBytes > maxBytes) {
        throw new Error(`Download exceeds size limit for ${model.name}`);
      }
      if (!response.body) throw new Error(`Empty response downloading ${model.name}`);

      const hash = createHash('sha256');
      let bytes = 0;
      const meter = new Transform({
        transform(chunk, _encoding, callback) {
          bytes += chunk.length;
          if (bytes > maxBytes) return callback(new Error(`Download exceeds size limit for ${model.name}`));
          hash.update(chunk);
          callback(null, chunk);
        },
      });
      try {
        await pipeline(Readable.fromWeb(response.body), meter, createWriteStream(temp, { flags: 'wx' }), { signal });
      } catch (error) {
        if (signal.aborted) throw new Error(`Timed out downloading ${model.name}`, { cause: error });
        throw error;
      }
      const actual = hash.digest('hex');
      if (actual !== model.sha256) throw new Error(`SHA-256 mismatch for ${model.name}: ${actual}`);
      await renameImpl(temp, target);
      console.log(`[models] verified ${model.name}`);
    } catch (error) {
      await rm(temp, { force: true });
      throw error;
    }
  }
}

export async function validateModelSetupInputs(root = projectRoot) {
  const path = join(resolve(root), 'models', ANIMEJANAI_SOURCE);
  let bytes;
  try { bytes = await readFile(path); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    throw new Error('AnimeJaNai source is missing; provide the official model and verify models/ANIMEJANAI.md before setup.');
  }
  const actual = createHash('sha256').update(bytes).digest('hex');
  if (actual !== ANIMEJANAI_SOURCE_SHA256) {
    throw new Error('AnimeJaNai source SHA-256 mismatch; preserve it and verify models/ANIMEJANAI.md before setup.');
  }
  return true;
}

export async function setupProjectModels({ root = projectRoot, setupModelsImpl = setupModels, validateInputsImpl = validateModelSetupInputs } = {}) {
  root = resolve(root);
  await validateInputsImpl(root);
  await setupModelsImpl(models, { directory: join(root, 'models') });
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  if (process.argv[2] === '--preflight-only') await validateModelSetupInputs();
  else await setupProjectModels();
}
