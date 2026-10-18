import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { setupModels, setupProjectModels, validateModelSetupInputs } from '../scripts/setup-models.mjs';

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const model = (bytes) => ({ name: 'model.onnx', url: 'https://example.invalid/model', sha256: hash(bytes) });
const animeSource = '2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx';

async function withDirectory(run) {
  const directory = await mkdtemp(join(tmpdir(), 'animescaler-setup-'));
  try { await run(directory); } finally { await rm(directory, { recursive: true, force: true }); }
}

test('setupModels writes a hash-verified download to the destination', async () => {
  await withDirectory(async (directory) => {
    const bytes = Buffer.from('verified model');
    await setupModels([model(bytes)], { directory, fetchImpl: async () => new Response(bytes) });
    assert.deepEqual(await readFile(join(directory, 'model.onnx')), bytes);
  });
});

test('model setup reports missing AnimeJaNai source before creating or downloading files', async () => {
  await withDirectory(async (directory) => {
    let downloadStarted = false;
    await assert.rejects(setupProjectModels({ root: directory,
      setupModelsImpl: async () => { downloadStarted = true; } }), /AnimeJaNai source.*models\/ANIMEJANAI\.md/);
    assert.equal(downloadStarted, false);
    assert.deepEqual(await readdir(directory), []);
  });
});

test('model setup begins download after its source preflight succeeds', async () => {
  await withDirectory(async (directory) => {
    const models = join(directory, 'models');
    let preflightPath;
    let downloadStarted = false;
    await setupProjectModels({ root: directory, validateInputsImpl: async (root) => { preflightPath = root; },
      setupModelsImpl: async (modelList, options) => {
      downloadStarted = true;
      assert.equal(modelList.length, 2);
      assert.equal(options.directory, models);
    } });
    assert.equal(preflightPath, directory);
    assert.equal(downloadStarted, true);
  });
});

test('full setup checks required model inputs before copying runtime files', async () => {
  const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  assert.equal(packageJson.scripts['setup:preflight'], 'node scripts/setup-models.mjs --preflight-only');
  assert.match(packageJson.scripts.setup, /^npm run setup:preflight && npm run setup:runtime && npm run setup:models/);
});

test('model setup rejects an AnimeJaNai source with the wrong hash', async () => {
  await withDirectory(async (directory) => {
    const models = join(directory, 'models');
    await mkdir(models);
    const source = join(models, animeSource);
    await writeFile(source, 'wrong source');
    await assert.rejects(validateModelSetupInputs(directory), /AnimeJaNai source.*SHA-256/);
  });
});

test('setupModels preserves an existing artifact when the replacement hash fails', async () => {
  await withDirectory(async (directory) => {
    const existing = Buffer.from('existing artifact');
    await writeFile(join(directory, 'model.onnx'), existing);
    await assert.rejects(
      setupModels([model(Buffer.from('expected'))], { directory, fetchImpl: async () => new Response('corrupt') }),
      /SHA-256 mismatch/
    );
    assert.deepEqual(await readFile(join(directory, 'model.onnx')), existing);
    assert.deepEqual(await readdir(directory), ['model.onnx']);
  });
});

test('setupModels preserves an existing artifact when atomic promotion fails', async () => {
  await withDirectory(async (directory) => {
    const existing = Buffer.from('existing artifact');
    await writeFile(join(directory, 'model.onnx'), existing);
    await assert.rejects(
      setupModels([model(Buffer.from('replacement'))], {
        directory,
        fetchImpl: async () => new Response('replacement'),
        renameImpl: async () => { throw new Error('rename failed'); },
      }),
      /rename failed/
    );
    assert.deepEqual(await readFile(join(directory, 'model.onnx')), existing);
    assert.deepEqual(await readdir(directory), ['model.onnx']);
  });
});

test('setupModels replaces an invalid existing artifact only after verifying the download', async () => {
  await withDirectory(async (directory) => {
    const existing = Buffer.from('old invalid artifact');
    const replacement = Buffer.from('verified replacement');
    await writeFile(join(directory, 'model.onnx'), existing);
    await setupModels([model(replacement)], { directory, fetchImpl: async () => new Response(replacement) });
    assert.deepEqual(await readFile(join(directory, 'model.onnx')), replacement);
    assert.deepEqual(await readdir(directory), ['model.onnx']);
  });
});

test('setupModels rejects declared and streamed size over the limit', async () => {
  await withDirectory(async (directory) => {
    const bytes = Buffer.from('too large');
    await assert.rejects(
      setupModels([model(bytes)], { directory, maxBytes: 4, fetchImpl: async () => new Response(bytes) }),
      /exceeds size limit/
    );
    await assert.rejects(
      setupModels([model(bytes)], {
        directory,
        maxBytes: 4,
        fetchImpl: async () => new Response(bytes, { headers: { 'content-length': String(bytes.length) } }),
      }),
      /exceeds size limit/
    );
    assert.deepEqual(await readdir(directory), []);
  });
});

test('setupModels stops a stalled response at the deadline', async () => {
  await withDirectory(async (directory) => {
    const body = new ReadableStream({ pull: () => new Promise(() => {}) });
    await assert.rejects(
      setupModels([model(Buffer.from('x'))], { directory, timeoutMs: 10, fetchImpl: async () => new Response(body) }),
      /timed out/i
    );
    assert.deepEqual(await readdir(directory), []);
  });
});

test('setupModels reports a timeout while waiting for response headers', async () => {
  await withDirectory(async (directory) => {
    await assert.rejects(
      setupModels([model(Buffer.from('x'))], {
        directory,
        timeoutMs: 10,
        fetchImpl: async (_url, { signal }) => new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason), { once: true });
        }),
      }),
      /timed out/i
    );
    assert.deepEqual(await readdir(directory), []);
  });
});
