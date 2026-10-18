import { createHash, randomUUID } from 'node:crypto';
import { link, lstat, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import protobuf from '../node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js';

const SOURCE = 'models/2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx';
const TARGET = 'models/2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu.onnx';
const SOURCE_SHA256 = 'f6bbfbed502b8d300ee9d678d2c277d4b56235c36d053c3a1a41be060c54c0e6';
const TARGET_SHA256 = '3eca0dee498cc0bfb311a6ed97e23bf225f7660de033ca9683ffefd155c0b164';

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function targetIsVerified(path) {
  let info;
  try { info = await lstat(path); }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
  if (!info.isFile() || info.isSymbolicLink()) throw new Error('Existing AnimeJaNai WebGPU output is not a regular file; preserved');
  if (sha256(await readFile(path)) !== TARGET_SHA256) throw new Error('Existing AnimeJaNai WebGPU output differs from the verified artifact; preserved');
  return true;
}

export async function prepareAnimejanai({ root = process.cwd() } = {}) {
  root = resolve(root);
  const sourcePath = join(root, SOURCE);
  const targetPath = join(root, TARGET);
  const sourceBytes = await readFile(sourcePath);
  if (sha256(sourceBytes) !== SOURCE_SHA256) {
    throw new Error('Unexpected AnimeJaNai source SHA-256; refusing to modify an unknown model.');
  }
  const model = protobuf.onnx.ModelProto.decode(sourceBytes);
  const dimensions = model.graph.output[0].type.tensorType.shape.dim;
  dimensions[2].dimParam = 'output_height';
  dimensions[3].dimParam = 'output_width';
  const prepared = Buffer.from(protobuf.onnx.ModelProto.encode(model).finish());
  const preparedHash = sha256(prepared);
  if (preparedHash !== TARGET_SHA256) throw new Error('Prepared AnimeJaNai output does not match the pinned SHA-256.');
  if (await targetIsVerified(targetPath)) {
    console.log('[models] verified AnimeJaNai WebGPU model already present');
    return { prepared: false, path: targetPath, sha256: preparedHash };
  }

  await mkdir(dirname(targetPath), { recursive: true });
  const temporaryPath = `${targetPath}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, prepared, { flag: 'wx' });
    try { await link(temporaryPath, targetPath); }
    catch (error) {
      if (error.code !== 'EEXIST' || !(await targetIsVerified(targetPath))) throw error;
      console.log('[models] verified AnimeJaNai WebGPU model created by another setup process');
      return { prepared: false, path: targetPath, sha256: preparedHash };
    }
  } finally {
    await rm(temporaryPath, { force: true });
  }
  console.log(`[models] prepared AnimeJaNai WebGPU: ${preparedHash}`);
  return { prepared: true, path: targetPath, sha256: preparedHash };
}
