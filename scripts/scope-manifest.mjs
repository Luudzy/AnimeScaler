import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { lstat, mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function relativePath(root, input) {
  const absolute = resolve(root, input);
  const path = relative(root, absolute);
  if (path === '..' || path.startsWith(`..${sep}`) || isAbsolute(path)) throw new Error(`Path outside project root: ${input}`);
  return { absolute, path: path.split(sep).join('/') };
}

async function hashFile(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

export async function buildScopeManifest(paths, { root = projectRoot, capturedAt = new Date().toISOString() } = {}) {
  root = resolve(root);
  if (!Array.isArray(paths) || paths.length === 0) throw new Error('At least one scope path is required');
  if (!Number.isFinite(Date.parse(capturedAt))) throw new Error('Invalid scope capture timestamp');
  const files = [];
  const seen = new Set();
  for (const input of paths) {
    const resolved = relativePath(root, input);
    if (seen.has(resolved.path)) throw new Error(`Duplicate scope path: ${resolved.path}`);
    seen.add(resolved.path);
    const info = await lstat(resolved.absolute, { bigint: true });
    if (!info.isFile() || info.isSymbolicLink()) throw new Error(`Scope path is not a regular file: ${resolved.path}`);
    const sha256 = await hashFile(resolved.absolute);
    const after = await lstat(resolved.absolute, { bigint: true });
    if (info.ino !== after.ino || info.size !== after.size || info.mtimeNs !== after.mtimeNs || info.ctimeNs !== after.ctimeNs) {
      throw new Error(`Scope path changed during capture: ${resolved.path}`);
    }
    files.push({ path: resolved.path, sha256 });
  }
  files.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  return { schemaVersion: 1, capturedAt, files };
}

export async function writeScopeManifest(output, paths, options = {}) {
  const root = resolve(options.root ?? projectRoot);
  const target = relativePath(root, output).absolute;
  if (paths.some((input) => relativePath(root, input).absolute === target)) throw new Error('Scope manifest output cannot replace an input file');
  const manifest = await buildScopeManifest(paths, { ...options, root });
  await mkdir(dirname(target), { recursive: true });
  const temp = `${target}.${randomUUID()}.tmp`;
  try {
    await writeFile(temp, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    await rename(temp, target);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
  return manifest;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  const [output, ...paths] = process.argv.slice(2);
  if (!output || paths.length === 0) throw new Error('Usage: node scripts/scope-manifest.mjs <output.json> <file> [file...]');
  const manifest = await writeScopeManifest(output, paths);
  console.log(`[scope] wrote ${manifest.files.length} file hashes to ${output}`);
}
