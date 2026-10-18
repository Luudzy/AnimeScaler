import http from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const projectRoot = fileURLToPath(new URL('..', import.meta.url));
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.wasm': 'application/wasm', '.mp4': 'video/mp4', '.webm': 'video/webm', '.mkv': 'video/x-matroska', '.json': 'application/json' };
const assetExtensions = new Set(['.html', '.js', '.mjs', '.css', '.png', '.wasm', '.onnx']);

export function createDiagnosticServer({ root = projectRoot } = {}) {
  root = path.resolve(root);
  return http.createServer(async (req, res) => {
    try {
      if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
      const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
      if (pathname === '/diagnostics/native-config' || pathname === '/diagnostics/fsr3-config') {
        const expectedHost = `127.0.0.1:${res.socket.localPort}`;
        if (req.headers.host !== expectedHost || req.headers['sec-fetch-site'] !== 'same-origin') { res.writeHead(403).end(); return; }
        const name = pathname.endsWith('fsr3-config') ? 'fsr3-connection.json' : 'connection.json';
        const body = await readFile(path.join(root, 'native', name));
        res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
        res.end(req.method === 'HEAD' ? undefined : body); return;
      }
      const resource = pathname === '/' ? '/diagnostics/performance.html' : pathname;
      const extension = path.extname(resource);
      const publicAsset = /^\/(src|popup|viewer|assets|vendor|models|diagnostics)\//.test(resource) && assetExtensions.has(extension);
      const media = resource.startsWith('/diagnostics/stability-media/') && ['.mp4', '.webm', '.mkv', '.png', '.json'].includes(extension);
      const evaluation = resource.startsWith('/diagnostics/video-sequences/') && ['.png', '.json'].includes(extension);
      if ((!publicAsset && !media && !evaluation) || resource.split(/[\\/]/).some(part => part.startsWith('.'))) { res.writeHead(403).end(); return; }
      const file = await realpath(path.resolve(root, '.' + resource));
      const actualRoot = await realpath(root);
      if (!file.toLowerCase().startsWith((actualRoot + path.sep).toLowerCase())) { res.writeHead(403).end(); return; }
      const info = await stat(file);
      if (!info.isFile()) { res.writeHead(404).end(); return; }
      let start = 0, end = info.size - 1, status = 200;
      const headers = { 'Content-Type': types[extension] || 'application/octet-stream', 'Cache-Control': 'no-store', 'Accept-Ranges': 'bytes' };
      if (req.headers.range) {
        const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
        if (!range || (!range[1] && !range[2])) { res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }).end(); return; }
        if (!range[1]) start = Math.max(0, info.size - Number(range[2]));
        else { start = Number(range[1]); if (range[2]) end = Math.min(end, Number(range[2])); }
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= info.size) { res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }).end(); return; }
        status = 206; headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
      }
      headers['Content-Length'] = Math.max(0, end - start + 1);
      res.writeHead(status, headers);
      if (req.method === 'HEAD' || info.size === 0) { res.end(); return; }
      const stream = createReadStream(file, { start, end });
      stream.on('error', () => res.destroy());
      res.on('close', () => stream.destroy());
      stream.pipe(res);
    } catch { if (!res.headersSent) res.writeHead(404).end(); else res.destroy(); }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const port = Number(process.env.ANIMESCALER_DIAGNOSTIC_PORT || 8765);
  createDiagnosticServer().listen(port, '127.0.0.1', () => console.log(`Benchmark: http://127.0.0.1:${port}/diagnostics/performance.html`));
}
