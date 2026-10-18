import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createDiagnosticServer } from '../scripts/serve-diagnostics.mjs';

test('diagnostic server protects both native credentials and supports bounded media ranges', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'animescaler-http-'));
  for (const folder of ['native', 'diagnostics/stability-media', 'src']) await mkdir(path.join(root, folder), { recursive: true });
  await writeFile(path.join(root, 'native/connection.json'), '{"url":"http://127.0.0.1:8766","token":"private-test"}');
  await writeFile(path.join(root, 'native/fsr3-connection.json'), '{"token":"other-private-test"}');
  await writeFile(path.join(root, 'diagnostics/stability-media/test.mp4'), Buffer.from('0123456789'));
  await writeFile(path.join(root, 'src/example.js'), 'export const value = 1;');
  const server = createDiagnosticServer({ root });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    for (const resource of ['/native/connection.json', '/native/fsr3-connection.json', '/.git/config', '/diagnostics/../native/fsr3-connection.json']) {
      const response = await fetch(base + resource);
      assert.equal(response.status, 403);
      assert.equal((await response.text()).includes('private-test'), false);
    }
    assert.equal((await fetch(base + '/diagnostics/native-config')).status, 403);
    const config = await fetch(base + '/diagnostics/native-config', { headers: { 'sec-fetch-site': 'same-origin' } });
    assert.equal(config.status, 200);
    assert.equal((await config.json()).token, 'private-test');
    const range = await fetch(base + '/diagnostics/stability-media/test.mp4', { headers: { Range: 'bytes=2-5' } });
    assert.equal(range.status, 206);
    assert.equal(range.headers.get('content-range'), 'bytes 2-5/10');
    assert.equal(await range.text(), '2345');
    const suffix = await fetch(base + '/diagnostics/stability-media/test.mp4', { headers: { Range: 'bytes=-3' } });
    assert.equal(await suffix.text(), '789');
    assert.equal((await fetch(base + '/diagnostics/stability-media/test.mp4', { headers: { Range: 'bytes=100-' } })).status, 416);
    assert.equal((await fetch(base + '/src/example.js')).status, 200);
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  }
});
