import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', root), 'utf8'));

test('manifest declares MV3 module background and existing popup/content entry points', async () => {
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.background.type, 'module');
  assert.equal(manifest.background.service_worker, 'background.js');
  assert.equal(manifest.action.default_popup, 'popup/popup.html');
  await access(new URL(manifest.background.service_worker, root));
  await access(new URL(manifest.action.default_popup, root));
  const content = manifest.content_scripts.find(script => script.js.includes('content-bootstrap.js'));
  assert.ok(content, 'content bootstrap must be registered');
  assert.equal(content.all_frames, true);
  assert.equal(content.world ?? 'ISOLATED', 'ISOLATED', 'bootstrap needs extension APIs, not the page MAIN world');
  assert.equal(content.match_origin_as_fallback, true);
  assert.ok(content.matches.includes('http://*/*'), 'loopback fixture requires HTTP content matching');
  for (const file of content.js) await access(new URL(file, root));
  assert.ok(manifest.permissions.includes('storage'));
  assert.ok(manifest.permissions.includes('webNavigation'));
});

test('web-accessible resources expose content modules but never private connection JSON', () => {
  const resources = manifest.web_accessible_resources.flatMap(entry => entry.resources);
  assert.ok(resources.includes('src/content/*'));
  assert.ok(resources.includes('src/engine/*'));
  for (const secret of ['native/connection.json', 'native/fsr3-connection.json']) {
    for (const pattern of resources) {
      const expression = new RegExp(`^${pattern.split('*').map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`);
      assert.equal(expression.test(secret), false, `private config exposed by ${pattern}`);
    }
  }
});
