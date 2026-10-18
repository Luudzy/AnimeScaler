import test from 'node:test';
import assert from 'node:assert/strict';
import { sessionTag, isStaleOrigin } from '../src/core/session-diagnostics.js';

test('session tags are stable, bounded, and do not expose the protocol ID', () => {
  const id = '1b31dbd7-2acf-4e57-bff0-9a2b53b02ba1';
  const tag = sessionTag(id);
  assert.match(tag, /^[a-f0-9]{16}$/);
  assert.equal(sessionTag(id), tag);
  assert.notEqual(tag, id);
  assert.equal(sessionTag(''), null);
});

test('stale origin accepts only known component labels', () => {
  assert.equal(isStaleOrigin('native_client'), true);
  assert.equal(isStaleOrigin('arbitrary payload'), false);
});
