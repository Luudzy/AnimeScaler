import test from 'node:test';
import assert from 'node:assert/strict';
import { handleViewerKeydown } from '../viewer/viewer-hotkeys.js';

function key(code, { button = false, input = false, ...overrides } = {}) {
  let prevented = false;
  const event = {
    code, repeat: false, ctrlKey: false, altKey: false, metaKey: false,
    target: {
      isContentEditable: false,
      matches: () => input,
      closest: selector => selector === 'button' && button ? {} : null,
    },
    preventDefault() { prevented = true; },
    ...overrides,
  };
  return { event, prevented: () => prevented };
}

test('Space toggles source playback only when focus is outside buttons and inputs', () => {
  let toggles = 0;
  const actions = { togglePlayback: () => toggles++ };
  const pageSpace = key('Space');
  assert.equal(handleViewerKeydown(pageSpace.event, {}, actions), true);
  assert.equal(pageSpace.prevented(), true);
  assert.equal(toggles, 1);

  const buttonSpace = key('Space', { button: true });
  assert.equal(handleViewerKeydown(buttonSpace.event, {}, actions), false);
  assert.equal(buttonSpace.prevented(), false);
  assert.equal(toggles, 1);

  const inputSpace = key('Space', { input: true });
  assert.equal(handleViewerKeydown(inputSpace.event, {}, actions), false);
  assert.equal(toggles, 1);
});

test('Escape restores comparison first, then exits fullscreen', () => {
  const calls = [];
  const actions = { closeComparison: () => calls.push('comparison'), exitFullscreen: () => calls.push('fullscreen') };
  const comparisonEscape = key('Escape');
  assert.equal(handleViewerKeydown(comparisonEscape.event, { comparing: true }, actions), true);
  assert.equal(comparisonEscape.prevented(), true);
  assert.deepEqual(calls, ['comparison']);

  const fullscreenEscape = key('Escape');
  handleViewerKeydown(fullscreenEscape.event, { comparing: false }, actions);
  assert.deepEqual(calls, ['comparison', 'fullscreen']);
});

test('viewer shortcuts ignore repeats, modifiers, and editable targets', () => {
  let calls = 0;
  const actions = { togglePlayback: () => calls++ };
  for (const event of [
    key('Space', { repeat: true }).event,
    key('Space', { ctrlKey: true }).event,
    key('Space', { target: { isContentEditable: true, matches: () => false, closest: () => null } }).event,
  ]) {
    assert.equal(handleViewerKeydown(event, {}, actions), false);
  }
  assert.equal(calls, 0);
});

test('arrows seek ten seconds, but comparison blocks playback controls', () => {
  const calls = [];
  const actions = { seek: seconds => calls.push(seconds), togglePlayback: () => calls.push('play') };
  const left = key('ArrowLeft');
  handleViewerKeydown(left.event, {}, actions);
  const right = key('ArrowRight');
  handleViewerKeydown(right.event, {}, actions);
  const blocked = key('ArrowRight');
  handleViewerKeydown(blocked.event, { comparing: true }, actions);
  assert.deepEqual(calls, [-10, 10]);
  assert.equal(blocked.prevented(), true);
});
