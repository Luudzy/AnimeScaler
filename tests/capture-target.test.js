import test from 'node:test';
import assert from 'node:assert/strict';
import { findCaptureTargetElement, scoreFrameElement } from '../src/content/capture-target.js';

function frame({ src, width, height, visible = true }) {
  return {
    src,
    isConnected: true,
    getBoundingClientRect() {
      return { width, height, top: visible ? 20 : -5000, left: 20, bottom: visible ? 20 + height : -4900, right: 20 + width };
    },
  };
}

test('prefers iframe whose src matches the outer capture frame URL', () => {
  const ad = frame({ src: 'https://ads.example/banner', width: 1200, height: 900 });
  const player = frame({ src: 'https://player.example/embed/abc#autoplay', width: 900, height: 506 });
  const root = { querySelectorAll: () => [ad, player] };

  const selected = findCaptureTargetElement(root, {
    frameId: 3,
    url: 'https://player.example/embed/abc',
    viewport: { width: 1920, height: 1080 },
  });

  assert.equal(selected, player);
});

test('falls back to the largest visible iframe when frame URL cannot be matched', () => {
  const tiny = frame({ src: 'about:blank', width: 300, height: 250 });
  const player = frame({ src: 'about:blank', width: 1000, height: 562 });
  const root = { querySelectorAll: () => [tiny, player] };

  assert.equal(findCaptureTargetElement(root, {
    frameId: 5,
    url: 'about:blank',
    viewport: { width: 1920, height: 1080 },
  }), player);
});

test('frame scoring rejects invisible embeds', () => {
  assert.equal(scoreFrameElement(frame({ src: 'https://x.test', width: 1000, height: 600, visible: false }), {
    url: 'https://x.test',
    viewport: { width: 1920, height: 1080 },
  }), -1);
});

test('describes the visible capture target geometry for manual viewer cropping', async () => {
  const { describeCaptureTarget } = await import('../src/content/capture-target.js');
  const iframe = {
    src: 'https://player.example/embed/abc',
    isConnected: true,
    getBoundingClientRect() {
      return { left: 120, top: 80, width: 960, height: 540, right: 1080, bottom: 620 };
    },
  };
  const root = { querySelectorAll() { return [iframe]; } };
  const result = describeCaptureTarget(root, {
    frameId: 4,
    url: 'https://player.example/embed/abc',
    viewport: { width: 1280, height: 720 },
  });

  assert.deepEqual(result, {
    rect: { left: 120, top: 80, width: 960, height: 540 },
    viewport: { width: 1280, height: 720 },
  });
});
