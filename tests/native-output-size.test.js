import test from 'node:test';
import assert from 'node:assert/strict';
import {resolveOutputSize, resolveInputSize, normalizeSettings} from '../src/core/settings.js';

test('1440p and 4K presets select bounded 2x inputs and retain fixed output under adaptive',()=>{
  for(const preset of ['1440p','4k'])assert.equal(normalizeSettings({preset}).preset,preset);
  assert.deepEqual(resolveInputSize(1920,1080,'1440p'),{width:1280,height:720});
  assert.deepEqual(resolveInputSize(3840,2160,'4k'),{width:1920,height:1080});
  assert.deepEqual(resolveInputSize(1280,720,'4k'),{width:1920,height:1080});
  assert.deepEqual(resolveOutputSize(1920,1080,'1440p',640,360),{width:2560,height:1440});
  assert.deepEqual(resolveOutputSize(1920,1080,'4k',960,540),{width:3840,height:2160});
  const wide=resolveOutputSize(2560,1080,'4k',1280,540);
  assert.deepEqual(wide,{width:3840,height:1620});
});

test('target output survives reduced AI input and preserves portrait aspect',()=>{
  assert.deepEqual(resolveOutputSize(1280,720,'1080p',640,360),{width:1920,height:1080});
  assert.deepEqual(resolveOutputSize(640,360,'720p',320,180),{width:1280,height:720});
  assert.deepEqual(resolveOutputSize(720,1280,'1080p',360,640),{width:608,height:1080});
  assert.equal(normalizeSettings({backend:'tensorrt'}).backend,'tensorrt');
});

test('input geometry stays inside native axis and pixel limits for portrait, square, ultrawide and small sources',()=>{
  const limit = 1920 * 1080;
  for (const [sourceWidth, sourceHeight, preset] of [
    [1080, 1920, '4k'], [2048, 2048, 'quality'], [3440, 1440, '4k'], [320, 180, '4k'],
  ]) {
    const input = resolveInputSize(sourceWidth, sourceHeight, preset);
    assert.ok(input.width > 0 && input.height > 0);
    assert.ok(input.width <= 1920 && input.height <= 1920);
    assert.ok(input.width * input.height <= limit);
    assert.equal(input.width % 4, 0);
    assert.equal(input.height % 4, 0);
    assert.ok(Math.abs((input.width / input.height) / (sourceWidth / sourceHeight) - 1) < 0.01);
  }
});
