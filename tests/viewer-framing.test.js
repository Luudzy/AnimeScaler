import test from 'node:test';
import assert from 'node:assert/strict';
import * as geometry from '../src/core/capture-geometry.js';

test('contained 16:9 content excludes bars in a 1427 by 901 player', () => {
  const rect = geometry.containedVideoRect({left:0,top:0,width:1427,height:901},1920,1080);
  assert.equal(rect.width,1427);
  assert.equal(rect.height,802.6875);
  assert.equal(rect.top,49.15625);
});

test('nested frame geometry maps actual video content through each viewport', () => {
  const result = geometry.composeCaptureGeometry([
    {rect:{left:100,top:50,width:800,height:600},viewport:{width:1200,height:900}},
    {rect:{left:0,top:0,width:400,height:300},viewport:{width:400,height:300}},
    {rect:{left:0,top:37.5,width:400,height:225},viewport:{width:400,height:300}},
  ]);
  assert.deepEqual(result,{rect:{left:100,top:125,width:800,height:450},viewport:{width:1200,height:900}});
});

test('partially hidden video is rejected instead of changing its aspect ratio', () => {
  assert.throws(() => geometry.composeCaptureGeometry([
    {rect:{left:0,top:-20,width:1280,height:720},viewport:{width:1280,height:720}},
  ]), /CAPTURE_TARGET_CLIPPED/);
});
