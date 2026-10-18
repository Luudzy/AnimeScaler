import test from 'node:test';
import assert from 'node:assert/strict';
import { GpuPrePost } from '../src/engine/gpu-prepost.js';

test('uploadVideoFrame copies only the requested source rectangle', () => {
  let call;
  const gpu = Object.create(GpuPrePost.prototype);
  gpu.config = { sourceWidth: 1280, sourceHeight: 720 };
  gpu.sourceTexture = { id: 'texture' };
  gpu.preBindGroup={};
  gpu.device = {
    queue: {
      copyExternalImageToTexture(source, destination, size) { call = { source, destination, size }; },
    },
  };
  const video = { tag: 'video' };

  gpu.uploadVideoFrame(video, { x: 120, y: 80, width: 1280, height: 720 });

  assert.equal(call.source.source, video);
  assert.deepEqual(call.source.origin, { x: 120, y: 80 });
  assert.deepEqual(call.size, [1280, 720]);
  assert.equal(call.destination.texture, gpu.sourceTexture);
});

test('uploadVideoFrame defaults to the full configured source', () => {
  let sourceArg;
  const gpu = Object.create(GpuPrePost.prototype);
  gpu.config = { sourceWidth: 640, sourceHeight: 360 };
  gpu.sourceTexture = {};
  gpu.preBindGroup={};
  gpu.device = { queue: { copyExternalImageToTexture(source) { sourceArg = source; } } };

  gpu.uploadVideoFrame({});
  assert.equal('origin' in sourceArg, false);
});
