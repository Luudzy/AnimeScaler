import test from 'node:test';
import assert from 'node:assert/strict';
import { TabCaptureSource } from '../src/content/tab-capture-source.js';

function makeVideo() {
  return {
    muted: false,
    autoplay: false,
    playsInline: false,
    srcObject: null,
    videoWidth: 1280,
    videoHeight: 720,
    readyState: 4,
    async play() {},
    pause() {},
    addEventListener() {},
    removeEventListener() {},
  };
}

test('tab capture source consumes the stream and restricts it to the player element', async () => {
  let constraints;
  let restrictedWith;
  const track = { async restrictTo(target) { restrictedWith = target; }, stop() {} };
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
  const target = { style: { isolation: '' } };
  const token = { id: 'restriction' };
  const video = makeVideo();
  const source = new TabCaptureSource({
    mediaDevices: { async getUserMedia(value) { constraints = value; return stream; } },
    restrictionTargetApi: { async fromElement(element) { assert.equal(element, target); return token; } },
    documentRef: { createElement(tag) { assert.equal(tag, 'video'); return video; } },
  });

  const result = await source.start({ streamId: 'abc', targetElement: target });

  assert.equal(result, video);
  assert.equal(video.srcObject, stream);
  assert.equal(restrictedWith, token);
  assert.equal(target.style.isolation, 'isolate');
  assert.equal(constraints.audio, false);
  assert.equal(constraints.video.mandatory.chromeMediaSourceId, 'abc');
});

test('destroy stops tab capture tracks and restores target isolation', async () => {
  let stopped = 0;
  const track = { async restrictTo() {}, stop() { stopped += 1; } };
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
  const target = { style: { isolation: 'auto' } };
  const video = makeVideo();
  const source = new TabCaptureSource({
    mediaDevices: { async getUserMedia() { return stream; } },
    restrictionTargetApi: { async fromElement() { return {}; } },
    documentRef: { createElement() { return video; } },
  });

  await source.start({ streamId: 'abc', targetElement: target });
  source.destroy();

  assert.equal(stopped, 1);
  assert.equal(target.style.isolation, 'auto');
  assert.equal(video.srcObject, null);
});

test('tab capture reports Element Capture unavailable without opening a viewer', async () => {
  const source = new TabCaptureSource({
    mediaDevices: { async getUserMedia() { throw new Error('must not be called'); } },
    restrictionTargetApi: null,
    documentRef: { createElement() { return makeVideo(); } },
  });

  await assert.rejects(
    () => source.start({ streamId: 'abc', targetElement: { style: {} } }),
    (error) => error.code === 'ELEMENT_CAPTURE_UNAVAILABLE',
  );
});

test('tab capture reports missing restrictTo and stops its stream', async () => {
  let stopped = 0;
  const track = { stop() {} };
  track.stop = () => { stopped += 1; };
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
  const source = new TabCaptureSource({
    mediaDevices: { async getUserMedia() { return stream; } },
    restrictionTargetApi: { async fromElement() { return {}; } },
    documentRef: { createElement() { return makeVideo(); } },
  });

  await assert.rejects(
    () => source.start({ streamId: 'abc', targetElement: { style: {} } }),
    (error) => error.code === 'ELEMENT_CAPTURE_UNAVAILABLE',
  );
  assert.equal(stopped, 1);
});

test('tab capture reports failed isolation and does not create a source video', async () => {
  let createdVideo = false;
  const track = { async restrictTo() { throw new DOMException('not supported', 'NotSupportedError'); }, stop() {} };
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
  const source = new TabCaptureSource({
    mediaDevices: { async getUserMedia() { return stream; } },
    restrictionTargetApi: { async fromElement() { return {}; } },
    documentRef: { createElement() { createdVideo = true; return makeVideo(); } },
  });

  await assert.rejects(
    () => source.start({ streamId: 'abc', targetElement: { style: {} } }),
    (error) => error.code === 'SOURCE_NOT_ISOLATED',
  );
  assert.equal(createdVideo, false);
});

test('tab capture waits for restriction before creating or consuming the captured video', async () => {
  let releaseRestriction;
  let createdVideo = false;
  const track = { restrictTo() { return new Promise(resolve => { releaseRestriction = resolve; }); }, stop() {} };
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
  const source = new TabCaptureSource({
    mediaDevices: { async getUserMedia() { return stream; } },
    restrictionTargetApi: { async fromElement() { return {}; } },
    documentRef: { createElement() { createdVideo = true; return makeVideo(); } },
  });

  const starting = source.start({ streamId: 'abc', targetElement: { style: {} } });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(createdVideo, false);
  releaseRestriction();
  await starting;
  assert.equal(createdVideo, true);
});
