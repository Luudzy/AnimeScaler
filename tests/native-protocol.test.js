import test from 'node:test';
import assert from 'node:assert/strict';
import { HEADER_BYTES, writeHeader, readResponse, jpegResponse, rgbResponse } from '../src/engine/native-protocol.js';
import { NativeEngine } from '../src/engine/native-engine.js';
import { normalizeSettings } from '../src/core/settings.js';

test('RGB response preserves color bytes and rejects wrong payload sizes',()=>{
  const buffer=new ArrayBuffer(24+80*80*3);
  writeHeader(buffer,1,80,80,80,80);new DataView(buffer).setUint32(0,0x33435341,true);
  new Uint8Array(buffer,24).set([19,61,179]);
  assert.deepEqual([...rgbResponse(buffer,1,80,80).subarray(0,3)],[19,61,179]);
  assert.throws(()=>rgbResponse(buffer.slice(0,-1),1,80,80));
  assert.throws(()=>rgbResponse(buffer,2,80,80));
});

test('compressed response validates frame identity and JPEG framing',()=>{
  const buffer=new ArrayBuffer(28);
  writeHeader(buffer,1,80,80,80,80);
  new DataView(buffer).setUint32(0,0x32435341,true);
  new Uint8Array(buffer,24).set([255,216,255,217]);
  assert.equal(jpegResponse(buffer,1,80,80).type,'image/jpeg');
  assert.throws(()=>jpegResponse(buffer,2,80,80));
  new Uint8Array(buffer)[27]=0;
  assert.throws(()=>jpegResponse(buffer,1,80,80));
});

test('native geometry changes invalidate calibration identity and samples',async()=>{
  const engine=new NativeEngine();
  engine.settings=normalizeSettings({preset:'720p',backend:'rocm'});
  await engine.reconfigure({videoWidth:1280,videoHeight:720});
  const key=engine.profileId;engine.profileSamples.record(10,0,30);
  await engine.reconfigure({videoWidth:640,videoHeight:360});
  assert.notEqual(engine.profileId,key);
  assert.equal(engine.profileSamples.level,undefined);
  await engine.destroy();
});

test('native response must match frame id, dimensions and exact RGBA length', () => {
  const buffer = new ArrayBuffer(HEADER_BYTES + 80 * 80 * 4);
  writeHeader(buffer, 42, 80, 80, 80, 80);
  assert.equal(readResponse(buffer, 42, 80, 80).length, 80 * 80 * 4);
  assert.throws(() => readResponse(buffer, 41, 80, 80));
  assert.throws(() => readResponse(buffer, 42, 40, 160));
  assert.throws(() => readResponse(buffer.slice(1), 42, 80, 80));
});

test('native preset scales output on GPU and preserves adaptive dimensions', async () => {
  const engine = new NativeEngine();
  engine.settings = normalizeSettings({ model: 'adore', backend: 'rocm', preset: '1080p' });
  const video = { videoWidth: 1920, videoHeight: 1080 };
  await engine.reconfigure(video);
  // The named 1080p path is 540p -> 1080p. The output remains full HD;
  // sending 960x540 keeps the native worker within its intended x2 geometry.
  assert.equal(engine.currentSize.inputWidth, 960);
  assert.equal(engine.currentSize.outputWidth, 1920);
  assert.equal(engine.currentSize.outputHeight, 1080);
  assert.equal(engine.needsReconfigure(video, 0), false);
  assert.equal(engine.needsReconfigure(video, 1), true);
  await engine.reconfigure(video, 2);
  assert.equal(engine.currentSize.inputWidth, 480);
  assert.equal(engine.currentSize.outputWidth, 1920);
  assert.equal(engine.currentSize.outputHeight, 1080);
  await assert.rejects(engine.reconfigure({videoWidth: 20, videoHeight: 20}));
  await engine.destroy();
  await assert.rejects(engine.reconfigure(video));
});

test('Anime4KCPP 4K preset uses a native 2x input instead of a costly final resize', async () => {
  const engine = new NativeEngine();
  engine.settings = normalizeSettings({ model: 'anime4kcpp-acnet-f8b4', backend: 'rocm', preset: '4k' });
  await engine.reconfigure({ videoWidth: 1411, videoHeight: 890 });
  assert.deepEqual(
    { width: engine.currentSize.inputWidth, height: engine.currentSize.inputHeight },
    { width: 1712, height: 1080 },
  );
  assert.deepEqual(
    { width: engine.currentSize.outputWidth, height: engine.currentSize.outputHeight },
    { width: 3424, height: 2160 },
  );
});
import { repeatResponse } from '../src/engine/native-protocol.js';

test('repeat responses require exact identity, dimensions and header-only payload',()=>{
  const buffer=new ArrayBuffer(24),view=new DataView(buffer);
  [0x34435341,9,3840,2160,3840,2160].forEach((v,i)=>view.setUint32(i*4,v,true));
  assert.equal(repeatResponse(buffer,9,3840,2160),true);
  assert.throws(()=>repeatResponse(buffer,10,3840,2160));
  assert.throws(()=>repeatResponse(buffer,9,2560,1440));
  const extra=new Uint8Array(25);extra.set(new Uint8Array(buffer));
  assert.throws(()=>repeatResponse(extra.buffer,9,3840,2160));
  assert.equal(repeatResponse(new ArrayBuffer(0),9,3840,2160),false);
});
