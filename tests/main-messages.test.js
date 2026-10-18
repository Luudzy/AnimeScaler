import test from 'node:test';
import assert from 'node:assert/strict';
import { handleAnimeScalerMessage } from '../src/content/main.js';

test('routes tab capture start messages to controller.startTabCapture', async () => {
  let payload;
  const controller = {
    async startTabCapture(value) { payload = value; return { state: 'running', captureMode: 'tab' }; },
  };
  const message = {
    type: 'ANIMESCALER_START_TAB_CAPTURE',
    streamId: 'stream-id',
    captureFrame: { frameId: 3, url: 'https://player.test/' },
    settings: { enabled: true },
  };

  const result = await handleAnimeScalerMessage(controller, message);
  assert.equal(result.captureMode, 'tab');
  assert.deepEqual(payload, {
    streamId: 'stream-id',
    captureFrame: message.captureFrame,
    settings: message.settings,
  });
});

test('routes capture geometry requests for the external viewer', async () => {
  let payload;
  const controller = {
    getCaptureGeometry(value) { payload = value; return { rect: { left: 1, top: 2, width: 3, height: 4 } }; },
  };
  const captureFrame = { frameId: 2, url: 'https://player.test/embed' };
  const result = await handleAnimeScalerMessage(controller, { type: 'ANIMESCALER_CAPTURE_GEOMETRY', captureFrame });
  assert.equal(payload, captureFrame);
  assert.equal(result.rect.width, 3);
});

test('routes viewer playback commands to the source video frame', async () => {
  let command;
  const controller = {
    async playerCommand(value) { command = value; return { ok: true }; },
  };
  const result = await handleAnimeScalerMessage(controller, { type: 'ANIMESCALER_PLAYER_COMMAND', command: { type: 'seek', seconds: 10 } });
  assert.deepEqual(command, { type: 'seek', seconds: 10 });
  assert.equal(result.ok, true);
});

test('viewer geometry and commands forward the same session identity',async()=>{
  const sessions=[];
  const controller={getCaptureGeometry(frame,sessionId){sessions.push(sessionId);return {};},
    playerCommand(command,sessionId){sessions.push(sessionId);return {ok:true};}};
  await handleAnimeScalerMessage(controller,{type:'ANIMESCALER_CAPTURE_GEOMETRY',captureFrame:{frameId:0},sessionId:'viewer-a'});
  await handleAnimeScalerMessage(controller,{type:'ANIMESCALER_PLAYER_COMMAND',command:{type:'toggle'},sessionId:'viewer-a'});
  assert.deepEqual(sessions,['viewer-a','viewer-a']);
});
