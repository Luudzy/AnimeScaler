import test from 'node:test';
import assert from 'node:assert/strict';
import { startManualViewer } from '../src/core/start-viewer.js';

function fixture() {
  const updates = new Set(), removed = new Set(), messages = new Set(), saved = {};
  let loaded = false, captures = 0, closed = false, created = 0;
  const url = 'chrome-extension://test/viewer/viewer.html?session=test';
  const api = {
    runtime: {id:'test',getURL: () => url,onMessage:{addListener:fn=>messages.add(fn),removeListener:fn=>messages.delete(fn)}},
    windows: {
      create: async () => { created++; return {id: 7, tabs: [{id: 88}]}; },
      get: async id => ({id, state:'normal'}),
      update: async () => { assert.ok(Object.keys(saved).length, 'store session before focusing'); },
      remove: async () => { closed = true; },
    },
    tabs: {
      get: async () => ({id: 88, url: loaded ? url : '', status: loaded ? 'complete' : 'loading'}),
      onUpdated: {addListener: fn => updates.add(fn), removeListener: fn => updates.delete(fn)},
      onRemoved: {addListener: fn => removed.add(fn), removeListener: fn => removed.delete(fn)},
    },
    tabCapture: {getMediaStreamId: async options => {
      captures++;
      assert.deepEqual(options, {targetTabId: 12, consumerTabId: 88});
      if (!loaded) throw Error('Caller tab.url is not a valid URL.');
      return 'stream';
    }},
    storage: {session: {get: async () => structuredClone(saved), set: async data => Object.assign(saved, structuredClone(data)), remove: async keys => keys.forEach(key => delete saved[key])}},
  };
  return {api, saved, updates, removed, messages, get captures() {return captures;}, get created() {return created;}, get closed() {return closed;},
    load() { loaded = true; for (const fn of updates) fn(88, {status:'complete'}, {id:88,url,status:'complete'}); },
  };
}
const frames = [{frameId:0,parentFrameId:-1,url:'https://anime.test'}, {frameId:3,parentFrameId:0,url:'https://www.blogger.com/video.g'}];
const start = f => startManualViewer({id:12,windowId:4}, {frameId:3}, {}, frames, f.api);

test('viewer capture waits for the consumer URL to commit before requesting a stream', async () => {
  const f = fixture();
  const pending = start(f);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.captures, 0);
  f.load();
  const result = await pending;
  assert.equal(result.state, 'running');
  assert.equal(f.captures, 1);
  assert.equal(f.updates.size, 0);
  assert.equal(f.removed.size, 0);
  assert.equal(f.closed, false);
  assert.ok(Object.values(f.saved).some(value => value.streamId === 'stream'));
});

test('closing the viewer during navigation aborts without requesting capture', async () => {
  const f = fixture();
  const pending = start(f);
  await new Promise(resolve => setImmediate(resolve));
  for (const fn of f.removed) fn(88);
  const result = await pending;
  assert.equal(result.state, 'error');
  assert.equal(f.captures, 0);
  assert.equal(f.updates.size, 0);
});

test('already loaded viewer starts without waiting for another navigation event', async () => {
  const f = fixture();
  f.load();
  assert.equal((await start(f)).state, 'running');
  assert.equal(f.captures, 1);
  assert.equal(f.updates.size, 0);
});

test('starting an already active native route focuses its viewer instead of opening a duplicate', async () => {
  const f = fixture();
  const key = 'animeScalerViewerActive:12';
  f.saved[key] = { sessionId: 'existing', viewerWindowId: 7, viewerTabId: 88 };
  f.load();
  const result = await start(f);
  assert.equal(result.state, 'running');
  assert.equal(result.viewer.sessionId, 'existing');
  assert.equal(f.captures, 0);
  assert.equal(f.closed, false);
});

test('concurrent starts for one source tab share one viewer session', async () => {
  const f = fixture();
  const first = start(f);
  const second = start(f);
  await new Promise(resolve => setImmediate(resolve));
  f.load();
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a.state, 'running');
  assert.equal(b.state, 'running');
  assert.equal(f.created, 1);
  assert.equal(f.captures, 1);
  assert.equal(a.viewer.sessionId, b.viewer.sessionId);
});

test('trusted viewer readiness works when Chrome hides its extension tab URL', async t => {
  const f=fixture(); f.load();
  const expected=f.api.runtime.getURL();
  f.api.tabs.get=async()=>({id:88,status:'complete'});
  f.api.webNavigation={getFrame:async()=>null};
  t.after(()=>{for(const fn of f.removed)fn(88);});
  const pending=start(f);
  await new Promise(resolve=>setImmediate(resolve));
  for(const fn of f.messages)fn({type:'ANIMESCALER_VIEWER_READY'},{id:'test',tab:{id:88},frameId:0,url:expected},()=>{});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.captures,1,'a loaded consumer without tabs.url must not wait until viewer session timeout');
  assert.equal((await pending).state,'running');
  assert.equal(f.messages.size,0,'readiness listener is released after startup');
});

test('readiness cannot authorize another extension, URL, tab or subframe',async()=>{
  const f=fixture();f.load();f.api.tabs.get=async()=>({id:88,status:'complete'});
  const pending=start(f);await new Promise(resolve=>setImmediate(resolve));
  const valid={id:'test',tab:{id:88},frameId:0,url:f.api.runtime.getURL()};
  for(const sender of [{...valid,id:'other'},{...valid,url:'https://untrusted.test'},{...valid,tab:{id:99}},{...valid,frameId:3}]) {
    for(const fn of f.messages)fn({type:'ANIMESCALER_VIEWER_READY'},sender,()=>{});
  }
  assert.equal(f.captures,0);
  for(const fn of f.removed)fn(88);
  assert.equal((await pending).state,'error');
});

test('capture failure closes the viewer and removes session state', async () => {
  const f = fixture();
  f.load();
  f.api.tabCapture.getMediaStreamId = async () => { throw Error('capture denied'); };
  const result = await start(f);
  assert.equal(result.error.message, 'capture denied');
  assert.equal(f.closed, true);
  assert.deepEqual(f.saved, {});
});

test('exclusive viewer expands every ancestor iframe and restores them when capture fails', async () => {
  const f = fixture(); f.load();
  const calls = [];
  f.api.tabs.sendMessage = async (tab, message, options) => { calls.push({tab, message, options}); return {ok:true}; };
  f.api.tabCapture.getMediaStreamId = async () => { throw Error('capture denied'); };
  const nested = [...frames, {frameId:9,parentFrameId:3,url:'https://player.test'}];
  await startManualViewer({id:12,windowId:4}, {frameId:9}, {exclusiveViewer:true}, nested, f.api);
  const enter = calls.filter(c=>c.message.type==='ANIMESCALER_EXCLUSIVE_ENTER');
  const leave = calls.filter(c=>c.message.type==='ANIMESCALER_EXCLUSIVE_EXIT');
  assert.deepEqual(enter.map(c=>c.options.frameId), [0,3,9]);
  assert.equal(enter[0].message.captureFrame.frameId, 3);
  assert.equal(enter[1].message.captureFrame.frameId, 9);
  assert.equal(enter[2].message.captureFrame.frameId, 0);
  assert.deepEqual(leave.map(c=>c.options.frameId).sort((a,b)=>a-b), [0,3,9]);
});

test('exclusive viewer requests its own fullscreen window after session is saved', async () => {
  const f = fixture(); f.load();
  const windowUpdates = [];
  f.api.tabs.sendMessage = async () => ({ok:true});
  f.api.windows.update = async (id, options) => { assert.ok(Object.keys(f.saved).length); windowUpdates.push(options); };
  const result = await startManualViewer({id:12,windowId:4},{frameId:3},{exclusiveViewer:true},frames,f.api);
  assert.equal(result.state,'running');
  assert.equal(windowUpdates.at(-1).state,'fullscreen');
  assert.ok(Object.values(f.saved).some(v=>v.exclusiveFrames?.length===2));
});

test('service worker handles viewer startup independently of the popup response', async () => {
  const f = fixture();
  const original = globalThis.chrome;
  f.api.runtime.id = 'test';
  const viewerUrl = f.api.runtime.getURL();
  f.api.runtime.getURL = path => path === 'popup/popup.html' ? 'chrome-extension://test/popup/popup.html' : viewerUrl;
  f.api.runtime.onMessage = {addListener: fn => f.messages.add(fn),removeListener:fn=>f.messages.delete(fn)};
  f.api.runtime.onConnect = {addListener() {}};
  f.api.runtime.onInstalled = {addListener() {}};
  globalThis.chrome = f.api;
  try {
    await import('../background.js');
    const dispatch = (message, sender, respond) => {
      for (const listener of f.messages) {
        const result = listener(message, sender, respond);
        if (result !== undefined) return result;
      }
    };
    const message = {type:'ANIMESCALER_START_VIEWER', tab:{id:12,windowId:4}, target:{frameId:3}, settings:{}, frames};
    assert.equal(dispatch(message, {id:'test',url:'https://untrusted.test'}, () => {}), undefined);
    let respond;
    const result = new Promise(resolve => {respond = resolve;});
    assert.equal(dispatch(message, {id:'test',url:'chrome-extension://test/popup/popup.html'}, respond), true);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(f.captures, 0);
    f.load();
    assert.equal((await result).state, 'running');

    const popupSender = { id: 'test', url: 'chrome-extension://test/popup/popup.html' };
    const viewerStatus = new Promise(resolve => {
      assert.equal(dispatch({type:'ANIMESCALER_VIEWER_STATUS',sourceTabId:12}, popupSender, resolve), true);
    });
    assert.equal((await viewerStatus).captureMode, 'viewer');
    const viewerStop = new Promise(resolve => {
      assert.equal(dispatch({type:'ANIMESCALER_STOP_VIEWER',sourceTabId:12}, popupSender, resolve), true);
    });
    assert.equal((await viewerStop).state, 'stopped');
    assert.equal(f.closed, true);
    assert.deepEqual(f.saved, {});
  } finally { globalThis.chrome = original; }
});
