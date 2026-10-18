import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import {
  classifyNativeServiceClose,
  classifyNativeServiceError,
  nativeServiceFailure,
  validateNativeService,
} from '../src/engine/native-service-contract.js';

test('native worker replaces private service errors with stable backend codes', async () => {
  const source = (await readFile(new URL('../src/engine/native-worker.js', import.meta.url), 'utf8'))
    .replace(/^import .*;\r?\n/gm, '');
  const messages = [];
  const token = 'a'.repeat(64);
  class FakeWebSocket {
    constructor() { queueMicrotask(() => this.onopen?.()); }
    send(value) {
      if (value === token) queueMicrotask(() => this.onmessage?.({
        data: JSON.stringify({ error: 'C:\\Users\\private\\token=secret' }),
      }));
    }
    close() {}
  }
  const scope = {
    self: { postMessage: message => messages.push(message) },
    WebSocket: FakeWebSocket,
    setTimeout,
    clearTimeout,
    createNativeGpuRenderer: async () => null,
    validateNativeService,
    nativeServiceFailure,
    classifyNativeServiceError,
    classifyNativeServiceClose,
  };
  vm.runInNewContext(source, scope, { filename: 'native-worker.js' });
  await scope.self.onmessage({ data: { id: 7, type: 'init', backend: 'd3d12', model: 'fsr3-native',
    connection: { url: 'http://127.0.0.1:8767', token }, canvas: {} } });

  assert.equal(messages.length, 1);
  assert.equal(messages[0].errorCode, 'BACKEND_UNAVAILABLE');
  assert.match(messages[0].error, /^\[BACKEND_UNAVAILABLE\] Falha no serviço FSR 3\.$/);
  assert.doesNotMatch(messages[0].error, /Users|secret|token=/);
});

test('native worker preserves allowlisted server error codes and hides error text',async()=>{
  const source=(await readFile(new URL('../src/engine/native-worker.js',import.meta.url),'utf8')).replace(/^import .*;\r?\n/gm,'');
  const messages=[];const token='b'.repeat(64);
  class FakeWebSocket {
    constructor(){queueMicrotask(()=>this.onopen?.());}
    send(value){if(value===token)queueMicrotask(()=>this.onmessage?.({data:JSON.stringify({errorCode:'BACKEND_TIMEOUT',error:'token=private'})}));}
    close(){}
  }
  const scope={self:{postMessage:message=>messages.push(message)},WebSocket:FakeWebSocket,setTimeout,clearTimeout,
    createNativeGpuRenderer:async()=>null,validateNativeService,nativeServiceFailure,classifyNativeServiceError,classifyNativeServiceClose};
  vm.runInNewContext(source,scope,{filename:'native-worker.js'});
  await scope.self.onmessage({data:{id:8,type:'init',backend:'opencl',model:'anime4kcpp-acnet-f8b4',
    connection:{url:'http://127.0.0.1:8766',token},canvas:{}}});
  assert.equal(messages[0].errorCode,'BACKEND_TIMEOUT');
  assert.match(messages[0].error,/OpenCL excedeu/);
  assert.doesNotMatch(messages[0].error,/private|token=/);
});

test('native worker checks TensorRT profile over the authenticated service websocket',async()=>{
  const source=(await readFile(new URL('../src/engine/native-worker.js',import.meta.url),'utf8')).replace(/^import .*;\r?\n/gm,'');
  const messages=[],sent=[],token='c'.repeat(64);
  let returnBusy=true;
  const info={protocol:1,model:'adore',backend:'tensorrt',precision:'fp16',gpu:'test',architecture:'test',
    models:['adore'],revisions:{adore:'d'.repeat(64)},metadata:{adore:{id:'adore',status:'external-baseline'}}};
  class FakeWebSocket {
    constructor(){queueMicrotask(()=>this.onopen?.());}
    send(value){
      sent.push(value);
      if(value===token)queueMicrotask(()=>this.onmessage?.({data:JSON.stringify(info)}));
      else {
        const control=JSON.parse(value);
        if(returnBusy) {
          returnBusy=false;
          queueMicrotask(()=>this.onmessage?.({data:JSON.stringify({errorCode:'NATIVE_SESSION_BUSY'})}));
        } else queueMicrotask(()=>this.onmessage?.({data:JSON.stringify({profileCheck:true,supported:true,available:false,
          model:control.model,shapes:control.shapes,missingShapes:[[640,1136]]})}));
      }
    }
    close(){}
  }
  class FakeCanvas {getContext(){return {};}}
  class FakeOffscreenCanvas {getContext(){return {};}}
  const scope={self:{postMessage:message=>messages.push(message)},WebSocket:FakeWebSocket,OffscreenCanvas:FakeOffscreenCanvas,
    setTimeout,clearTimeout,createNativeGpuRenderer:async()=>null,validateNativeService,nativeServiceFailure,
    classifyNativeServiceError,classifyNativeServiceClose};
  vm.runInNewContext(source,scope,{filename:'native-worker.js'});
  await scope.self.onmessage({data:{id:1,type:'init',backend:'tensorrt',model:'adore',connection:{url:'http://127.0.0.1:8766',token},canvas:new FakeCanvas()}});
  await scope.self.onmessage({data:{id:2,type:'profile-check',model:'adore',shapes:[[640,1136]]}});
  await scope.self.onmessage({data:{id:3,type:'profile-check',model:'adore',shapes:[[640,1136]]}});
  assert.deepEqual(messages.map(message=>message.id),[1,2,3]);
  assert.equal(messages[1].errorCode,'NATIVE_SESSION_BUSY');
  assert.equal(messages[2].result.available,false);
  assert.equal(sent[0],token);
  assert.deepEqual(JSON.parse(sent[1]),{type:'profile-check',model:'adore',shapes:[[640,1136]]});
});
