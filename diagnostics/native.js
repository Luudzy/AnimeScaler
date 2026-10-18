import { NativeEngine } from '../src/engine/native-engine.js';
import { normalizeSettings } from '../src/core/settings.js';
import { sharedPattern } from './shared-pattern.js';

const run = document.querySelector('#run'), result = document.querySelector('#result');
run.onclick = async () => {
  run.disabled = true;
  let engine;
  try {
    const preset = document.querySelector('#preset').value;
    const model = document.querySelector('#model').value;
    const width = preset === '1080p' ? 1280 : 640, height = preset === '1080p' ? 720 : 360;
    const source = document.createElement('canvas'); source.width = width; source.height = height;
    const pattern = sharedPattern(width, height);
    source.getContext('2d').putImageData(new ImageData(pattern.rgba, width, height), 0, 0);
    const old = document.querySelector('#output'), canvas = old.cloneNode(false); old.replaceWith(canvas);
    const connectionResponse = await fetch('/diagnostics/native-config');
    const connection = await connectionResponse.json();
    engine = new NativeEngine();
    const crop = {x:0,y:0,width,height};
    await engine.initialize({ canvas, video: source, sourceRect: crop, connection,
      settings: {...normalizeSettings({model,backend:'rocm',preset}),nativeTransport:document.querySelector('#rgb').checked?'rgb':document.querySelector('#jpeg').checked?'jpeg':'rgba'} });
    const samples = [], service = [], transport = [];let compositor;
    for (let index = 0; index < 65; index++) {
      result.textContent = `Quadro ${index+1}/65`;
      const stats = await engine.processFrame(source, crop);
      compositor=stats.compositor;
      if (index >= 5) { samples.push(stats.totalMs); service.push(stats.inferenceMs); transport.push(stats.transportMs); }
    }
    const median = a => [...a].sort((a,b)=>a-b)[Math.floor(a.length/2)];
    result.textContent = JSON.stringify({ model, transport:document.querySelector('#rgb').checked?'rgb':document.querySelector('#jpeg').checked?'jpeg98':'rgba', input:[width,height], output:[engine.currentSize.outputWidth,engine.currentSize.outputHeight],
      frames:samples.length, compositor, medianMs:median(samples),p95Ms:[...samples].sort((a,b)=>a-b)[Math.ceil(samples.length*.95)-1],
      processingMedianMs:median(service),transportMedianMs:median(transport),
      throughputFps:1000/(samples.reduce((a,b)=>a+b,0)/samples.length),
      scope:'Canvas sintético → ImageBitmap → worker → WebSocket → ROCm → canvas; sem decodificação/captura de aba' },null,2);
  } catch (error) { result.textContent = error.stack || error.message; }
  finally { await engine?.destroy(); run.disabled = false; }
};

document.querySelector('#compare').onclick=async()=>{
  const button=document.querySelector('#compare');button.disabled=true;
  try {
    const results=[];
    document.querySelector('#jpeg').checked=false;
    for(const enabled of [false,true]) {
      document.querySelector('#rgb').checked=enabled;
      await run.onclick();
      results.push(JSON.parse(result.textContent));
    }
    document.querySelector('#comparison').textContent=JSON.stringify(results,null,2);
  } catch(error) { document.querySelector('#comparison').textContent=error.message; }
  finally {button.disabled=false;}
};

