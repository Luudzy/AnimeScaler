import { sharedPattern } from './shared-pattern.js';
let listener;
const sessionId = 'diagnostic-session';
const sessionKey = `animeScalerViewer:${sessionId}`;
document.querySelector('#start').onclick = async () => {
  document.querySelector('#start').disabled = true;
  const source = document.createElement('canvas'); source.width = 640; source.height = 360;
  const context = source.getContext('2d');
  const pattern = new ImageData(sharedPattern(640,360).rgba,640,360);
  let frame = 0;
  const timer = setInterval(() => { context.putImageData(pattern,0,0); context.fillStyle='white'; context.fillRect(frame++ % 600,100,40,40); }, 1000/60);
  const stream = source.captureStream(60);
  window.addEventListener('pagehide', () => { clearInterval(timer); stream.getTracks().forEach(t=>t.stop()); });
  const settings = {backend:'rocm',model:document.querySelector('#model').value,fluidVideo:document.querySelector('#fluid').checked,preset:'720p',maxFps:30,adaptive:false,enabled:true};
  globalThis.chrome = {
    runtime: {getURL: path => path === 'native/connection.json' ? '/diagnostics/native-config' : '/' + path,
      sendMessage: async message => { if (message.event?.type === 'performance') document.querySelector('#report').textContent = JSON.stringify(message.event.data,null,2); }},
    storage: {session: {get:async key=>({[key]:{settings,sourceTabId:1,viewerTabId:2,streamId:'synthetic',captureFrame:{frameId:0}}}),remove:async()=>{}},
      onChanged:{addListener: callback=>{listener=callback;}}},
    windows: {get:async id=>({id,state:'normal',left:0,top:0,width:1280,height:800}),onBoundsChanged:{addListener(){},removeListener(){}}},
    tabs:{sendMessage:async()=>({rect:{left:0,top:0,width:640,height:360},viewport:{width:640,height:360}})},
  };
  navigator.mediaDevices.getUserMedia = async () => stream;
  history.replaceState(null, '', `?session=${sessionId}`);
  await import('../viewer/viewer.js');
};
document.querySelector('#switch').onclick = () => listener?.({animeScalerSettings:{newValue:{backend:'webgpu',enabled:true}}},'local');
