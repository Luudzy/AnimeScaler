import { HEADER_BYTES, writeHeader, readResponse, jpegResponse, rgbResponse, repeatResponse } from './native-protocol.js';
import { TemporalPresenter } from './temporal-presenter.js';
import { createNativeGpuRenderer } from './native-gpu-renderer.js';
import { validateNativeService, nativeServiceFailure, classifyNativeServiceError, classifyNativeServiceClose } from './native-service-contract.js';

let output, context, connection, input, inputContext, body, frameId = 0, socket, pending, processingMs = 0, warmup = false;
let directCopySupported = true;
let temporal = null;
let gpu = null;
let cachedImage=null,lastPresentedImage=null,reused=false;
let fsr3=false,nativeBackend='rocm',nativeModel='adore',lastFsrCrop=null,lastFsrTime=null;
const serviceError = failure => Object.assign(new Error(`[${failure.code}] ${failure.message}`), { code: failure.code });
const SERVER_FAILURE_KINDS = Object.freeze({NATIVE_SESSION_BUSY:'busy',BACKEND_TIMEOUT:'timeout',
  BACKEND_UNAVAILABLE:'unavailable',BACKEND_INVALID_RESPONSE:'invalid-response'});
function receive() {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending = null; socket.close(); reject(serviceError(nativeServiceFailure(nativeBackend, 'timeout', nativeModel))); }, 30000);
    pending = { resolve(value) { clearTimeout(timer); pending = null; resolve(value); },
      reject(error) { clearTimeout(timer); pending = null; reject(error); } };
  });
}
async function connect(model,transport='rgba') {
  socket = new WebSocket(connection.url.replace('http:', 'ws:') + '/stream?model=' + encodeURIComponent(model)+'&transport='+transport+'&repeat=1');
  socket.binaryType = 'arraybuffer';
  const ready = receive();
  socket.onopen = () => socket.send(connection.token);
  socket.onerror = () => pending?.reject(serviceError(nativeServiceFailure(nativeBackend, 'disconnected', nativeModel)));
  socket.onclose = event => pending?.reject(serviceError(classifyNativeServiceClose(nativeBackend, event, nativeModel)));
  socket.onmessage = ({data}) => {
    if (typeof data === 'string') {
      try {
        const value = JSON.parse(data);
        if (typeof value.errorCode === 'string') {
          const kind=SERVER_FAILURE_KINDS[value.errorCode]??'invalid-response';
          pending?.reject(serviceError(nativeServiceFailure(nativeBackend,kind,nativeModel)));
        } else if (value.error) pending?.reject(serviceError(classifyNativeServiceError(nativeBackend, value.error, nativeModel)));
        else if (value.processingMs !== undefined) { processingMs = value.processingMs; warmup = value.warmup === true; reused=value.reused===true; }
        else pending?.resolve(value);
      } catch { pending?.reject(serviceError(nativeServiceFailure(nativeBackend, 'invalid-response', nativeModel))); }
    } else pending?.resolve(data);
  };
  return ready;
}
async function checkTensorRTProfile({model,shapes}) {
  if(nativeBackend!=='tensorrt'||model!==nativeModel||!Array.isArray(shapes)||shapes.length<1||shapes.length>8)
    throw new Error('TensorRT profile check unavailable.');
  const reply=receive();
  socket.send(JSON.stringify({type:'profile-check',model,shapes}));
  return reply;
}
self.onmessage = async ({ data }) => {
  const { id, type, bitmap } = data;
  let bitmapClosed = false;
  try {
    let result;
    if (type === 'init') {
      connection = data.connection;
      nativeBackend=data.backend;
      nativeModel=data.model;
      fsr3=nativeBackend==='d3d12';
      if (connection.url !== (fsr3 ? 'http://127.0.0.1:8767' : 'http://127.0.0.1:8766') || !/^[a-f0-9]{64}$/.test(connection.token)) throw new Error('Configuração do serviço local inválida.');
      output = data.canvas;
      gpu = await createNativeGpuRenderer(output);
      result = await connect(data.model,fsr3 ? 'rgba' : (data.transport??(gpu?'rgb':'rgba')));
      validateNativeService(result,data);
      context = gpu ? {putImageData:image=>gpu.presentAI(image)} : output.getContext('2d', { alpha: false, desynchronized: true });
      input = new OffscreenCanvas(1, 1);
      inputContext = input.getContext('2d', { alpha: false, willReadFrequently: true });
    } else if (type === 'profile-check') {
      result=await checkTensorRTProfile(data);
    } else if (type === 'present') {
      lastPresentedImage=null;
      temporal ??= new TemporalPresenter(output,context,gpu);
      const presented=temporal.present(bitmap,data.crop,data.size,data.sourceSequence);
      bitmap.close(); bitmapClosed = true;
      result={presented,sourceMethod:gpu?.sourceMethod,temporal:{...temporal.metrics}};
    } else if (type === 'frame') {
      const started = performance.now();
      const { size, crop } = data;
      let resetFsr3,fsrDeltaMs;
      if (fsr3) {
        const cropKey=JSON.stringify(crop);
        const timestamp=Number.isFinite(data.mediaTime) ? Math.round(data.mediaTime*1e6) : Number.isFinite(bitmap.timestamp) ? bitmap.timestamp : null;
        resetFsr3=fsr3 && (cropKey!==lastFsrCrop || (timestamp!==null && lastFsrTime!==null && (timestamp<=lastFsrTime || timestamp-lastFsrTime>250000)));
        fsrDeltaMs=timestamp!==null && lastFsrTime!==null && timestamp>lastFsrTime ? Math.min(250,(timestamp-lastFsrTime)/1000) : 1000/24;
        lastFsrCrop=cropKey;lastFsrTime=timestamp;
      }
      const { inputWidth: w, inputHeight: h, outputWidth: ow, outputHeight: oh } = size;
      if (input.width !== w || input.height !== h) {
        input.width = w; input.height = h;
        body = new ArrayBuffer(HEADER_BYTES + w * h * 4);
      }
      if (output.width !== ow || output.height !== oh) { output.width = ow; output.height = oh; }
      let reference=null;
      if (data.fluidVideo) {
        temporal ??= new TemporalPresenter(output,context,gpu);
        // Establish a displayed source before inference, including the first
        // frame after comparison. A stopped capture may never send 'present'.
        temporal.resize(ow,oh);
        if (data.pngOutput || !temporal.current) temporal.present(bitmap,crop,size,data.sourceSequence);
        reference=temporal.sample(bitmap,crop,data.sourceSequence);
      }
      let copied = false;
      if (directCopySupported && typeof bitmap.copyTo === 'function' && crop.width === w && crop.height === h) {
        try {
          await bitmap.copyTo(new Uint8Array(body, HEADER_BYTES), { rect: crop, format: 'RGBA', layout: [{offset: 0, stride: w * 4}] });
          copied = true;
        } catch { directCopySupported = false; }
      }
      if (!copied) {
        inputContext.drawImage(bitmap, crop.x, crop.y, crop.width, crop.height, 0, 0, w, h);
        new Uint8Array(body, HEADER_BYTES).set(inputContext.getImageData(0, 0, w, h).data);
      }
      bitmap.close(); bitmapClosed = true;
      const captureMs = performance.now() - started;
      const currentId = ++frameId >>> 0;
      writeHeader(body, currentId, w, h, ow, oh);
      const sent = performance.now();
      if (socket.readyState !== WebSocket.OPEN) throw serviceError(nativeServiceFailure(nativeBackend,'disconnected',nativeModel));
      const reply = receive();
      if (fsr3) socket.send(JSON.stringify({reset:resetFsr3,frameDeltaMs:fsrDeltaMs}));
      socket.send(body);
      const buffer = await reply;
      const transportMs = performance.now() - sent;
      const renderStart = performance.now();
      const repeated=repeatResponse(buffer,currentId,ow,oh);
      const jpeg=jpegResponse(buffer,currentId,ow,oh);
      const rgb=rgbResponse(buffer,currentId,ow,oh);
      let image;
      if(repeated) {
        if(!cachedImage || cachedImage.width!==ow || cachedImage.height!==oh)throw new Error('Quadro repetido sem imagem anterior válida.');
        image=cachedImage;
      } else if(rgb && gpu) image={width:ow,height:oh,rgb};
      else if(rgb) {
        const rgba=new Uint8ClampedArray(ow*oh*4);
        for(let i=0,j=0;i<rgb.length;i+=3,j+=4){rgba[j]=rgb[i];rgba[j+1]=rgb[i+1];rgba[j+2]=rgb[i+2];rgba[j+3]=255;}
        image=new ImageData(rgba,ow,oh);
      } else image=jpeg?await createImageBitmap(jpeg):new ImageData(readResponse(buffer,currentId,ow,oh),ow,oh);
      if(!jpeg)cachedImage=image;
      try {
        if(image.width!==ow||image.height!==oh)throw new Error('Dimensões JPEG inválidas.');
        if (data.fluidVideo) {
          lastPresentedImage=null;temporal.accept(image,reference,repeated);
          // Inline output has no separate page presentation clock.
          if (data.pngOutput) temporal.draw();
        }
        else {
          temporal=null;
          if(repeated && lastPresentedImage===image) {
            if(gpu)gpu.draw(null,true);
          } else if(gpu)gpu.presentAI(image);else if(jpeg)context.drawImage(image,0,0);else context.putImageData(image,0,0);
          lastPresentedImage=jpeg?null:image;
        }
      } finally {image.close?.();}
      result = { inferenceMs: processingMs, warmup, compositor:gpu?'webgpu':'canvas2d', fluidVideo:!!data.fluidVideo,
        sourceMethod:gpu?.sourceMethod,
        temporal:temporal?{...temporal.metrics}:null,
        captureMethod: copied ? 'VideoFrame.copyTo' : 'canvas',
        transportMs, reused, repeated, transport:repeated?'repeat':rgb?'rgb':jpeg?'jpeg98':'rgba', responseBytes:buffer.byteLength,
        captureMs, renderMs: performance.now() - renderStart, workerMs:performance.now()-started };
      if (data.pngOutput) {
        const blob = await output.convertToBlob({ type: 'image/png' });
        if (blob.size > 16 * 1024 * 1024) throw new Error('FRAME_TOO_LARGE');
        if (blob.type !== 'image/png' || !blob.size) throw new Error('INVALID_PNG_OUTPUT');
        const pngBytes = await blob.arrayBuffer();
        Object.assign(result, { sequence: data.sequence, mediaTime: data.mediaTime, width: ow, height: oh, mime: 'image/png', pngBytes });
      }
    } else throw new Error('Operação ROCm desconhecida.');
    self.postMessage({ id, result }, result.pngBytes ? [result.pngBytes] : []);
  } catch (error) {
    if (!bitmapClosed) bitmap?.close();
    self.postMessage({ id, error: error.message, errorCode: error.code ?? 'BACKEND_UNAVAILABLE' });
  }
};
