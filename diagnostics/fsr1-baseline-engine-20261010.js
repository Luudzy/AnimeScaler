import { resolveInputSize, resolveOutputSize, resolveFsrInputSize } from '../src/core/settings.js';
import { DiagnosticLog } from '../src/core/diagnostic-log.js';
import { AsrTemporalFilter } from '../src/engine/asr-temporal.js';
import { ASR_ANIME_WGSL, FSR_EASU_WGSL, FSR_RCAS_WGSL, VIDEO_DENOISE_WGSL, FSR_INPUT_RESIZE_WGSL } from '../src/engine/fsr-shaders.js';

const textureUsage = globalThis.GPUTextureUsage ?? { COPY_DST: 1, TEXTURE_BINDING: 2, RENDER_ATTACHMENT: 4, COPY_SRC:8 };
const bufferUsage = globalThis.GPUBufferUsage ?? { UNIFORM: 1, COPY_DST: 2 };

export class FsrEngineError extends Error {
  constructor(code, message, cause) { super(message, { cause }); this.code = code; this.name = 'FsrEngineError'; }
}

class FsrRenderer {
  constructor(device, canvas, format, refinementShader = FSR_RCAS_WGSL, temporalEnabled = false) {
    this.device = device; this.canvas = canvas; this.format = format;
    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new FsrEngineError('WEBGPU_CANVAS_UNAVAILABLE', 'Canvas WebGPU indisponível.');
    this.refinementShader = refinementShader;
    this.linear = device.createSampler({minFilter:'linear',magFilter:'linear'});
    if(temporalEnabled) this.temporal=new AsrTemporalFilter(device);
  }
  async configure(sourceWidth, sourceHeight, outputWidth, outputHeight, inputWidth = sourceWidth, inputHeight = sourceHeight) {
    this.device.pushErrorScope?.('validation');
    let failure;
    try { await this.configureResources(sourceWidth, sourceHeight, outputWidth, outputHeight, inputWidth, inputHeight); }
    catch (error) { failure = error; }
    const validation = await this.device.popErrorScope?.();
    if (failure) throw failure;
    if (validation) throw new FsrEngineError('GPU_VALIDATION_ERROR', `FSR/ASR WebGPU: ${validation.message}`, validation);
  }
  async configureResources(sourceWidth, sourceHeight, outputWidth, outputHeight, inputWidth, inputHeight) {
    if (this.sourceWidth === sourceWidth && this.sourceHeight === sourceHeight && this.outputWidth === outputWidth && this.outputHeight === outputHeight && this.inputWidth === inputWidth && this.inputHeight === inputHeight) return;
    this.source?.destroy(); this.easu?.destroy(); this.dimensions?.destroy(); this.filtered?.destroy(); this.denoiseParameters?.destroy();
    this.denoiseMode = null;
    this.resized?.destroy(); this.resized = null;
    this.inputWidth = inputWidth; this.inputHeight = inputHeight;
    this.needsResize = sourceWidth !== inputWidth || sourceHeight !== inputHeight;
    this.sourceWidth = sourceWidth; this.sourceHeight = sourceHeight; this.outputWidth = outputWidth; this.outputHeight = outputHeight;
    this.canvas.width = outputWidth; this.canvas.height = outputHeight;
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.source = this.device.createTexture({ size: [sourceWidth, sourceHeight], format: 'rgba8unorm', usage: textureUsage.COPY_DST | textureUsage.TEXTURE_BINDING | textureUsage.RENDER_ATTACHMENT | (this.temporal ? textureUsage.COPY_SRC : 0) });
    this.filtered = this.device.createTexture({ size: [sourceWidth, sourceHeight], format: 'rgba8unorm', usage: textureUsage.TEXTURE_BINDING | textureUsage.RENDER_ATTACHMENT });
    if (this.needsResize) this.resized = this.device.createTexture({size:[inputWidth,inputHeight],format:'rgba8unorm',usage:textureUsage.TEXTURE_BINDING | textureUsage.RENDER_ATTACHMENT});
    this.denoiseParameters = this.device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    this.easu = this.device.createTexture({ size: [outputWidth, outputHeight], format: 'rgba8unorm', usage: textureUsage.RENDER_ATTACHMENT | textureUsage.TEXTURE_BINDING });
    this.dimensions = this.device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.dimensions, 0, new Float32Array([inputWidth, inputHeight, outputWidth, outputHeight]));
    if (!this.easuPipeline) {
      const easu = this.device.createShaderModule({ code: FSR_EASU_WGSL });
      const rcas = this.device.createShaderModule({ code: this.refinementShader });
      const denoise = this.device.createShaderModule({ code: VIDEO_DENOISE_WGSL });
      const resize = this.device.createShaderModule({code:FSR_INPUT_RESIZE_WGSL});
      this.resizePipeline = await this.device.createRenderPipelineAsync({layout:'auto',vertex:{module:resize,entryPoint:'vertexMain'},fragment:{module:resize,entryPoint:'resizeMain',targets:[{format:'rgba8unorm'}]},primitive:{topology:'triangle-list'}});
      this.denoisePipeline = await this.device.createRenderPipelineAsync({ layout: 'auto', vertex: { module: denoise, entryPoint: 'vertexMain' }, fragment: { module: denoise, entryPoint: 'denoiseMain', targets: [{ format: 'rgba8unorm' }] }, primitive: { topology: 'triangle-list' } });
      this.easuPipeline = await this.device.createRenderPipelineAsync({ layout: 'auto', vertex: { module: easu, entryPoint: 'vertexMain' }, fragment: { module: easu, entryPoint: 'easuMain', targets: [{ format: 'rgba8unorm' }] }, primitive: { topology: 'triangle-list' } });
      this.rcasPipeline = await this.device.createRenderPipelineAsync({ layout: 'auto', vertex: { module: rcas, entryPoint: 'vertexMain' }, fragment: { module: rcas, entryPoint: 'rcasMain', targets: [{ format: this.format }] }, primitive: { topology: 'triangle-list' } });
    }
    // EASU uses textureLoad: auto layout contains only texture and dimensions.
    this.easuBind = this.device.createBindGroup({ layout: this.easuPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.source.createView() }, { binding: 2, resource: { buffer: this.dimensions } }] });
    this.filteredBind = this.device.createBindGroup({ layout: this.easuPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.filtered.createView() }, { binding: 2, resource: { buffer: this.dimensions } }] });
    this.denoiseBind = this.device.createBindGroup({ layout: this.denoisePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.source.createView() }, { binding: 2, resource: { buffer: this.denoiseParameters } }] });
    if(this.temporal) {
      await this.temporal.configure(this.source,this.filtered,sourceWidth,sourceHeight);
      this.temporalEasuBinds=this.temporal.history.map(texture=>this.device.createBindGroup({layout:this.easuPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()},{binding:2,resource:{buffer:this.dimensions}}]}));
    }
    if (this.needsResize) {
      const bind = texture => this.device.createBindGroup({layout:this.resizePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()},{binding:1,resource:this.linear},{binding:2,resource:{buffer:this.dimensions}}]});
      this.resizeRawBind = bind(this.source); this.resizeFilteredBind = bind(this.filtered);
      if(this.temporal) this.resizeTemporalBinds=this.temporal.history.map(bind);
      this.resizedEasuBind = this.device.createBindGroup({layout:this.easuPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.resized.createView()},{binding:2,resource:{buffer:this.dimensions}}]});
    }
    this.rcasBind = this.device.createBindGroup({ layout: this.rcasPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.easu.createView() }] });
  }
  render(video, sourceRect = null, denoiseMode = 'balanced') {
    const cropKey=JSON.stringify(sourceRect);
    if(this.lastCropKey!==cropKey || this.lastDenoiseMode!==denoiseMode) this.temporal?.reset();
    this.lastCropKey=cropKey; this.lastDenoiseMode=denoiseMode;
    const denoise = denoiseMode !== 'off';
    if (denoise && this.denoiseMode !== denoiseMode) {
      this.device.queue.writeBuffer(this.denoiseParameters, 0, new Float32Array(denoiseMode === 'strong' ? [.10, .95, 0, 0] : [.055, .75, 0, 0]));
      this.denoiseMode = denoiseMode;
    }
    const source = { source: video };
    if (sourceRect) source.origin = { x: sourceRect.x, y: sourceRect.y };
    this.device.queue.copyExternalImageToTexture(source, { texture: this.source, colorSpace: 'srgb' }, [this.sourceWidth, this.sourceHeight]);
    const encoder = this.device.createCommandEncoder({ label: 'AnimeScaler FSR 1 encoder' });
    if (denoise) {
      const prefilter = encoder.beginRenderPass({ colorAttachments: [{ view: this.filtered.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      prefilter.setPipeline(this.denoisePipeline); prefilter.setBindGroup(0, this.denoiseBind); prefilter.draw(3); prefilter.end();
    }
    const temporal=this.temporal?.render(encoder,denoise);
    this.lastTemporalHistoryAvailable=temporal?.historyAvailable ?? false;
    if (this.needsResize) {
      const resize = encoder.beginRenderPass({colorAttachments:[{view:this.resized.createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});
      resize.setPipeline(this.resizePipeline); resize.setBindGroup(0,temporal ? this.resizeTemporalBinds[temporal.index] : (denoise ? this.resizeFilteredBind : this.resizeRawBind)); resize.draw(3); resize.end();
    }
    const easu = encoder.beginRenderPass({ colorAttachments: [{ view: this.easu.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    easu.setPipeline(this.easuPipeline); easu.setBindGroup(0, this.needsResize ? this.resizedEasuBind : (temporal ? this.temporalEasuBinds[temporal.index] : (denoise ? this.filteredBind : this.easuBind))); easu.draw(3); easu.end();
    const rcas = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    rcas.setPipeline(this.rcasPipeline); rcas.setBindGroup(0, this.rcasBind); rcas.draw(3); rcas.end();
    this.device.queue.submit([encoder.finish()]);
  }
  destroy() { this.temporal?.destroy(); this.source?.destroy(); this.easu?.destroy(); this.dimensions?.destroy(); this.filtered?.destroy(); this.denoiseParameters?.destroy(); this.resized?.destroy(); this.context.unconfigure?.(); }
}

export class FsrEngine {
  constructor({ requestAdapter } = {}) {
    this.requestAdapter = requestAdapter ?? globalThis.navigator?.gpu?.requestAdapter?.bind(globalThis.navigator.gpu);
    this.diagnostics = new DiagnosticLog(); this.pending = Promise.resolve();
  }
  exclusive(operation) { const result = this.pending.then(operation); this.pending = result.catch(() => {}); return result; }
  sourceDimensions(source) {
    const width = source?.videoWidth ?? source?.displayWidth ?? source?.width;
    const height = source?.videoHeight ?? source?.displayHeight ?? source?.height;
    return { width, height };
  }
  async initialize({ canvas, video, settings, adaptiveLevel = 0, sourceRect = null }) {
    if (!this.requestAdapter) throw new FsrEngineError('WEBGPU_UNAVAILABLE', 'WebGPU não está disponível neste navegador.');
    const source = this.sourceDimensions(video);
    if (!source.width || !source.height) throw new FsrEngineError('VIDEO_NOT_READY', 'O vídeo ainda não possui dimensões decodificadas.');
    this.settings = settings; this.diagnostics.event('initializing', settings);
    const adapter = await this.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new FsrEngineError('WEBGPU_ADAPTER_MISSING', 'Não foi possível obter uma GPU WebGPU.');
    this.device = await adapter.requestDevice();
    this.validationError = null;
    this.device.addEventListener?.('uncapturederror', event => {
      this.validationError = new FsrEngineError('GPU_VALIDATION_ERROR', `FSR/ASR WebGPU: ${event.error.message}`, event.error);
      this.diagnostics.event('gpu-error', { message: event.error.message });
      event.preventDefault();
    });
    const isAsr = settings.model === 'asr-anime-spatial' || settings.model === 'asr-anime-temporal';
    this.renderer = new FsrRenderer(this.device, canvas, globalThis.navigator?.gpu?.getPreferredCanvasFormat?.() ?? 'bgra8unorm', isAsr ? ASR_ANIME_WGSL : FSR_RCAS_WGSL, settings.model === 'asr-anime-temporal');
    try { await this.reconfigure(video, adaptiveLevel, sourceRect); }
    catch (error) { this.renderer.destroy(); this.device.destroy(); throw error; }
    this.device.lost?.then(info => { this.deviceLostInfo = info; }).catch(() => {});
  }
  targetSize(sourceWidth, sourceHeight, preset, adaptiveLevel) {
    const input = resolveInputSize(sourceWidth, sourceHeight, preset, adaptiveLevel);
    return resolveOutputSize(sourceWidth, sourceHeight, preset, input.width, input.height);
  }
  reconfigure(video, adaptiveLevel = 0, sourceRect = this.sourceRect) { return this.exclusive(() => this.reconfigureNow(video, adaptiveLevel, sourceRect)); }
  async reconfigureNow(video, adaptiveLevel = 0, sourceRect = this.sourceRect) {
    const dimensions = this.sourceDimensions(video);
    const sourceWidth = sourceRect?.width ?? dimensions.width; const sourceHeight = sourceRect?.height ?? dimensions.height;
    const output = this.targetSize(sourceWidth, sourceHeight, this.settings?.preset ?? 'balanced', adaptiveLevel);
    const input = resolveFsrInputSize(sourceWidth,sourceHeight,output.width,output.height,this.settings?.fsrQuality);
    await this.renderer.configure(sourceWidth, sourceHeight, output.width, output.height, input.width, input.height);
    this.sourceRect = sourceRect ? { ...sourceRect } : null;
    this.currentSize = { sourceWidth, sourceHeight, inputWidth: input.width, inputHeight: input.height, fsrQuality:this.settings?.fsrQuality ?? 'native-aa', modelOutputWidth: output.width, modelOutputHeight: output.height, outputWidth: output.width, outputHeight: output.height, adaptiveLevel };
    return this.currentSize;
  }
  needsReconfigure(video, adaptiveLevel = 0, sourceRect = this.sourceRect) {
    const dimensions = this.sourceDimensions(video);
    const width = sourceRect?.width ?? dimensions.width; const height = sourceRect?.height ?? dimensions.height;
    return !this.currentSize || this.currentSize.sourceWidth !== width || this.currentSize.sourceHeight !== height || this.currentSize.adaptiveLevel !== adaptiveLevel;
  }
  updateSettings(settings, video, adaptiveLevel = 0, sourceRect = this.sourceRect) { return this.exclusive(async () => { this.settings = settings; await this.reconfigureNow(video, adaptiveLevel, sourceRect); }); }
  resetHistory() { return this.exclusive(async () => { this.renderer?.temporal?.reset(); this.lastMediaTime=undefined; this.lastFrameClock=undefined; }); }
  processFrame(video, sourceRect = this.sourceRect) { return this.exclusive(async () => {
    if (this.validationError) throw this.validationError;
    if (this.deviceLostInfo) throw new FsrEngineError('GPU_DEVICE_LOST', `Dispositivo WebGPU perdido: ${this.deviceLostInfo.message || this.deviceLostInfo.reason}`);
    const started = performance.now(); this.sourceRect = sourceRect ? { ...sourceRect } : null;
    const mediaTime=Number.isFinite(video.currentTime) ? video.currentTime : (Number.isFinite(video.timestamp) ? video.timestamp/1e6 : null);
    if((mediaTime!==null && this.lastMediaTime!==undefined && (mediaTime<=this.lastMediaTime || mediaTime-this.lastMediaTime>.25)) || (this.lastFrameClock!==undefined && started-this.lastFrameClock>500)) this.renderer.temporal?.reset();
    this.lastMediaTime=mediaTime===null ? undefined : mediaTime; this.lastFrameClock=started;
    try { this.renderer.render(video, sourceRect, this.settings?.spatialDenoise ?? 'balanced'); }
    catch (error) { if (error?.name === 'SecurityError') throw new FsrEngineError('CROSS_ORIGIN_VIDEO', 'O navegador bloqueou o acesso a este frame por CORS/origem cruzada.', error); throw error; }
    const modelLabel = this.settings?.model === 'asr-anime-temporal' ? 'ASR Anime Temporal (experimental)' : (this.settings?.model === 'asr-anime-spatial' ? 'ASR Anime Spatial' : 'AMD FSR 1 Spatial');
    const stats = { totalMs: performance.now() - started, inferenceMs: 0, backend: 'webgpu', modelPrecision: 'native', layout: 'RGBA', modelLabel, temporalHistoryAvailable:this.renderer.lastTemporalHistoryAvailable, spatialDenoise: this.settings?.spatialDenoise ?? 'balanced', ...this.currentSize };
    this.diagnostics.frame(stats); return stats;
  }); }
  destroy() { return this.exclusive(async () => { this.renderer?.destroy(); this.renderer = null; this.device?.destroy(); this.device = null; this.currentSize = null; this.diagnostics.flush(); }); }
}
