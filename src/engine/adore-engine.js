import { resolveInputSize, resolveOutputSize } from '../core/settings.js';
import { GpuPrePost } from './gpu-prepost.js';
import { getModelCandidates, supportsFloat32GpuIo } from './model-config.js';
import { loadOrt } from './ort-loader.js';
import { DiagnosticLog } from '../core/diagnostic-log.js';

export class AnimeScalerError extends Error {
  constructor(code, message, cause) {
    super(message, { cause });
    this.name = 'AnimeScalerError';
    this.code = code;
  }
}

export class AdoreEngine {
  constructor() {
    this.diagnostics = new DiagnosticLog();
    this.ort = null;
    this.session = null;
    this.model = null;
    this.device = null;
    this.gpu = null;
    this.inputTensor = null;
    this.outputTensor = null;
    this.inputName = null;
    this.outputName = null;
    this.settings = null;
    this.currentSize = null;
    this.sourceRect = null;
    this.pending = Promise.resolve();
    this.closed = false;
    this.failure = null;
    this.failureSignal = new Promise((_, reject) => { this.rejectFailure = reject; });
    this.failureSignal.catch(() => {});
  }

  assertActive() {
    if (this.failure) throw this.failure;
    if (this.closed) throw new AnimeScalerError('ENGINE_CLOSED', 'Engine encerrada.');
  }

  fail(error) {
    if (this.closed || this.failure) return;
    this.failure = error;
    this.rejectFailure(error);
    this.diagnostics.error(error);
  }

  watchDevice() {
    const device = this.device;
    this.onGpuError = event => {
      event.preventDefault?.();
      this.fail(new AnimeScalerError('GPU_VALIDATION_ERROR', 'Erro não capturado no dispositivo WebGPU.', event.error));
    };
    device.addEventListener?.('uncapturederror', this.onGpuError);
    device.lost?.then(info => {
      if (this.closed) return;
      this.deviceLostInfo = info;
      this.fail(new AnimeScalerError('GPU_DEVICE_LOST', `Dispositivo WebGPU perdido: ${info.message || info.reason}`));
    }).catch(error => this.fail(new AnimeScalerError('GPU_DEVICE_LOST', 'Falha no dispositivo WebGPU.', error)));
  }

  exclusive(operation) {
    if (this.closing) return Promise.reject(this.failure ?? new AnimeScalerError('ENGINE_CLOSED', 'Engine encerrada.'));
    const result = this.pending.then(operation);
    this.pending = result.catch(() => {});
    return result;
  }

  updateSettings(settings, video, adaptiveLevel = 0, sourceRect = this.sourceRect) {
    return this.exclusive(async () => {
      this.assertActive();
      const modelChanged = settings.model !== this.settings?.model;
      this.settings = settings;
      this.diagnostics.event('settings', settings);
      if (modelChanged) {
        await this.session?.release();
        this.session = null;
        this.model = null;
        await this.createSession();
      }
      await this.reconfigureNow(video, adaptiveLevel, sourceRect);
    });
  }

  async initialize({ canvas, video, settings, adaptiveLevel = 0, sourceRect = null }) {
    if (!globalThis.navigator?.gpu) {
      throw new AnimeScalerError('WEBGPU_UNAVAILABLE', 'WebGPU não está disponível neste navegador.');
    }
    if (!video?.videoWidth || !video?.videoHeight) {
      throw new AnimeScalerError('VIDEO_NOT_READY', 'O vídeo ainda não possui dimensões decodificadas.');
    }

    this.settings = settings;
    this.diagnostics.event('initializing', settings);
    this.ort = await loadOrt();
    this.device = await this.ort.env.webgpu.device;
    if (!this.device) {
      throw new AnimeScalerError('WEBGPU_DEVICE_MISSING', 'ONNX Runtime não expôs um dispositivo WebGPU.');
    }

    this.watchDevice();
    try {
      await this.createSession();
      this.gpu = new GpuPrePost({ device: this.device, canvas });
      await this.reconfigure(video, adaptiveLevel, sourceRect);
      this.verifySource(video, sourceRect);
    } catch (error) { await this.destroy(); throw error; }
    const info = this.device.adapterInfo ?? {};
    this.diagnostics.event('gpu', {
      vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
      shaderF16: this.device.features?.has('shader-f16') ?? false,
      timestampQuery: this.device.features?.has('timestamp-query') ?? false,
      browser: globalThis.navigator?.userAgent,
    });
  }

  async createSession() {
    this.assertActive();
    this.device ??= await this.ort.env.webgpu.device;
    const failures = [];
    for (const candidate of getModelCandidates(undefined, this.settings?.model)) {
      if (candidate.precision === 'fp16' && !this.device?.features?.has('shader-f16')) continue;
      let session;
      try {
        let layout = 'NHWC';
        const create = preferredLayout => this.ort.InferenceSession.create(candidate.url, {
          executionProviders: [{name: 'webgpu', preferredLayout}],
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3,
        });
        try { session = await create(layout); }
        catch (error) {
          if (layout !== 'NHWC') throw error;
          this.diagnostics.error(error);
          layout = 'NCHW';
          session = await create(layout);
        }
        const inputName = session.inputNames?.[0];
        const outputName = session.outputNames?.[0];
        if (!inputName || !outputName || session.inputNames.length !== 1 || session.outputNames.length !== 1) {
          throw new Error('model has no tensor input/output');
        }
        if (!supportsFloat32GpuIo(session.inputMetadata?.[0], session.outputMetadata?.[0])) {
          const inputType = session.inputMetadata?.[0]?.type ?? 'unknown';
          const outputType = session.outputMetadata?.[0]?.type ?? 'unknown';
          throw new Error(`GPU I/O ${inputType} -> ${outputType} is not supported by the float32 shader path`);
        }
        this.assertActive();
        this.session = session;
        this.model = {...candidate, layout};
        this.inputName = inputName;
        this.outputName = outputName;
        this.diagnostics.event('model', {model: this.settings?.model, precision: candidate.precision, layout});
        return;
      } catch (error) {
        if (session) await session.release();
        this.assertActive();
        this.diagnostics.error(error);
        failures.push(`${candidate.precision}: ${error?.message ?? String(error)}`);
      }
    }

    throw new AnimeScalerError(
      'MODEL_INIT_FAILED',
      `Não foi possível iniciar o modelo selecionado no WebGPU. ${failures.join(' | ')}`,
    );
  }

  reconfigure(video, adaptiveLevel = 0, sourceRect = this.sourceRect) {
    return this.exclusive(() => this.reconfigureNow(video, adaptiveLevel, sourceRect));
  }

  async reconfigureNow(video, adaptiveLevel = 0, sourceRect = this.sourceRect) {
    this.assertActive();
    this.diagnostics.flush();
    if (!this.gpu || !this.ort) throw new AnimeScalerError('NOT_INITIALIZED', 'Engine não inicializada.');
    const sourceWidth = sourceRect?.width ?? video.videoWidth;
    const sourceHeight = sourceRect?.height ?? video.videoHeight;
    const { width, height } = resolveInputSize(
      sourceWidth,
      sourceHeight,
      this.settings?.preset ?? 'balanced',
      adaptiveLevel,
    );

    if (this.session && !supportsFloat32GpuIo(this.session.inputMetadata[0], this.session.outputMetadata[0],
      [1, 3, height, width], [1, 3, height * 2, width * 2])) {
      throw new AnimeScalerError('MODEL_SHAPE_UNSUPPORTED', 'Shape ONNX incompatível com buffers NCHW RGB 2x.');
    }
    this.sourceRect = sourceRect ? { ...sourceRect } : null;
    this.disposeTensors();
    const {width:displayWidth,height:displayHeight} = resolveOutputSize(sourceWidth,sourceHeight,this.settings?.preset,width,height);
    await this.gpu.configure({
      sourceWidth,
      sourceHeight,
      inputWidth: width,
      inputHeight: height,
      displayWidth,
      displayHeight,
    });

    this.inputTensor = this.ort.Tensor.fromGpuBuffer(this.gpu.getInputBuffer(), {
      dataType: 'float32',
      dims: this.gpu.getInputShape(),
    });
    this.outputTensor = this.ort.Tensor.fromGpuBuffer(this.gpu.getOutputBuffer(), {
      dataType: 'float32',
      dims: this.gpu.getOutputShape(),
    });
    this.currentSize = {
      sourceWidth,
      sourceHeight,
      inputWidth: width,
      inputHeight: height,
      modelOutputWidth: width * 2,
      modelOutputHeight: height * 2,
      outputWidth: displayWidth,
      outputHeight: displayHeight,
      adaptiveLevel,
    };
    this.diagnostics.event('resolution', this.currentSize);
    return this.currentSize;
  }

  needsReconfigure(video, adaptiveLevel, sourceRect = this.sourceRect) {
    const sourceWidth = sourceRect?.width ?? video.videoWidth;
    const sourceHeight = sourceRect?.height ?? video.videoHeight;
    return !this.currentSize
      || this.currentSize.sourceWidth !== sourceWidth
      || this.currentSize.sourceHeight !== sourceHeight
      || this.currentSize.adaptiveLevel !== adaptiveLevel;
  }

  verifySource(video, sourceRect = this.sourceRect) {
    if (!this.gpu) throw new AnimeScalerError('NOT_INITIALIZED', 'Pipeline WebGPU não inicializado.');
    try {
      this.gpu.uploadVideoFrame(video, sourceRect);
    } catch (error) {
      if (error?.name === 'SecurityError') {
        throw new AnimeScalerError(
          'CROSS_ORIGIN_VIDEO',
          'O navegador bloqueou o acesso a este frame por CORS/origem cruzada.',
          error,
        );
      }
      throw error;
    }
  }

  processFrame(video, sourceRect = this.sourceRect) {
    return this.exclusive(async () => {
      try {
        const stats = await this.processFrameNow(video, sourceRect);
        this.diagnostics.frame(stats);
        return stats;
      } catch (error) { this.diagnostics.error(error); throw error; }
    });
  }

  async processFrameNow(video, sourceRect = this.sourceRect) {
    this.assertActive();
    if (this.deviceLostInfo) {
      throw new AnimeScalerError('GPU_DEVICE_LOST', `Dispositivo WebGPU perdido: ${this.deviceLostInfo.message || this.deviceLostInfo.reason}`);
    }
    if (!this.session || !this.gpu || !this.inputTensor || !this.outputTensor) {
      throw new AnimeScalerError('NOT_INITIALIZED', 'Engine não inicializada.');
    }

    const started = performance.now();
    this.sourceRect = sourceRect ? { ...sourceRect } : null;
    this.verifySource(video, sourceRect);

    this.gpu.preprocess();
    const inferenceStarted = performance.now();
    const work = (async () => {
      try {
        await this.session.run(
          { [this.inputName]: this.inputTensor },
          { [this.outputName]: this.outputTensor },
        );
        const inferenceMs = performance.now() - inferenceStarted;
        this.assertActive();
        this.gpu.render();
        return inferenceMs;
      } finally {
        // Keep reconfiguration/disposal queued until the frame has finished on the GPU.
        await this.device?.queue.onSubmittedWorkDone();
      }
    })();
    // A deadline stops callers, not ORT: its borrowed buffers stay alive until work settles.
    this.activeWork = work.catch(() => {});
    let timer, inferenceMs;
    try {
      inferenceMs = await Promise.race([work, this.failureSignal, new Promise((_, reject) => {
        timer = setTimeout(() => {
          const error = new AnimeScalerError('INFERENCE_TIMEOUT', 'Inferência/GPU excedeu 30000 ms.');
          this.fail(error);
          reject(error);
        }, 30000);
      })]);
      this.assertActive();
    } catch (error) {
      this.fail(error);
      throw error;
    } finally { clearTimeout(timer); }

    return {
      totalMs: performance.now() - started,
      inferenceMs,
      modelPrecision: this.model?.precision ?? 'unknown',
      layout: this.model?.layout ?? 'unknown',
      modelLabel: this.model?.label ?? 'Modelo',
      captureMethod:this.gpu.captureMethod,
      ...this.currentSize,
    };
  }

  destroy() {
    if (this.destroying) return this.destroying;
    this.destroying = this.exclusive(async () => {
      await this.activeWork;
      this.closed = true;
      await this.destroyNow();
    });
    this.closing = true;
    return this.destroying;
  }

  disposeTensors() {
    // fromGpuBuffer has no dispose callback: GpuPrePost owns the underlying buffers.
    this.inputTensor?.dispose?.();
    this.outputTensor?.dispose?.();
    this.inputTensor = null;
    this.outputTensor = null;
  }

  async destroyNow() {
    this.diagnostics.flush();
    this.diagnostics.event('stopped');
    this.device?.removeEventListener?.('uncapturederror', this.onGpuError);
    this.disposeTensors();
    this.gpu?.destroy();
    this.gpu = null;
    if (this.session) {
      try { await this.session.release(); } catch { /* no-op */ }
    }
    this.session = null;
    this.device = null;
    this.ort = null;
    this.currentSize = null;
    this.sourceRect = null;
  }
}
