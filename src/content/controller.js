import { FramePacer } from '../core/frame-pacer.js';
import { DiagnosticLog } from '../core/diagnostic-log.js';
import { AdaptiveQualityController } from '../core/adaptive-quality.js';
import { normalizeSettings } from '../core/settings.js';
import { AnimeScalerError } from '../engine/adore-engine.js';
import { createProcessingEngine, requiresEngineRecreation } from '../engine/engine-factory.js';
import { describeCaptureTarget, findCaptureTargetElement } from './capture-target.js';
import { findBestVideo, inspectBestVideo } from './video-finder.js';
import { AnimeScalerHud } from './hud.js';
import { VideoOverlay } from './overlay.js';
import { TabCaptureSource } from './tab-capture-source.js';
import { ExclusiveSource } from './exclusive-source.js';
import { SustainedFpsGate } from '../core/sustained-fps-gate.js';

export class AnimeScalerController {
  constructor() {
    this.diagnostics = new DiagnosticLog();
    this.exclusiveSource = new ExclusiveSource();
    this.state = 'stopped';
    this.settings = normalizeSettings();
    this.captureMode = 'direct';
    this.video = null;
    this.viewerPlayer = null;
    this.viewerPlayerSession = null;
    this.displayTarget = null;
    this.tabCaptureSource = null;
    this.overlay = null;
    this.hud = null;
    this.engine = null;
    this.adaptive = new AdaptiveQualityController();
    this.busy = false;
    this.frameHandle = null;
    this.playerWatchTimer = null;
    this.waitingForPlayer = false;
    this.framePacer = new FramePacer();
    this.lastMediaTime = null;
    this.lastRenderedSourceTime = null;
    this.pausedFramePending = false;
    this.lastFrameInterval = 41.7;
    this.observedSourceFps = null;
    this.error = null;
    this.stats = null;
    this.processedFrames = 0;
    this.fpsWindowStart = performance.now();
    this.processedFps = 0;
    this.throughputGate = null;
    this.boundSync = () => this.syncGeometry();
    this.mutationObserver = new MutationObserver(() => this.ensureVideoStillExists());
  }

  getStatus() {
    return {
      state: this.state,
      captureMode: this.captureMode,
      settings: this.settings,
      error: this.error ? { code: this.error.code ?? 'UNKNOWN', message: this.error.message } : null,
      stats: this.stats,
      processedFps: this.processedFps,
      waitingForPlayer: this.waitingForPlayer,
      video: this.video ? { width: this.video.videoWidth, height: this.video.videoHeight } : null,
    };
  }

  probe() {
    const { video, score } = inspectBestVideo(document);
    return {
      ...this.getStatus(),
      hasVideo: Boolean(video),
      videoScore: score,
      candidate: video ? {
        width: video.getBoundingClientRect().width,
        height: video.getBoundingClientRect().height,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
        readyState: video.readyState,
        paused: video.paused,
      } : null,
    };
  }


  resolveViewerPlayer(sessionId = null) {
    if(this.viewerPlayerSession!==sessionId) {
      this.viewerPlayerSession=sessionId;this.viewerPlayer=null;
    }
    if(!this.viewerPlayer?.isConnected) {
      const exclusive=this.exclusiveSource.sessionId===sessionId?this.exclusiveSource.target:null;
      this.viewerPlayer=exclusive?.videoWidth>0?exclusive:
        (this.captureMode==='direct'&&this.video?.isConnected?this.video:findBestVideo(document));
    }
    return this.viewerPlayer;
  }

  getCaptureGeometry(captureFrame = {}, sessionId = null) {
    // Geometry and commands must share an element, not rescore after pause.
    const target=captureFrame.frameId===0?this.resolveViewerPlayer(sessionId):null;
    return describeCaptureTarget(document, {
      ...captureFrame,
      viewport: { width: globalThis.innerWidth ?? 0, height: globalThis.innerHeight ?? 0 },
    },target);
  }

  async playerCommand(command = {}, sessionId = null) {
    const video = this.resolveViewerPlayer(sessionId);
    if (!video) return { ok: false, code: 'NO_VIDEO' };
    if (command.type === 'toggle') {
      if (video.paused) await video.play?.();
      else video.pause?.();
      return { ok: true, paused: Boolean(video.paused), currentTime: video.currentTime ?? 0 };
    }
    if (command.type === 'seek') {
      const delta = Number(command.seconds) || 0;
      const duration = Number.isFinite(video.duration) ? video.duration : Infinity;
      video.currentTime = Math.max(0, Math.min(duration, (Number(video.currentTime) || 0) + delta));
      return { ok: true, paused: Boolean(video.paused), currentTime: video.currentTime };
    }
    return { ok: false, code: 'UNKNOWN_PLAYER_COMMAND' };
  }

  async waitForVideoReady(video, timeoutMs = 8000) {
    if (video?.videoWidth > 0 && video?.videoHeight > 0) return video;
    await new Promise((resolve, reject) => {
      let timer = null;
      let interval = null;
      const events = ['loadedmetadata', 'loadeddata', 'canplay', 'playing', 'resize'];
      const cleanup = () => {
        if (timer) clearTimeout(timer);
        if (interval) clearInterval(interval);
        for (const event of events) video?.removeEventListener?.(event, check);
      };
      const check = () => {
        if (!video?.isConnected) {
          cleanup();
          reject(new AnimeScalerError('VIDEO_REMOVED', 'O player foi substituído enquanto o vídeo carregava.'));
          return;
        }
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          cleanup();
          resolve();
        }
      };
      for (const event of events) video?.addEventListener?.(event, check, { passive: true });
      interval = setInterval(check, 200);
      timer = setTimeout(() => {
        cleanup();
        reject(new AnimeScalerError('VIDEO_NOT_READY', 'O vídeo foi encontrado, mas ainda não carregou os metadados. Dê play e tente novamente.'));
      }, timeoutMs);
      check();
    });
    return video;
  }

  async start(rawSettings = this.settings) {
    if (this.state === 'starting') return this.getStatus();
    if (this.state !== 'stopped') await this.stop();

    this.captureMode = 'direct';
    this.settings = normalizeSettings({ ...rawSettings, enabled: true });
    this.state = 'starting';
    this.error = null;
    this.stats = null;

    try {
      this.video = findBestVideo(document);
      if (!this.video) throw new AnimeScalerError('NO_VIDEO', 'Nenhum vídeo HTML5 reproduzível foi encontrado neste frame.');
      await this.waitForVideoReady(this.video);
      this.displayTarget = this.video;
      await this.beginProcessing();
    } catch (error) {
      await this.fail(error);
    }
    return this.getStatus();
  }

  async startTabCapture({ streamId, captureFrame, settings = this.settings } = {}) {
    if (this.state === 'starting') return this.getStatus();
    if (this.state !== 'stopped') await this.stop();

    this.captureMode = 'tab';
    this.settings = normalizeSettings({ ...settings, enabled: true });
    this.state = 'starting';
    this.error = null;
    this.stats = null;

    try {
      if (globalThis.window && window.top !== window.self) {
        throw new AnimeScalerError('TAB_CAPTURE_TOP_FRAME_ONLY', 'O fallback de captura da aba precisa iniciar no frame principal.');
      }

      this.displayTarget = findCaptureTargetElement(document, {
        ...captureFrame,
        viewport: { width: globalThis.innerWidth ?? 0, height: globalThis.innerHeight ?? 0 },
      });
      if (!this.displayTarget) {
        throw new AnimeScalerError('CAPTURE_TARGET_NOT_FOUND', 'Não foi possível localizar o iframe/player correspondente no frame principal.');
      }

      this.tabCaptureSource = new TabCaptureSource();
      this.video = await this.tabCaptureSource.start({ streamId, targetElement: this.displayTarget });
      await this.beginProcessing();
    } catch (error) {
      await this.fail(error);
    }
    return this.getStatus();
  }

  async beginProcessing() {
    this.overlay = new VideoOverlay();
    this.overlay.attach(this.displayTarget);
    this.hud = new AnimeScalerHud();
    this.hud.attach(this.displayTarget);
    this.hud.setVisible(this.settings.showHud);
    this.hud.update(this.getStatus());

    this.adaptive.reset(0);
    this.observedSourceFps = null;
    this.lastFrameInterval = 41.7;
    this.throughputGate = this.settings.backend !== 'webgpu'
      ? new SustainedFpsGate({ targetFps: this.settings.maxFps, windowMs: 5000 })
      : null;
    this.engine = createProcessingEngine(this.settings);
    this.engine.diagnostics = this.diagnostics;
    this.diagnostics.event('start', {mode: this.captureMode, ...this.settings});
    await this.engine.initialize({
      canvas: this.overlay.canvas,
      video: this.video,
      settings: this.settings,
      adaptiveLevel: this.adaptive.level,
      captureMode: this.captureMode,
      presenter: this.overlay,
    });

    this.state = 'running';
    this.overlay.show();
    this.hud.update(this.getStatus());
    window.addEventListener('scroll', this.boundSync, { passive: true });
    window.addEventListener('resize', this.boundSync, { passive: true });
    this.mutationObserver.observe(document.documentElement, { childList: true, subtree: true });
    // MutationObserver on the document cannot see mutations within shadow roots.
    this.playerWatchTimer = setInterval(() => this.ensureVideoStillExists(), 750);
    this.scheduleNextFrame();
  }

  async stop() {
    clearInterval(this.playerWatchTimer);
    this.playerWatchTimer = null;
    this.waitingForPlayer = false;
    this.cancelFrame();
    this.state = 'stopped';
    this.busy = false;
    this.pausedFramePending = false;
    this.mutationObserver.disconnect();
    window.removeEventListener('scroll', this.boundSync);
    window.removeEventListener('resize', this.boundSync);
    await this.engine?.destroy();
    this.engine = null;
    this.tabCaptureSource?.destroy();
    this.tabCaptureSource = null;
    this.overlay?.destroy();
    this.overlay = null;
    this.hud?.destroy();
    this.hud = null;
    this.video = null;
    this.displayTarget = null;
    this.stats = null;
    this.framePacer = new FramePacer();
    this.throughputGate = null;
    this.lastMediaTime = null;
    this.observedSourceFps = null;
    this.lastRenderedSourceTime = null;
    return this.getStatus();
  }

  async updateSettings(rawSettings) {
    const next = normalizeSettings({ ...this.settings, ...rawSettings });
    const requiresReconfigure = next.model !== this.settings.model || next.backend !== this.settings.backend ||
      next.preset !== this.settings.preset || next.adaptive !== this.settings.adaptive;
    const recreateEngine = requiresEngineRecreation(this.settings, next);
    this.settings = next;
    this.hud?.setVisible(next.showHud);
    if (!next.enabled) return this.stop();
    if (this.state !== 'running') return this.start(next);
    if (recreateEngine) {
      await this.stop();
      return this.start(next);
    }
    if (requiresReconfigure && this.engine && this.video) {
      this.adaptive.reset(0);
      this.throughputGate?.reset();
      this.stats = null;
      this.lastRenderedSourceTime = null;
      this.overlay?.hide();
      try { await this.engine.updateSettings(next, this.video, this.adaptive.level); }
      catch (error) { await this.fail(error); }
    }
    if (next.backend !== 'webgpu' && (this.throughputGate?.targetFps !== next.maxFps)) {
      this.throughputGate = new SustainedFpsGate({ targetFps: next.maxFps, windowMs: 5000 });
    } else if (next.backend === 'webgpu') {
      this.throughputGate = null;
    }
    return this.getStatus();
  }

  scheduleNextFrame() {
    if (this.state !== 'running' || !this.video) return;
    if (typeof this.video.requestVideoFrameCallback === 'function') {
      this.frameHandle = this.video.requestVideoFrameCallback((now, metadata) => this.onVideoFrame(now, metadata));
    } else {
      this.frameHandle = requestAnimationFrame((now) => this.onVideoFrame(now, { mediaTime: this.video.currentTime }));
    }
  }

  cancelFrame() {
    if (this.frameHandle == null || !this.video) return;
    if (typeof this.video.cancelVideoFrameCallback === 'function') this.video.cancelVideoFrameCallback(this.frameHandle);
    else cancelAnimationFrame(this.frameHandle);
    this.frameHandle = null;
  }

  async onVideoFrame(_now, metadata = {}) {
    this.frameHandle = null;
    if (this.state !== 'running' || !this.video) return;
    if (this.captureMode === 'direct' && !this.video.isConnected) {
      this.ensureVideoStillExists();
      return;
    }
    this.scheduleNextFrame();
    this.syncGeometry();
    if (this.video.ended || this.video.readyState < 2) return;
    // A paused seek supplies a fresh still frame. Compare the source cursor,
    // not decoder mediaTime (which can differ from the requested seek time).
    const sourceTime=Number.isFinite(this.video.currentTime)?this.video.currentTime:
      (Number.isFinite(metadata.mediaTime)?metadata.mediaTime:null);
    if (Number.isFinite(metadata.mediaTime) && this.lastMediaTime != null) {
      const delta = (metadata.mediaTime - this.lastMediaTime) * 1000;
      if (delta >= 10 && delta <= 1000) this.observedSourceFps = 1000 / delta;
      if (delta >= 10 && delta <= 100) this.lastFrameInterval = delta;
    }
    if (Number.isFinite(metadata.mediaTime)) this.lastMediaTime = metadata.mediaTime;
    this.diagnostics.sourceFrame(sourceTime,_now,{
      model:this.settings.model,backend:this.settings.backend,captureMethod:this.captureMode,
      sourceWidth:this.video.videoWidth,sourceHeight:this.video.videoHeight,
    });
    const pausedStill=this.video.paused;
    if(this.busy) {
      if(pausedStill&&sourceTime!==null&&sourceTime!==this.lastRenderedSourceTime) {
        this.pausedFramePending=true;this.overlay?.hide();
      }
      return;
    }
    if (pausedStill) this.throughputGate?.reset();
    if(pausedStill&&(sourceTime===null||sourceTime===this.lastRenderedSourceTime))return;

    if (!pausedStill && !this.framePacer.take(_now, this.settings.maxFps)) return;
    this.busy = true;
    try {
      if (this.engine.needsReconfigure(this.video, this.adaptive.level)) {
        this.throughputGate?.reset();
        await this.engine.reconfigure(this.video, this.adaptive.level);
      }
      this.stats = await this.engine.processFrame(this.video);
      if (this.state !== 'running') return;
      this.lastRenderedSourceTime=sourceTime;
      if (this.displayTarget?.isConnected && !this.pausedFramePending) this.overlay?.show();
      this.noteProcessedFrame();
      if (!pausedStill && !this.stats.warmup && this.throughputGate) {
        const sourceFps = this.observedSourceFps ?? (1000 / Math.max(this.lastFrameInterval, 1));
        this.throughputGate.targetFps = Math.min(this.settings.maxFps, sourceFps) * 0.8;
        const throughput = this.throughputGate.record(_now);
        if (throughput.status === 'below-target') {
          this.diagnostics.event('throughput_guard', {
            backend: this.settings.backend,
            measuredFps: throughput.fps,
            targetFps: this.settings.maxFps,
            minimumFps: this.throughputGate.targetFps,
            windowMs: 5000,
          });
          await this.fail(new AnimeScalerError(
            'THROUGHPUT_BELOW_TARGET',
            `Video processing sustained ${throughput.fps.toFixed(2)} FPS; minimum is ${this.throughputGate.targetFps.toFixed(2)} FPS.`,
          ));
          return;
        }
      }

      if (this.settings.adaptive && !this.stats.warmup && !this.stats.reused) {
        const adaptive = this.adaptive.record(this.stats.totalMs, Math.max(this.lastFrameInterval, 1000 / this.settings.maxFps));
        if (adaptive.changed) {
          this.throughputGate?.reset();
          await this.engine.reconfigure(this.video, adaptive.level);
        }
      }
      this.hud?.update(this.getStatus());
    } catch (error) {
      await this.fail(error);
    } finally {
      this.busy = false;
      if(this.pausedFramePending) {
        this.pausedFramePending=false;
        if(this.state==='running'&&this.video?.paused) {
          this.cancelFrame();
          queueMicrotask(()=>this.onVideoFrame(performance.now(),{mediaTime:this.video?.currentTime}));
        }
      }
    }
  }

  noteProcessedFrame() {
    this.processedFrames += 1;
    const now = performance.now();
    const elapsed = now - this.fpsWindowStart;
    if (elapsed >= 1000) {
      this.processedFps = (this.processedFrames * 1000) / elapsed;
      this.processedFrames = 0;
      this.fpsWindowStart = now;
    }
  }

  syncGeometry() {
    this.overlay?.sync();
    this.hud?.sync();
  }

  ensureVideoStillExists() {
    if (this.state !== 'running') return;
    if (!this.displayTarget?.isConnected || this.waitingForPlayer) {
      if (this.captureMode !== 'direct') {
        void this.fail(new AnimeScalerError('VIDEO_REMOVED', 'O iframe capturado foi substituído. Ative novamente para autorizar uma nova captura.'));
        return;
      }
      this.overlay?.hide();
      this.waitingForPlayer = true;
      if (this.busy) return;
      const replacement = findBestVideo(document);
      if (!replacement || replacement.videoWidth <= 0 || replacement.videoHeight <= 0) return;
      this.cancelFrame();
      this.video = this.displayTarget = replacement;
      this.framePacer = new FramePacer();
      this.lastMediaTime = null;
      this.observedSourceFps = null;
      this.lastFrameInterval = 41.7;
      this.lastRenderedSourceTime = null;
      this.stats = null;
      this.processedFrames = 0;
      this.processedFps = 0;
      this.throughputGate?.reset();
      this.fpsWindowStart = performance.now();
      this.waitingForPlayer = false;
      this.overlay?.attach(replacement);
      this.hud?.attach(replacement);
      // Show the canvas only after a fresh frame, never the previous episode.
      this.scheduleNextFrame();
    }
  }

  async fail(error) {
    // STOP intentionally invalidates in-flight native work; its stale rejection is cancellation, not playback failure.
    if (this.state === 'stopped' && error?.code === 'STALE_SESSION') return this.getStatus();
    this.diagnostics.error(error);
    clearInterval(this.playerWatchTimer);
    this.playerWatchTimer = null;
    this.error = error instanceof Error ? error : new Error(String(error));
    this.state = 'error';
    this.overlay?.hide();
    this.hud?.update(this.getStatus());
    this.cancelFrame();
    this.mutationObserver.disconnect();
    try { await this.engine?.destroy(); } catch { /* ignore cleanup failures */ }
    this.engine = null;
    try { this.tabCaptureSource?.destroy(); } catch { /* ignore cleanup failures */ }
    this.tabCaptureSource = null;
  }
}
