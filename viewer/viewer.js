import { FramePacer } from '../src/core/frame-pacer.js';
import { DiagnosticLog } from '../src/core/diagnostic-log.js';
import { AdaptiveQualityController } from '../src/core/adaptive-quality.js';
import { mapCssRectToCapturePixels, composeCaptureGeometry } from '../src/core/capture-geometry.js';
import { normalizeSettings } from '../src/core/settings.js';
import { ADDED_MODELS, isAddedModel } from '../src/core/local-models.js';
import { buildTabCaptureConstraints } from '../src/core/tab-capture.js';
import { activeViewerKey, viewerSessionKey } from '../src/core/viewer-session.js';
import { NativeEngine } from '../src/engine/native-engine.js';
import { restoreExclusiveSource } from '../src/core/exclusive-viewer.js';
import { createProcessingEngine } from '../src/engine/engine-factory.js';
import { renderStillComparison } from './still-comparison.js';
import { ViewerFullscreen } from './fullscreen.js';
import { formatPresentationMetrics } from '../src/core/presentation-label.js';
import { handleViewerKeydown } from './viewer-hotkeys.js';

const $ = (id) => document.getElementById(id);
let output = $('output');
let pendingSettings = null;
const capture = $('capture');
const status = $('status');
const metrics = $('metrics');
const stage = $('stage');
const modelBrand = $('model-brand');

const params = new URLSearchParams(location.search);
const sessionId = params.get('session');
let session = null;
let stream = null;
let engine = null;
let settings = normalizeSettings();
let crop = null;
let stats = null;
let running = true;
let busy = false;
let comparing = false;
let comparisonBusy = false;
let fullscreenControl = null;
let idleTimer = null;
function revealControls() {
  stage.classList.remove('ui-idle');
  clearTimeout(idleTimer);
  idleTimer=setTimeout(()=>{
    if(running && !comparing && !stage.classList.contains('error') && !document.querySelector('#controls:focus-within'))stage.classList.add('ui-idle');
  },2500);
}
for(const event of ['pointermove','pointerdown','keydown','focusin','focusout'])document.addEventListener(event,revealControls,{passive:true});
// Pointer activation should not pin controls forever; keyboard focus remains.
$('controls').addEventListener('pointerup',event=>{
  if(event.target.closest('button'))setTimeout(()=>{
    if(document.activeElement===event.target)event.target.blur();
  },0);
});
capture.addEventListener('resize',()=>{lastGeometryAt=0;});
let latestFramePending = false;
let frameHandle = null;
const framePacer = new FramePacer();
let lastGeometryAt = 0;
let processedFrames = 0;
let fpsStartedAt = performance.now();
let processedFps = 0;
const adaptive = new AdaptiveQualityController();
const diagnostics = new DiagnosticLog();
const targetFrameMs = () => 1000 / settings.maxFps;

function updateModelBrand() {
  const model = settings.model;
  const label = ADDED_MODELS[model]?.label
    ?? ({ adore: 'Adore', 'adore-lite': 'Adore Lite', 'adore-lite-v3': 'Adore Lite V3', animejanai: 'AnimeJaNai', 'fsr1-spatial': 'AMD FSR 1 Spatial', 'asr-anime-spatial': 'ASR Anime Spatial', 'asr-anime-temporal': 'ASR Anime Temporal (experimental)' }[model] ?? model);
  const scale = ADDED_MODELS[model]?.scale ?? 2;
  modelBrand.textContent = model === 'fsr3-native' ? label : `${label} ${scale}×`;
}

function setStatus(text, detail = '') {
  status.textContent = text;
  metrics.textContent = detail;
}

async function waitForSession(timeoutMs = 7000) {
  if (!sessionId) throw new Error('VIEWER_SESSION_MISSING: URL sem session id.');
  const key = viewerSessionKey(sessionId);
  const started = performance.now();
  while (performance.now() - started < timeoutMs) {
    const stored = await chrome.storage.session.get(key);
    if (stored[key]) return stored[key];
    // Retry covers the viewer executing before windows.create has returned
    // its consumer tab ID to the worker. Sender identity is checked there.
    try { await chrome.runtime.sendMessage({type:'ANIMESCALER_VIEWER_READY'}); } catch { /* Worker may still be attaching. */ }
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
  throw new Error('VIEWER_SESSION_TIMEOUT: configuração da captura não chegou ao viewer.');
}

async function waitForVideo(timeoutMs = 8000) {
  if (capture.videoWidth > 0 && capture.videoHeight > 0) return;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('TAB_CAPTURE_NOT_READY: a captura não entregou frames a tempo.'));
    }, timeoutMs);
    const events = ['loadedmetadata', 'loadeddata', 'playing', 'resize'];
    const check = () => {
      if (capture.videoWidth > 0 && capture.videoHeight > 0) {
        cleanup();
        resolve();
      }
    };
    const cleanup = () => {
      clearTimeout(timer);
      for (const event of events) capture.removeEventListener(event, check);
    };
    for (const event of events) capture.addEventListener(event, check, { passive: true });
    check();
  });
}

async function queryGeometry() {
  const chain = session.captureChain;
  const geometry = chain?.length ? composeCaptureGeometry(await Promise.all(chain.map((frame,i) =>
    chrome.tabs.sendMessage(session.sourceTabId,
      {type:'ANIMESCALER_CAPTURE_GEOMETRY',sessionId,captureFrame:chain[i+1] ?? {frameId:0}},
      {frameId:frame.frameId})))) : await chrome.tabs.sendMessage(
    session.sourceTabId,
    { type: 'ANIMESCALER_CAPTURE_GEOMETRY', sessionId, captureFrame: session.captureFrame },
    { frameId: 0 },
  );
  if (!geometry?.rect || !geometry?.viewport) throw new Error('CAPTURE_TARGET_NOT_FOUND: player não está visível na aba original.');
  const mapped = mapCssRectToCapturePixels({
    rect: geometry.rect,
    viewport: geometry.viewport,
    video: { width: capture.videoWidth, height: capture.videoHeight },
  });
  if (!mapped) throw new Error('CAPTURE_TARGET_OFFSCREEN: o player saiu da área visível da aba original.');
  return mapped;
}

async function sendPlayerCommand(command) {
  try {
    return await chrome.tabs.sendMessage(
      session.sourceTabId,
      { type: 'ANIMESCALER_PLAYER_COMMAND', sessionId, command },
      { frameId: session.targetFrameId },
    );
  } catch {
    return null;
  }
}

function noteFrame() {
  processedFrames += 1;
  const now = performance.now();
  const elapsed = now - fpsStartedAt;
  if (elapsed >= 1000) {
    processedFps = processedFrames * 1000 / elapsed;
    processedFrames = 0;
    fpsStartedAt = now;
  }
}

function updateHud() {
  if (!stats || !crop) return;
  const presentation = settings.fluidVideo && engine.backend !== 'webgpu' ? formatPresentationMetrics(engine.lastPresentationMetrics) : '';
  setStatus(
    `Viewer ${engine.backend.toUpperCase()} ativo${settings.fluidVideo && engine.backend!=='webgpu' ? ' · híbrido experimental' : ''}`,
    `${crop.width}×${crop.height} → AI ${stats.outputWidth}×${stats.outputHeight}\n${stats.modelLabel || "Modelo"} ${String(stats.modelPrecision).toUpperCase()} · ${stats.totalMs.toFixed(1)} ms/frame · ${processedFps.toFixed(1)} processados/s${presentation ? `\n${presentation}` : ''}\nPreset ${settings.preset} · Adaptive ${settings.adaptive ? 'ON' : 'OFF'}${stats.fsrQuality ? `\nFSR ${stats.fsrQuality} · ruído ${stats.spatialDenoise}` : ''}`,
  );
}

async function processCapturedFrame(now) {
  if (!running || comparing) return;
  if (busy || !latestFramePending || !framePacer.take(now, settings.maxFps)) return;
  latestFramePending = false;
  busy = true;
  try {
    if (pendingSettings) {
      const next = pendingSettings; pendingSettings = null;
      await configureSettings(next);
    }
    if (!crop || now - lastGeometryAt > 500) {
      const nextCrop = await queryGeometry();
      lastGeometryAt = now;
      if (!crop || engine.needsReconfigure(capture, adaptive.level, nextCrop)) {
        await engine.reconfigure(capture, adaptive.level, nextCrop);
      }
      crop = nextCrop;
    }

    stats = await engine.processFrame(capture, crop);
    noteFrame();
    if (settings.adaptive && !stats.warmup && !stats.reused) {
      const change = adaptive.record(stats.totalMs, targetFrameMs());
      if (change.changed) await engine.reconfigure(capture, change.level, crop);
    }
    stage.classList.remove('error');
    updateHud();
  } catch (error) {
    diagnostics.error(error);
    if (!String(error?.message).startsWith('CAPTURE_TARGET_') && ['rocm','tensorrt'].includes(engine?.backend)) {
      try {
        await initializeEngine('webgpu');
        setStatus('WebGPU ativo', 'ROCm interrompido; processamento transferido para WebGPU.');
        return;
      } catch (fallbackError) { diagnostics.error(fallbackError); error = fallbackError; }
    }
    stage.classList.add('error');
    setStatus('Erro no viewer', `${error?.code || error?.name || 'ERROR'}: ${error?.message || String(error)}`);
  } finally {
    busy = false;
    // A newer frame may have arrived during inference. Process the current
    // video immediately instead of waiting an extra display refresh interval.
    if (running && !comparing && latestFramePending) queueMicrotask(() => processCapturedFrame(performance.now()));
  }
}

function capturedFrameAvailable(now) {
  frameHandle = null;
  if (!running) return;
  scheduleFrame();
  if (comparing) return;
  latestFramePending = true;
  if (crop && settings.fluidVideo && ['rocm','tensorrt'].includes(engine?.backend)) {
    void engine.presentFrame(capture,crop).catch(error => diagnostics.error(error));
  }
  processCapturedFrame(now);
}

function scheduleFrame() {
  if (!running) return;
  if (typeof capture.requestVideoFrameCallback === 'function') {
    frameHandle = capture.requestVideoFrameCallback(capturedFrameAvailable);
  } else {
    frameHandle = requestAnimationFrame(capturedFrameAvailable);
  }
}

function replaceOutput() {
  const next = output.cloneNode(false);
  output.replaceWith(next); output = next;
}

async function initializeEngine(backend) {
  await engine?.destroy();
  if (!running) return;
  replaceOutput();
  engine = backend !== 'webgpu' ? new NativeEngine() : createProcessingEngine(settings);
  engine.backend = backend;
  engine.diagnostics = diagnostics;
  try {
    const engineSettings = backend === 'webgpu' && settings.model.startsWith('adore-lite') ? {...settings,model:'adore'} : settings;
    await engine.initialize({ canvas: output, video: capture, settings:engineSettings, adaptiveLevel: adaptive.level, sourceRect: crop });
    adaptive.reset(engine.currentSize?.adaptiveLevel??0);
    if (!running) await engine.destroy();
  } catch (error) {
    if (!running) return;
    if (backend === 'webgpu' || isAddedModel(settings.model)) throw error;
    diagnostics.error(error);
    diagnostics.event('backend_fallback', { backend: 'webgpu', message: `${backend} indisponível: ${error.message}` });
    setStatus('Usando WebGPU', `${backend} indisponível: ${error.message}. Modelo WebGPU: ${settings.model.startsWith('adore-lite')?'Adore':settings.model}.`);
    await initializeEngine('webgpu');
  }
}

async function configureSettings(next) {
  const previous = settings;
  settings = normalizeSettings({ ...settings, ...next, enabled: true });
  updateModelBrand();
  adaptive.reset(0); stats = null;
  if (settings.backend !== previous.backend || settings.model !== previous.model) {
    await initializeEngine(settings.backend);
  } else await engine.updateSettings(engine.backend==='webgpu' && settings.model.startsWith('adore-lite') ? {...settings,model:'adore'} : settings, capture, adaptive.level, crop);
}

async function applySettings(next) {
  if (next.enabled === false) { await cleanup(); window.close(); return; }
  pendingSettings = next;
}

async function cleanup() {
  if (!running) return;
  running = false;
  clearTimeout(idleTimer);
  chrome.windows.onBoundsChanged?.removeListener(onWindowBoundsChanged);
  if (frameHandle != null) {
    if (typeof capture.cancelVideoFrameCallback === 'function') capture.cancelVideoFrameCallback(frameHandle);
    else cancelAnimationFrame(frameHandle);
  }
  try { await engine?.destroy(); } catch { /* no-op */ }
  for (const track of stream?.getTracks?.() ?? []) {
    try { track.stop(); } catch { /* no-op */ }
  }
  if (session) {
    try {
      await restoreExclusiveSource(session);
      await chrome.storage.session.remove([
        viewerSessionKey(sessionId),
        activeViewerKey(session.sourceTabId),
      ]);
    } catch { /* page may already be unloading */ }
  }
}

async function init() {
  session = await waitForSession();
  fullscreenControl = new ViewerFullscreen(chrome.windows,session.viewerWindowId,session.windowedBounds);
  chrome.windows.onBoundsChanged?.addListener(onWindowBoundsChanged);
  await refreshFullscreenButton();
  settings = normalizeSettings({ ...session.settings, enabled: true });
  updateModelBrand();
  setStatus('Capturando aba original…');
  stream = await navigator.mediaDevices.getUserMedia(buildTabCaptureConstraints(session.streamId));
  capture.srcObject = stream;
  await capture.play();
  await waitForVideo();
  crop = await queryGeometry();
  adaptive.reset(0);
  diagnostics.event('start', {mode: 'viewer', ...settings});
  await initializeEngine(settings.backend);
  setStatus(`Viewer ${engine.backend.toUpperCase()} ativo`, 'Inicializando o primeiro frame do modelo…');
  scheduleFrame();
  $('compare').disabled = false;
  revealControls();
}

async function endComparison() {
  $('original').hidden = true;
  $('original').width = $('original').height = 1;
  $('comparison-toggle').hidden = true;
  $('compare').textContent = 'Comparar';
  for (const id of ['back','play','forward']) $(id).disabled = false;
  comparing = false;
  revealControls();
  latestFramePending = false;
  framePacer.reset?.();
  if (running) {
    await capture.play();
    if (frameHandle == null) scheduleFrame();
    updateHud();
  }
}

$('compare').addEventListener('click', async () => {
  if (comparisonBusy || !running) return;
  comparisonBusy = true;
  $('compare').disabled = true;
  try {
    if (comparing) { await endComparison(); return; }
    comparing = true;
    latestFramePending = false;
    for (const id of ['back','play','forward']) $(id).disabled = true;
    setStatus('Preparando comparação…','Congelando o mesmo quadro para original e IA.');
    const started = performance.now();
    while (busy || engine.presentBusy) {
      if (!running || performance.now()-started > 15000) throw new Error('Não foi possível concluir o quadro pendente.');
      await new Promise(resolve => setTimeout(resolve,20));
    }
    if (!running) return;
    // Resolve geometry while capture still advances; freeze only after the
    // source frame and current viewport are available.
    crop = await queryGeometry();
    if (engine.needsReconfigure(capture,adaptive.level,crop)) await engine.reconfigure(capture,adaptive.level,crop);
    stats = await renderStillComparison({capture,original:$('original'),engine,crop});
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    if (!running) return;
    $('compare').textContent = 'Voltar ao vídeo';
    $('comparison-toggle').hidden = false;
    $('comparison-toggle').textContent = 'Mostrar original';
    setStatus('Comparação · IA',`${crop.width}×${crop.height} → ${stats.outputWidth}×${stats.outputHeight}\nMesmo quadro e enquadramento · reprodução da origem continua.`);
  } catch (error) {
    diagnostics.error(error);
    await endComparison();
    setStatus('Comparação indisponível',error.message);
  } finally {
    comparisonBusy = false;
    $('compare').disabled = !running;
  }
});

$('comparison-toggle').addEventListener('click', () => {
  const original = $('original');
  original.hidden = !original.hidden;
  $('comparison-toggle').textContent = original.hidden ? 'Mostrar original' : 'Mostrar IA';
  status.textContent = original.hidden ? 'Comparação · IA' : 'Comparação · original ampliado pelo navegador';
});

$('back').addEventListener('click', () => sendPlayerCommand({ type: 'seek', seconds: -10 }));
$('forward').addEventListener('click', () => sendPlayerCommand({ type: 'seek', seconds: 10 }));
$('play').addEventListener('click', () => sendPlayerCommand({ type: 'toggle' }));
async function refreshFullscreenButton() {
  if(!session)return;
  const current=await chrome.windows.get(session.viewerWindowId);
  const active=current.state==='fullscreen'||!!document.fullscreenElement;
  $('fullscreen').textContent=active?'Sair da tela cheia':'Tela cheia';
  $('fullscreen').setAttribute('aria-pressed',String(active));
}
function onWindowBoundsChanged(current) {
  if(current.id===session?.viewerWindowId)void refreshFullscreenButton().catch(error=>diagnostics.error(error));
}
async function changeFullscreen(exitOnly=false) {
  try {
    if(document.fullscreenElement)await document.exitFullscreen();
    else if(fullscreenControl)await (exitOnly?fullscreenControl.exit():fullscreenControl.toggle());
    await refreshFullscreenButton();revealControls();
  }catch(error){diagnostics.error(error);setStatus('Falha ao alterar tela cheia',error.message);}
}
$('fullscreen').addEventListener('click',()=>changeFullscreen());
stage.addEventListener('dblclick',event=>{if(!event.target.closest('button,nav'))void changeFullscreen();});
$('info').addEventListener('click',()=>{
  const hidden=stage.classList.toggle('info-hidden');$('info').setAttribute('aria-pressed',String(!hidden));
});
$('close').addEventListener('click', () => window.close());

document.addEventListener('keydown', (event) => {
  handleViewerKeydown(event, { comparing, comparisonBusy }, {
    closeComparison: () => $('compare').click(),
    exitFullscreen: () => void changeFullscreen(true),
    toggleFullscreen: () => void changeFullscreen(),
    toggleInfo: () => $('info').click(),
    toggleComparison: () => $('compare').click(),
    togglePlayback: () => sendPlayerCommand({ type: 'toggle' }),
    seek: seconds => sendPlayerCommand({ type: 'seek', seconds }),
  });
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.animeScalerSettings?.newValue) {
    applySettings(changes.animeScalerSettings.newValue).catch((error) => {
      diagnostics.error(error);
      setStatus('Falha ao trocar modelo', error?.message || String(error));
    });
  }
});

window.addEventListener('pagehide', () => { cleanup(); }, { once: true });

init().catch(async (error) => {
  diagnostics.error(error);
  await cleanup();
  stage.classList.add('error');
  setStatus('Falha ao iniciar', `${error?.code || error?.name || 'ERROR'}: ${error?.message || String(error)}`);
});
