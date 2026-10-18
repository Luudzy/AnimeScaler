import { normalizeSettings } from '../src/core/settings.js';
import { isFsrSpatialModel, isSpatialUpscalerModel, resolveModelCompatibility } from '../src/core/local-models.js';
import { resolveBackendDescriptor } from '../src/core/backend-descriptor.js';
import { DiagnosticLog } from '../src/core/diagnostic-log.js';
import { chooseFrameForStart, chooseFrameForStatus, chooseRunningFrame } from '../src/core/frame-routing.js';
import {
  buildTabCaptureStartMessage,
  shouldFallbackToTabCapture,
} from '../src/core/tab-capture.js';
import { resolveActivationRoute } from '../src/core/activation-route.js';

const $ = (id) => document.getElementById(id);
const controls = {
  spatialDenoise: $('spatialDenoise'),
  fsrQuality: $('fsrQuality'),
  fluidVideo: $('fluidVideo'),
  status: $('status'), metrics: $('metrics'), dot: $('dot'), toggle: $('toggle'),
  backend: $('backend'), model: $('model'), maxFps: $('maxFps'), preset: $('preset'), adaptive: $('adaptive'), showHud: $('showHud'),
};
let currentStatus = { state: 'stopped' };
let currentFrameId = null;
let activationBusy = false;
const FRAME_PROBE_TIMEOUT_MS = 4500;
const diagnostics = new DiagnosticLog();

const profileBtns = document.querySelectorAll('.profile-btn');

function updateProfileButtons() {
  const currentPreset = controls.preset.value;
  profileBtns.forEach((btn) => {
    const isSelected = btn.dataset.preset === currentPreset;
    btn.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
    btn.classList.toggle('active', isSelected);
  });
}

profileBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    const preset = btn.dataset.preset;
    if (preset && controls.preset.value !== preset) {
      controls.preset.value = preset;
      controls.preset.dispatchEvent(new Event('change'));
    }
  });
});

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function sendToFrame(tabId, frameId, message) {
  return chrome.tabs.sendMessage(tabId, message, { frameId });
}

async function getFrameInfos(tabId) {
  try {
    const frames = await chrome.webNavigation.getAllFrames({ tabId });
    return frames?.length ? frames : [{ frameId: 0, parentFrameId: -1, url: '' }];
  } catch {
    return [{ frameId: 0, parentFrameId: -1, url: '' }];
  }
}

async function probeFrames(tab, timeoutMs = FRAME_PROBE_TIMEOUT_MS) {
  if (!tab?.id) return [];
  const frames = await getFrameInfos(tab.id);
  const probes = await Promise.all(frames.map(async (frame) => {
    let timer;
    try {
      const result = await Promise.race([
        sendToFrame(tab.id, frame.frameId, { type: 'ANIMESCALER_PROBE' }),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(Object.assign(
            new Error('O player não respondeu. Recarregue a página do vídeo e tente novamente.'),
            { code: 'FRAME_RESPONSE_TIMEOUT' },
          )), timeoutMs);
        }),
      ]);
      return { ...result, frameId: frame.frameId, frameUrl: frame.url, parentFrameId: frame.parentFrameId, ok: true };
    } catch (error) {
      return {
        frameId: frame.frameId,
        frameUrl: frame.url,
        parentFrameId: frame.parentFrameId,
        ok: false,
        hasVideo: false,
        videoScore: -1,
        state: 'unreachable',
        error,
      };
    } finally { clearTimeout(timer); }
  }));
  return probes;
}

async function waitForVideoFrame(tab, timeoutMs = 4500) {
  const started = performance.now();
  let probes = [];
  do {
    probes = await probeFrames(tab, Math.max(0, timeoutMs - (performance.now() - started)));
    const running = chooseRunningFrame(probes);
    if (running) return { target: running, probes };
    const target = chooseFrameForStart(probes);
    if (target) return { target, probes };
    const remaining = timeoutMs - (performance.now() - started);
    if (remaining <= 0) break;
    await new Promise((resolve) => setTimeout(resolve, Math.min(250, remaining)));
  } while (performance.now() - started < timeoutMs);
  return { target: null, probes };
}

async function loadSettings() {
  const { animeScalerSettings = {} } = await chrome.storage.local.get('animeScalerSettings');
  const saved = normalizeSettings(animeScalerSettings);
  if (![...controls.model.options].some(option => option.value === saved.model)) {
    controls.model.add(new Option(`Indisponível (preferência salva): ${saved.model}`, saved.model));
  }
  controls.model.value = saved.model;
  controls.fluidVideo.checked = saved.fluidVideo;
  controls.maxFps.value = String(saved.maxFps);
  controls.preset.value = saved.preset;
  controls.adaptive.checked = animeScalerSettings.adaptive !== false;
  controls.backend.value = saved.backend;
  controls.spatialDenoise.value = saved.spatialDenoise;
  controls.fsrQuality.value = saved.fsrQuality;
  updateBackendOptions();
  controls.showHud.checked = animeScalerSettings.showHud !== false;
  updateProfileButtons();
}

function updateBackendOptions() {
  const fsr = isFsrSpatialModel(controls.model.value);
  const spatial = isSpatialUpscalerModel(controls.model.value);
  const fsr3 = controls.model.value === 'fsr3-native';
  $('spatialDenoiseField').hidden = !spatial;
  $('fsrQualityField').hidden = !spatial && !fsr3;
  controls.fluidVideo.disabled = fsr3;
  if (fsr3) controls.fluidVideo.checked = false;
  const note = $('modelNote');
  const model = controls.model.value;
  const compatibility = resolveModelCompatibility(model, controls.backend.value);
  const detail = fsr3
    ? 'Abra Iniciar-FSR3.cmd. FSR 3.1 oficial via DirectX 12, com movimento estimado na GPU para vídeo. Experimental: depth aproximado, sem jitter de câmera nem geração de frames. Há transporte de pixels pela CPU; não é zero-copy.'
    : controls.model.value === 'asr-anime-temporal'
    ? 'Limpeza temporal na GPU em regiões estáveis. Movimentos e cortes rejeitam o histórico. Experimental; não é FSR 3 nem gera frames.'
    : controls.model.value === 'asr-anime-spatial'
    ? 'ASR protege lineart, suaviza áreas planas e limita a nitidez para reduzir halo. Não usa nem precisa de modelo Lite.'
    : fsr
    ? 'EASU + RCAS em WebGPU. Escala e aumenta a nitidez sem servidor local; não recupera detalhe perdido como um modelo de IA.'
    : model === 'realesr-anime4k-s-360'
    ? 'Reduz a entrada do RealESR para até 640×360. Anime4K completa a ampliação; pode perder detalhes finos em troca de velocidade.'
    : '';
  const status = [compatibility.reason, compatibility.releaseStatus === 'experimental' ? 'Status experimental. Sucesso técnico não promove este modelo a release.' : ''].filter(Boolean).join(' ');
  note.textContent = [status, detail].filter(Boolean).join(' ');
  const canStop = currentStatus.state === 'running' || currentStatus.state === 'starting';
  const unavailable = !compatibility.available && !canStop;
  controls.toggle.disabled = activationBusy || unavailable;
  controls.toggle.setAttribute('aria-busy', String(activationBusy));
  controls.toggle.style.cursor = activationBusy ? 'wait' : unavailable ? 'not-allowed' : '';
  const activationReason = $('activationReason');
  activationReason.textContent = unavailable ? compatibility.reason : '';
  activationReason.hidden = !unavailable || activationBusy;
}

function readSettings(enabled = true) {
  return {
    enabled,
    exclusiveViewer: false,
    fluidVideo: controls.fluidVideo.checked,
    backend: controls.backend.value,
    model: controls.model.value,
    spatialDenoise: controls.spatialDenoise.value,
    fsrQuality: controls.fsrQuality.value,
    maxFps: Number(controls.maxFps.value),
    preset: controls.preset.value,
    adaptive: controls.adaptive.checked,
    showHud: controls.showHud.checked,
  };
}

async function saveSettings(enabled) {
  const settings = readSettings(enabled);
  await chrome.storage.local.set({ animeScalerSettings: settings });
  return settings;
}

function captureLabel(status) {
  if (status?.captureMode === 'viewer') return 'Janela separada';
  return status?.captureMode === 'tab' ? 'Tab Capture' : 'Direct Video';
}

function render(status) {
  currentStatus = status || { state: 'stopped' };
  if (currentStatus.error) diagnostics.error(currentStatus.error);
  const state = currentStatus.state || 'stopped';
  controls.dot.className = `dot ${state === 'running' ? 'running' : state === 'error' ? 'error' : ''}`;
  controls.status.textContent = state === 'running' ? 'Ativo' : state === 'starting' ? 'Inicializando' : state === 'error' ? 'Erro' : 'Parado';
  if (controls.status.classList) {
    controls.status.className = `status-badge ${state}`;
  }
  controls.toggle.textContent = state === 'running' || state === 'starting' ? 'Desativar AnimeScaler' : 'Ativar AnimeScaler';
  controls.toggle.classList.toggle('stop', state === 'running' || state === 'starting');
  controls.toggle.setAttribute('aria-pressed', state === 'running' || state === 'starting' ? 'true' : 'false');

  if (state === 'running' && currentStatus.waitingForPlayer) {
    controls.metrics.textContent = 'Aguardando o próximo vídeo; a reconexão será automática.';
  } else if (state === 'running' && currentStatus.captureMode === 'viewer' && !currentStatus.stats) {
    controls.metrics.textContent = 'Janela separada ativa · use Desativar para encerrar o processamento.';
  } else if (state === 'running' && currentStatus.stats) {
    const s = currentStatus.stats;
    controls.metrics.textContent = `${captureLabel(currentStatus)} · ${s.sourceWidth}×${s.sourceHeight} → AI ${s.outputWidth}×${s.outputHeight}\n${s.modelLabel || 'Modelo'} ${String(s.modelPrecision).toUpperCase()} · ${s.totalMs.toFixed(1)} ms/frame · ${(currentStatus.processedFps || 0).toFixed(1)} fps processados (inclui quadros reutilizados pelo serviço)`;

  } else if (state === 'error') {
    controls.metrics.textContent = `${currentStatus.error?.message || 'Falha desconhecida.'}\nCódigo: ${currentStatus.error?.code || 'ERROR'}`;
  } else if (state === 'starting') {
    controls.metrics.textContent = currentStatus.captureMode === 'tab'
      ? 'CORS detectado. Preparando Tab Capture + Element Capture…'
      : 'Localizando o player e preparando WebGPU/modelo…';
  } else if (currentStatus.hasVideo && currentStatus.candidate) {
    const c = currentStatus.candidate;
    const decoded = c.videoWidth > 0 ? `${c.videoWidth}×${c.videoHeight}` : 'aguardando metadata';
    controls.metrics.textContent = `Vídeo detectado no frame ${currentFrameId ?? 0} · ${decoded}`;
  } else {
    controls.metrics.textContent = 'Abra/dê play em um vídeo HTML5 e ative o upscaler.';
  }
  updateBackendOptions();
}

async function refreshStatus() {
  const tab = await activeTab();
  if (!tab?.id) return render({ state: 'stopped' });
  let viewer;
  try { viewer = await chrome.runtime.sendMessage({ type: 'ANIMESCALER_VIEWER_STATUS', sourceTabId: tab.id }); }
  catch { /* The inline probe below remains available if viewer status cannot be read. */ }
  if (viewer?.state === 'running') {
    currentFrameId = null;
    return render(viewer);
  }
  const probes = await probeFrames(tab);
  const selected = chooseFrameForStatus(probes);
  currentFrameId = selected?.frameId ?? null;
  render(selected || { state: 'stopped' });
}

async function stopAllFrames(tab) {
  let viewer = null;
  try { viewer = await chrome.runtime.sendMessage({ type: 'ANIMESCALER_STOP_VIEWER', sourceTabId: tab.id }); }
  catch { /* An inline session may be the only active route. */ }
  const probes = await probeFrames(tab);
  const active = probes.filter((probe) => probe.ok && probe.state !== 'stopped');
  const results = await Promise.all(active.map(async (probe) => {
    try { return await sendToFrame(tab.id, probe.frameId, { type: 'ANIMESCALER_STOP' }); }
    catch { return null; }
  }));
  return results.find(Boolean) || viewer || { state: 'stopped' };
}

async function startTabCaptureFallback(tab, target, settings) {
  if (!chrome.tabCapture?.getMediaStreamId) {
    return {
      state: 'error',
      captureMode: 'tab',
      error: { code: 'TAB_CAPTURE_API_UNAVAILABLE', message: 'chrome.tabCapture não está disponível neste navegador.' },
    };
  }

  try { await sendToFrame(tab.id, target.frameId, { type: 'ANIMESCALER_STOP' }); } catch { /* cleanup best effort */ }
  render({ state: 'starting', captureMode: 'tab' });

  const frames = await getFrameInfos(tab.id);
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id,
    consumerTabId: tab.id,
  });
  const message = buildTabCaptureStartMessage({
    streamId,
    settings,
    frames,
    targetFrameId: target.frameId,
  });

  currentFrameId = 0;
  return sendToFrame(tab.id, 0, message);
}

async function startScaler(tab, target, settings) {
  if (resolveActivationRoute(settings, { state: 'stopped' }) === 'viewer') {
    const frames = await getFrameInfos(tab.id);
    return chrome.runtime.sendMessage({ type: 'ANIMESCALER_START_VIEWER', tab, target, settings: {
      ...settings, exclusiveViewer: false,
    }, frames });
  }
  settings = { ...settings, exclusiveViewer: false };
  const direct = await sendToFrame(tab.id, target.frameId, { type: 'ANIMESCALER_START', settings });
  if (!shouldFallbackToTabCapture(direct)) {
    currentFrameId = target.frameId;
    return direct;
  }
  return startTabCaptureFallback(tab, target, settings);
}

controls.toggle.addEventListener('click', async () => {
  if (activationBusy) return;
  activationBusy = true;
  updateBackendOptions();
  try {
    const tab = await activeTab();
    if (!tab?.id) throw new Error('Aba ativa indisponível.');

    const settings = readSettings();
    const route = resolveActivationRoute(settings, currentStatus);
    if (route === 'stop') {
      await saveSettings(false);
      render(await stopAllFrames(tab));
      currentFrameId = null;
    } else {
      const settings = await saveSettings(true);
      render({ state: 'starting', captureMode: 'direct' });
      const { target } = await waitForVideoFrame(tab);
      if (!target) {
        throw new Error('Nenhum vídeo HTML5 foi encontrado na página nem nos iframes. Dê play no vídeo e tente novamente.');
      }
      render(await startScaler(tab, target, settings));
    }
  } catch (error) {
    render({
      state: 'error',
      captureMode: currentStatus.captureMode,
      error: { code: error?.code || 'POPUP', message: error?.message || String(error) },
    });
  } finally {
    activationBusy = false;
    updateBackendOptions();
  }
});

for (const input of [controls.fsrQuality, controls.spatialDenoise, controls.fluidVideo, controls.backend, controls.model, controls.maxFps, controls.preset, controls.adaptive, controls.showHud]) {
  input.addEventListener('change', async () => {
    if (input === controls.preset) {
      updateProfileButtons();
    }
    updateBackendOptions();
    const enabled = currentStatus.state === 'running';
    const settings = await saveSettings(enabled);
    if (currentFrameId !== null && currentStatus.captureMode !== 'viewer') {
      const tab = await activeTab();
      if (tab?.id) {
        try { render(await sendToFrame(tab.id, currentFrameId, { type: 'ANIMESCALER_SETTINGS', settings })); } catch { /* refresh below */ }
      }
    }
  });
}

const { setupDiagnostics } = await import('./diagnostics.js');
setupDiagnostics();
await loadSettings();
await refreshStatus();
setInterval(refreshStatus, 800);
