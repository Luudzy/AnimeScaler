import { WORKGROUPS } from '../src/engine/workgroup.js';
import { compareSamples, chooseCandidate } from './tuning-math.js';
const $ = id => document.getElementById(id);
let report;
function trial(config) {
  return new Promise((resolve, reject) => {
    const frame = document.createElement('iframe');
    const token = crypto.randomUUID();
    const cleanup = () => { clearTimeout(timer); window.removeEventListener('message', receive); frame.remove(); };
    const receive = event => {
      if (event.origin !== location.origin || event.source !== frame.contentWindow || event.data?.token !== token) return;
      cleanup(); resolve(event.data.result);
    };
    const timer = setTimeout(() => { cleanup(); reject(new Error('Configuração excedeu 90 segundos')); }, 90000);
    window.addEventListener('message', receive);
    frame.src = `tuning-trial.html#${encodeURIComponent(JSON.stringify({...config, token}))}`;
    $('preview').replaceChildren(frame);
  });
}
$('run').onclick = async () => {
  $('run').disabled = true; $('save').disabled = true;
  const preset = $('preset').value, graphCapture = $('graph').checked, profile = $('profile').checked;
  const model = $('model').value;
  const groups = $('quick').checked ? [[8, 8]] : WORKGROUPS;
  report = {schemaVersion: 1, date: new Date().toISOString(), browser: navigator.userAgent, model, preset, graphCapture, profile, results: []};
  let reference;
  try {
    // A fresh realm/device for every candidate avoids graph capture state leaking across sessions.
    for (const precision of ['fp32', 'fp16']) for (const layout of ['NCHW', 'NHWC']) for (const workgroup of groups) {
      const config = {model, preset, precision, layout, workgroup, graphCapture, profile};
      $('result').textContent = `Testando ${precision} ${layout} ${workgroup.join('×')}…\n${JSON.stringify(report, null, 2)}`;
      let result;
      try { result = await trial(config); } catch (error) { result = {error: error.message}; }
      if (!reference && precision === 'fp32' && layout === 'NCHW' && workgroup[0] === 8) reference = result.samples;
      const correctness = compareSamples(reference, result.samples);
      delete result.samples;
      report.results.push({...config, ...result, correctness});
      if (!reference) throw new Error('Referência FP32 falhou; comparação interrompida.');
    }
    report.recommendation = chooseCandidate(report.results);
    report.notes = 'Recomendação experimental por p95 em 20 frames após 5 aquecimentos. Workgroups afetam apenas pré/pós-processamento, não as convoluções do ORT. Validação em 4096 valores de uma imagem sintética; não garante qualidade em vídeos reais. Profiling pode afetar os tempos. Sem aplicação automática.';
  } catch (error) { report.error = error.message; }
  finally { $('result').textContent = JSON.stringify(report, null, 2); $('run').disabled = false; $('save').disabled = false; }
};
$('save').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url;
  link.download = `animescaler-tuning-${Date.now()}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
};
