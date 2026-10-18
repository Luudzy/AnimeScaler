import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { cadenceEvents, replayTemporal } from './temporal-replay.js';
import { analyzePresentation, summarize } from './playback-metrics.js';

const root=fileURLToPath(new URL('../',import.meta.url));
const input=resolve(root,process.argv[2]??'diagnostics/motion-lab-20261011-converged');
const output=resolve(root,process.argv[3]??'diagnostics/temporal-replay-20261011');
if(existsSync(output))throw new Error('Use a new output directory');
const sha=data=>createHash('sha256').update(data).digest('hex');
const corpus=JSON.parse(readFileSync(resolve(input,'manifest.json'),'utf8'));
if(corpus.width!==80||corpus.height!==45||corpus.framesPerWindow!==24||corpus.windows.length!==3)
  throw new Error('Expected the fixed 3-window 80x45 corpus');
const start=performance.now(),scenarios=[];
for(const window of corpus.windows) {
  const decoded=readFileSync(resolve(input,window.decodedFile)),frameBytes=80*45*4;
  if(decoded.length!==24*frameBytes||sha(decoded)!==window.decodedSha256)throw new Error('Thumbnail corpus changed');
  const frames=Array.from({length:24},(_,i)=>new Uint8ClampedArray(decoded.subarray(i*frameBytes,(i+1)*frameBytes)));
  for(const maxFps of [12,24])for(const latencyMs of [30,45,90]) {
    if(performance.now()-start>30000)throw new Error('Replay deadline exceeded');
    const events=cadenceEvents({frameCount:24,sourceFps:24,maxFps,latencyMs});
    const result=replayTemporal({frames,events});
    const presentations=result.rows.filter(r=>r.type==='present').map(r=>({atMs:r.atMs,frameIndex:r.frameIndex,...r.metrics}));
    const captures=result.rows.filter(r=>r.type==='capture').length;
    const completions=result.rows.filter(r=>r.type==='complete');
    const analysis=analyzePresentation(presentations);
    if(result.draws!==24||result.pendingCaptures!==0||result.uploads!==completions.filter(r=>r.accepted).length||
      analysis.missingComponents||analysis.inconsistentComponents)throw new Error('Replay invariant failed');
    scenarios.push({startSeconds:window.startSeconds,maxFps,latencyMs,sourceFps:24,sourceWindowMs:1000,captures,
      accepted:completions.filter(r=>r.accepted).length,rejected:completions.filter(r=>!r.accepted).length,
      completedAfterLastPresentation:completions.filter(r=>r.atMs>23*1000/24).length,
      draws:result.draws,uploads:result.uploads,pendingCaptures:result.pendingCaptures,
      aiCoverage:summarize(presentations.map(r=>r.aiCoverage)),historyAgeMs:summarize(presentations.map(r=>r.historyAgeMs)),
      contributionAnalysis:analysis,rows:result.rows});
  }
}
const policies=['src/engine/temporal-presenter.js','src/engine/temporal-motion.js','src/engine/temporal-history.js','src/engine/temporal-budget.js','src/core/frame-pacer.js'];
const manifest={protocol:'temporal-presenter-cpu-replay-v1',node:process.version,input:relative(root,input).replaceAll('\\','/'),
  corpusManifestSha256:sha(readFileSync(resolve(input,'manifest.json'))),source:corpus.source,
  windows:corpus.windows.map(w=>({startSeconds:w.startSeconds,decodedSha256:w.decodedSha256})),
  policyHashes:Object.fromEntries(policies.map(p=>[p,sha(readFileSync(resolve(root,p)))])),
  sourceFps:24,frameCountPerWindow:24,maxFpsScenarios:[12,24],latencyMsScenarios:[30,45,90],scenarioCount:scenarios.length,
  serializedCaptures:true,eventTieOrder:['complete','present','capture'],completionsDrained:true,
  aiPixels:'dimension-only sentinel; no AI image or final raster is evaluated',gpuUsed:false,modelInference:false,
  elapsedMs:performance.now()-start,deadlineMs:30000,
  caveats:['An artificial callback schedule with fixed latency; not NativeEngine transport or playback scheduling.',
    'Actual TemporalPresenter policies run with copied fixed thumbnails; canvas and drawing are stubs.',
    'Completions after the last presentation are drained but add no presentation sample.',
    'Each 1-second window starts with empty history, including warmup effects.',
    'Coverage is a policy metric, not a measure of high-resolution image quality or ghosting.',
    'The source is reused evaluation media, not an independent held-out model test.']};
mkdirSync(output,{recursive:true});
writeFileSync(resolve(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
writeFileSync(resolve(output,'results.json'),JSON.stringify({scenarios},null,2)+'\n');
console.log(JSON.stringify({output,scenarios:scenarios.map(({startSeconds,maxFps,latencyMs,captures,aiCoverage,contributionAnalysis})=>
  ({startSeconds,maxFps,latencyMs,captures,aiCoverageMean:aiCoverage.mean,ageLossMean:contributionAnalysis.lossMean.age}))},null,2));
