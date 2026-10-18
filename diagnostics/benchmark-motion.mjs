// Serial CPU-only replay of the fixed, hash-verified thumbnail corpus.
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cpus, release } from 'node:os';
import { performance } from 'node:perf_hooks';
import { motionReuse } from '../src/engine/temporal-motion.js';
import { summarize } from './playback-metrics.js';

const root=fileURLToPath(new URL('../',import.meta.url));
const input=resolve(root,process.argv[2]??'diagnostics/motion-lab-20261011-converged');
const output=resolve(root,process.argv[3]??'diagnostics/motion-cache-20261011');
if(existsSync(output))throw new Error('Use a new output directory');
const sha=data=>createHash('sha256').update(data).digest('hex');
const manifest=JSON.parse(readFileSync(resolve(input,'manifest.json'),'utf8'));
const baselineCommit='a12bcda';
const git=spawnSync('git',['show',`${baselineCommit}:src/engine/temporal-motion.js`],{cwd:root,windowsHide:true,timeout:10000,maxBuffer:1024*1024});
if(git.error)throw git.error;
if(git.status!==0)throw new Error('Cannot read the fixed baseline');
const baselineSource=git.stdout.toString();
const candidateSource=readFileSync(resolve(root,'src/engine/temporal-motion.js'),'utf8');
const baseline=(await import(`data:text/javascript;base64,${Buffer.from(baselineSource).toString('base64')}`)).motionReuse;
if(manifest.width!==80||manifest.height!==45||manifest.framesPerWindow!==24||manifest.windows.length!==3)
  throw new Error('Expected the fixed 3-window 80x45 corpus');
const width=80,height=45,frameBytes=width*height*4,pairs=[];
for(const window of manifest.windows) {
  const decoded=readFileSync(resolve(input,window.decodedFile));
  if(sha(decoded)!==window.decodedSha256||decoded.length!==24*frameBytes)throw new Error('Thumbnail corpus changed');
  const frames=Array.from({length:24},(_,i)=>new Uint8ClampedArray(decoded.subarray(i*frameBytes,(i+1)*frameBytes)));
  for(const lag of [0,1,2,3])for(let i=lag;i<24;i++)pairs.push({startSeconds:window.startSeconds,lagFrames:lag,
    referenceIndex:i-lag,currentIndex:i,reference:frames[i-lag],current:frames[i]});
}
const digest=result=>sha(Buffer.concat([Buffer.from(JSON.stringify(result.tiles)),Buffer.from(result.mask)]));
const begin=performance.now(),deadlineMs=30000;
const checkDeadline=()=>{if(performance.now()-begin>deadlineMs)throw new Error('CPU benchmark deadline exceeded');};
const memorySamples=[process.memoryUsage()],cpuBegin=process.cpuUsage();
for(const pair of pairs) {checkDeadline();pair.expected=digest(baseline(pair.reference,pair.current,width,height));}
const warmup=pairs.filter(p=>p.lagFrames>0).filter((_,i)=>i%6===0).slice(0,30);
for(const fn of [baseline,motionReuse])for(const p of warmup) {checkDeadline();fn(p.reference,p.current,width,height);}
const rows=[];
for(let round=0;round<5;round++) {
  const ordered=round%2?pairs.toReversed():pairs;
  for(let i=0;i<ordered.length;i++) {
    const pair=ordered[i];checkDeadline();
    const row={round,startSeconds:pair.startSeconds,lagFrames:pair.lagFrames,referenceIndex:pair.referenceIndex,currentIndex:pair.currentIndex};
    const implementations=(round+i)%2?[['candidate',motionReuse],['baseline',baseline]]:[['baseline',baseline],['candidate',motionReuse]];
    for(const [name,fn] of implementations) {
      const t=performance.now(),result=fn(pair.reference,pair.current,width,height);
      row[`${name}Ms`]=performance.now()-t;
      if(digest(result)!==pair.expected)throw new Error(`Result changed in ${name}: ${JSON.stringify(row)}`);
    }
    rows.push(row);
  }
  memorySamples.push(process.memoryUsage());
}
const summaries=[];
for(let round=0;round<5;round++)for(const group of ['all','moving','identity']) {
  const selected=rows.filter(r=>r.round===round&&(group==='all'||(group==='moving'?r.lagFrames>0:r.lagFrames===0)));
  summaries.push({round,group,pairs:selected.length,baseline:summarize(selected.map(r=>r.baselineMs)),candidate:summarize(selected.map(r=>r.candidateMs))});
}
const metadata={protocol:'motion-score-cache-cpu-v1',baselineCommit,baselineAlgorithmSha256:sha(baselineSource),candidateAlgorithmSha256:sha(candidateSource),
  input:relative(root,input).replaceAll('\\','/'),corpusManifestSha256:sha(readFileSync(resolve(input,'manifest.json'))),
  windows:manifest.windows.map(w=>({startSeconds:w.startSeconds,decodedSha256:w.decodedSha256})),source:manifest.source,
  node:process.version,v8:process.versions.v8,platform:process.platform,osRelease:release(),cpu:cpus()[0]?.model,
  rounds:5,uniquePairs:pairs.length,timedPairs:rows.length,warmupCallsPerImplementation:warmup.length,
  elapsedMs:performance.now()-begin,deadlineMs,changedResults:0,cpuUsage:process.cpuUsage(cpuBegin),memorySamples,
  sampledRssMaxBytes:Math.max(...memorySamples.map(m=>m.rss)),scoreScratchBytes:39*39*8,gpuUsed:false,modelInference:false,
  caveats:['Single-process serial Node execution; V8 JIT and host noise affect timings.',
    'Result hashing, baseline oracle generation and warmup are outside per-call timing.',
    'RSS samples are whole-process readings, not allocation profiling or a measured peak.',
    'Thumbnail motion timing is not player FPS, GPU inference latency or visual quality.']};
mkdirSync(output,{recursive:true});
writeFileSync(resolve(output,'manifest.json'),JSON.stringify(metadata,null,2)+'\n');
writeFileSync(resolve(output,'results.json'),JSON.stringify({summaries,rows},null,2)+'\n');
console.log(JSON.stringify({output,changedResults:0,elapsedMs:metadata.elapsedMs,moving:summaries.filter(s=>s.group==='moving'),sampledRssMaxBytes:metadata.sampledRssMaxBytes},null,2));
