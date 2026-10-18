import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { motionReuse } from '../src/engine/temporal-motion.js';
import { isSceneCut } from '../src/engine/temporal-history.js';
import { analyzeMotion } from './motion-analysis.js';
import { summarize } from './playback-metrics.js';

const root=fileURLToPath(new URL('../',import.meta.url));
const output=resolve(root,process.argv[2]??'diagnostics/motion-lab-20261011');
if(existsSync(output))throw new Error('Use a new output directory; previous evidence is preserved');
const source=resolve(root,'diagnostics/stability-media/rezero-ep19-360p.mp4');
const ffmpeg=resolve(root,'native/.tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe');
const baselineCommit='6fb14c4';
function command(program,args,limit=2*1024*1024) {
  const result=spawnSync(program,args,{cwd:root,timeout:30000,maxBuffer:limit,windowsHide:true});
  if(result.error)throw result.error;
  if(result.status!==0)throw new Error(`${program} exited ${result.status}: ${result.stderr?.toString()}`);
  return result.stdout;
}
const baselineSource=command('git',['show',`${baselineCommit}:src/engine/temporal-motion.js`]).toString();
const candidateSource=readFileSync(resolve(root,'src/engine/temporal-motion.js'),'utf8');
const baseline=(await import(`data:text/javascript;base64,${Buffer.from(baselineSource).toString('base64')}`)).motionReuse;
const sha=data=>createHash('sha256').update(data).digest('hex');
mkdirSync(output,{recursive:true});
const width=80,height=45,frameBytes=width*height*4,starts=[0,8,16],lags=[0,1,2,3];
const manifest={protocol:'motion-thumbnail-cpu-v1',baselineCommit,head:command('git',['rev-parse','HEAD']).toString().trim(),
  node:process.version,platform:process.platform,arch:process.arch,
  source:{path:relative(root,source).replaceAll('\\','/'),sha256:sha(readFileSync(source))},
  baselineAlgorithmSha256:sha(baselineSource),candidateAlgorithmSha256:sha(candidateSource),
  ffmpeg:command(ffmpeg,['-version']).toString().split(/\r?\n/)[0],width,height,framesPerWindow:24,fps:24,starts,lags,
  limits:{decodeTimeoutMs:30000,decoderThreads:1,filterThreads:1,outputThreads:1,maxDecodeBytes:2*1024*1024},
  trainingUse:false,gpuUsed:false,modelInference:false,warmupCallsPerImplementation:5,windows:[],
  caveats:['FFmpeg bilinear thumbnails differ from browser canvas resampling.',
    'Frame indices follow the fps=24 filter after input seek; they are not captured browser timestamps.',
    'Raw matching only: confidence history, capture cadence and age weighting are not simulated.',
    'Thumbnail residual cannot establish high-resolution AI quality or absence of ghosting.',
    'One local source, already used in prior evaluations; not an independent held-out model test.',
    'Single serial CPU pass with alternating implementation order; timings include JIT and host noise.']};
const rows=[];
for(const start of starts) {
  const args=['-nostdin','-hide_banner','-loglevel','error','-threads','1','-hwaccel','none','-ss',String(start),'-i',source,
    '-an','-sn','-dn','-filter_threads','1','-vf','fps=24,scale=80:45:flags=bilinear','-frames:v','24',
    '-pix_fmt','rgba','-threads','1','-f','rawvideo','pipe:1'];
  const decoded=command(ffmpeg,args);
  if(decoded.length!==frameBytes*24)throw new Error(`Window ${start}: expected exactly 24 frames`);
  const name=`window-${start}s.rgba`;writeFileSync(resolve(output,name),decoded);
  const frames=Array.from({length:24},(_,i)=>new Uint8ClampedArray(decoded.subarray(i*frameBytes,(i+1)*frameBytes)));
  const cuts=frames.map((f,i)=>i>0&&isSceneCut(frames[i-1],f));
  manifest.windows.push({startSeconds:start,command:[ffmpeg,...args],decodedFile:name,decodedSha256:sha(decoded),cutsAtFrameIndices:cuts.flatMap((v,i)=>v?[i]:[])});
  for(const fn of [baseline,motionReuse])for(let i=0;i<5;i++)fn(frames[0],frames[1],width,height);
  for(const lag of lags)for(let i=lag;i<frames.length;i++) {
    const reference=frames[i-lag],current=frames[i];
    const row={startSeconds:start,referenceIndex:i-lag,currentIndex:i,lagFrames:lag,nominalLagMs:lag*1000/24,
      identical:reference.every((v,j)=>v===current[j]),interveningSceneCut:cuts.slice(i-lag+1,i+1).some(Boolean)};
    const order=rows.length%2?[['candidate',motionReuse],['baseline',baseline]]:[['baseline',baseline],['candidate',motionReuse]];
    for(const [name,fn] of order) {
      const begin=performance.now(),reuse=fn(reference,current,width,height),elapsedMs=performance.now()-begin;
      row[name]={...analyzeMotion(reference,current,width,height,reuse),elapsedMs};
      if(name==='baseline')row.baselineResult=reuse;
      else row.candidateResult=reuse;
    }
    row.exactMatch=JSON.stringify(row.baselineResult.tiles)===JSON.stringify(row.candidateResult.tiles)
      &&row.baselineResult.mask.every((v,j)=>v===row.candidateResult.mask[j]);
    delete row.baselineResult;delete row.candidateResult;rows.push(row);
  }
}
const summaries=[];
for(const start of starts)for(const lag of lags) {
  const selected=rows.filter(r=>r.startSeconds===start&&r.lagFrames===lag);
  const sameScene=selected.filter(r=>!r.interveningSceneCut);
  const group={startSeconds:start,lagFrames:lag,pairs:selected.length,sameScenePairs:sameScene.length,
    identicalPairs:selected.filter(r=>r.identical).length,changedResults:selected.filter(r=>!r.exactMatch).length};
  for(const name of ['baseline','candidate'])group[name]=Object.fromEntries(
    ['coverage','beforeErosionCoverage','erosionLoss','supportedMaxChannelErrorMean','supportedMaxChannelErrorMax','elapsedMs','acceptedTiles','movedTiles']
      .map(key=>[key,summarize(sameScene.map(r=>r[name][key]))]));
  summaries.push(group);
}
writeFileSync(resolve(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
writeFileSync(resolve(output,'results.json'),JSON.stringify({pairs:rows.length,changedResults:rows.filter(r=>!r.exactMatch).length,summaries,rows},null,2)+'\n');
console.log(JSON.stringify({output,pairs:rows.length,changedResults:rows.filter(r=>!r.exactMatch).length,
  groups:summaries.map(g=>({start:g.startSeconds,lag:g.lagFrames,identical:g.identicalPairs,
    baselineCoverage:g.baseline.coverage.mean,candidateCoverage:g.candidate.coverage.mean,
    erosionLoss:g.candidate.erosionLoss.mean,baselineMs:g.baseline.elapsedMs.mean,candidateMs:g.candidate.elapsedMs.mean}))},null,2));
