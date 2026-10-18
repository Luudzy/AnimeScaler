const fs = require('node:fs');
function read(name) {
  const raw=JSON.parse(fs.readFileSync(`diagnostics/${name}.json`,'utf8'));
  if(!raw.result.startsWith('PASS\n')) throw new Error(`Invalid run: ${name}`);
  return JSON.parse(raw.result.slice(5));
}
const off=read('fsr-denoise-degraded-off-144');
const balanced=read('fsr-denoise-degraded-balanced-144');
const strong=read('fsr-denoise-degraded-strong-144');
const cleanOff=read('fsr-denoise-clean-off-144');
const cleanBalanced=read('fsr-denoise-clean-balanced-144');
function compare(base,runs) {
  return base.clips.flatMap((clip,index)=>['fsr1-spatial','asr-anime-spatial'].map(model=>{
    const reference=clip.metrics[model];
    return {clip:clip.clip,model,off:reference,...Object.fromEntries(Object.entries(runs).map(([label,run])=>{
      if(run.clips[index].clip!==clip.clip || run.framesPerClip!==base.framesPerClip || run.degradation!==base.degradation || run.fsrQuality!==base.fsrQuality) throw new Error('Protocol mismatch');
      for(const key of ['mae','psnr','temporalMae']) if(Math.abs(run.clips[index].metrics.bilinear[key]-clip.metrics.bilinear[key])>1e-12) throw new Error('Inputs/bilinear reference differ between runs');
      const metrics=run.clips[index].metrics[model];
      return [label,{...metrics,maeChangePercent:(metrics.mae/reference.mae-1)*100,temporalChangePercent:(metrics.temporalMae/reference.temporalMae-1)*100,psnrChange:metrics.psnr-reference.psnr}];
    }))};
  }));
}
const report={scope:'Matched 144-frame video pairs; synthetic noise/JPEG on degraded tests; not real stream codec capture.',degraded:compare(off,{balanced,strong}),clean:compare(cleanOff,{balanced:cleanBalanced})};
fs.writeFileSync('diagnostics/fsr-denoise-comparison.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
