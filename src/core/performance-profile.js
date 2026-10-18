export function profileKey(info, settings, width, height) {
  const identity = [3, info.backend, info.gpu, info.architecture, info.revisions?.[settings.model],
    settings.model,settings.preset,settings.maxFps,settings.fluidVideo,width,height];
  // Same weights, different execution cost: do not restore the old RealESR
  // downscale rung after enabling exact heads and bounded HIP capture.
  if (info.backend === 'rocm' && settings.model === 'realesr-anime'
      && info.metadata?.[settings.model]?.capture_graph_max_pixels > 0) {
    identity.push('realesr-exact-heads-v1');
  }
  return 'native-profile:' + JSON.stringify(identity);
}

export function restoreProfile(value, now=Date.now()) {
  return value && Number.isInteger(value.level) && value.level>=0 && value.level<=7
    && Number.isFinite(value.savedAt) && now>=value.savedAt && now-value.savedAt<7*86400000 ? value.level : 0;
}

export class ProfileSamples {
  record(ms, level, fps) {
    if(this.level!==level){this.level=level;this.samples=[];}
    if(!Number.isFinite(ms)||ms<=0||!Number.isFinite(fps)||fps<=0)return null;
    this.samples.push(ms);
    if(this.samples.length<30)return null;
    const sorted=this.samples.sort((a,b)=>a-b);this.samples=[];
    const p95=sorted[Math.ceil(sorted.length*.95)-1];
    return p95<1000/fps*.92 ? {level,p95Ms:p95,savedAt:Date.now()} : null;
  }
}
