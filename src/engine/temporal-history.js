// Conservative cut detection: large pixel disagreement plus a large color-
// histogram change. Pans with similar color distributions remain motion work.
export function isSceneCut(before, after) {
  if(!before || !after || before.length!==after.length || !before.length)return false;
  const a=new Uint32Array(24),b=new Uint32Array(24);
  let changed=0,total=0,error=0;
  for(let i=0;i<before.length;i+=16) {
    let maximum=0;
    for(let c=0;c<3;c++) {
      const delta=Math.abs(before[i+c]-after[i+c]);error+=delta;maximum=Math.max(maximum,delta);
      a[c*8+(before[i+c]>>5)]++;b[c*8+(after[i+c]>>5)]++;
    }
    if(maximum>40)changed++;total++;
  }
  let histogram=0;for(let i=0;i<a.length;i++)histogram+=Math.abs(a[i]-b[i]);
  return changed/total>.8 && error/(total*3)>40 && histogram/(total*6)>.6;
}

export class ConfidenceHistory {
  reset() {this.previous=null;this.at=null;this.recoverAfter=null;}
  clear(now) {this.previous?.fill(0);this.recoverAfter?.fill(now+150);this.at=now;}
  filter(reuse,now) {
    if(!reuse){this.clear(now);return null;}
    const source=reuse.mask;
    if(!this.previous||this.previous.length!==source.length) {
      this.previous=source.slice();this.at=now;
      this.recoverAfter=new Float64Array(source.length/4);
      for(let i=3;i<source.length;i+=4)if(source[i]<32)this.recoverAfter[i>>2]=now+150;
      return reuse;
    }
    const step=255*Math.min(50,Math.max(0,now-this.at))/120;
    this.at=now;
    const mask=source.slice();
    for(let i=3;i<mask.length;i+=4) {
      // Never keep invalid history. Only restoration is gradual, preventing
      // repeatedly rejected edges from flashing back to full AI strength.
      // An alternating match used to restore 28% on every other 30 FPS frame.
      // Require a quiet interval after rejection before restoring any detail.
      if(source[i]<32)this.recoverAfter[i>>2]=now+150;
      mask[i]=now<this.recoverAfter[i>>2]?Math.min(source[i],this.previous[i]):
        Math.min(source[i],Math.round(this.previous[i]+step));
    }
    this.previous=mask;
    return {...reuse,mask};
  }
}
