const mean=values=>values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
export function summarize(values) {
  const sorted=values.filter(Number.isFinite).sort((a,b)=>a-b),count=sorted.length;
  return {count,mean:mean(sorted),p50:count?sorted[Math.ceil(count*.5)-1]:null,
    p95:count?sorted[Math.ceil(count*.95)-1]:null,max:count?sorted[count-1]:null};
}

const unit=value=>Number.isFinite(value)&&value>=0&&value<=1;
function attribute(rows) {
  const totals={motion:0,confidence:0,age:0,unusableHistory:0,ai:0};
  let attributedSamples=0,missingComponents=0,inconsistentComponents=0;
  for(const row of rows) {
    const {motionCoverage:m,confidenceCoverage:c,temporalWeight:w,aiCoverage:ai}=row;
    if(![m,c,w,ai].every(unit)){missingComponents++;continue;}
    if(c>m+1e-8 || Math.abs(ai-c*w)>1e-8){inconsistentComponents++;continue;}
    attributedSamples++;
    // When strength is zero the presenter skips matching and reports zero
    // masks. Those zeros cannot be interpreted as measured motion rejection.
    if(w===0)totals.unusableHistory++;
    else {
      totals.motion+=1-m;totals.confidence+=m-c;totals.age+=c*(1-w);
      totals.ai+=ai;
    }
  }
  return {attributedSamples,missingComponents,inconsistentComponents,
    lossMean:attributedSamples?Object.fromEntries(Object.entries(totals).map(([key,value])=>[key,value/attributedSamples])):null};
}

export function analyzePresentation(rows) {
  const components=Object.fromEntries(['motionCoverage','confidenceCoverage','temporalWeight'].map(key=>
    [key,summarize(rows.map(row=>row[key]).filter(unit))]));
  const edges=[0,40,80,150,300,Infinity];
  const ageBuckets=edges.slice(0,-1).map((minimum,index)=>{
    const maximum=edges[index+1];
    const group=rows.filter(row=>Number.isFinite(row.historyAgeMs)&&row.historyAgeMs>=minimum&&row.historyAgeMs<maximum);
    return {minimumMs:minimum,maximumMs:Number.isFinite(maximum)?maximum:null,samples:group.length,
      aiCoverage:summarize(group.map(row=>row.aiCoverage).filter(unit)),...attribute(group)};
  });
  return {samples:rows.length,components,...attribute(rows),ageBuckets,
    notes:['Loss means are per-frame attribution over complete, consistent components; not products of component means.',
      'Zero temporal weight marks unusable history, not independently measured motion loss. Paused unchanged history can exceed 300ms with weight one.',
      'Motion matching and confidence losses describe the compositor masks; they do not measure model quality or establish causality.']};
}
