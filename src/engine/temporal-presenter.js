import { motionReuse } from './temporal-motion.js';
import { TEMPORAL_MAX_SOURCE_AGE_MS, temporalStrength } from './temporal-budget.js';
import { ConfidenceHistory, isSceneCut } from './temporal-history.js';

export class TemporalPresenter {
  constructor(output, context, gpu=null) {
    this.output=output; this.context=context;this.gpu=gpu;
    this.base=new OffscreenCanvas(1,1); this.baseContext=this.base.getContext('2d',{alpha:false});
    this.ai=new OffscreenCanvas(1,1); this.aiContext=this.ai.getContext('2d');
    this.layer=new OffscreenCanvas(1,1); this.layerContext=this.layer.getContext('2d');
    this.thumb=new OffscreenCanvas(80,45); this.thumbContext=this.thumb.getContext('2d',{willReadFrequently:true});
    this.mask=new OffscreenCanvas(80,45); this.maskContext=this.mask.getContext('2d');
    this.current=null; this.reference=null;
    this.epoch=0;this.confidence=new ConfidenceHistory();
    this.metrics={sceneCuts:0,rejectedResults:0,acceptedResults:0,presentedFrames:0,lowConfidenceFrames:0,aiCoverage:0,historyAgeMs:0};
  }
  resetHistory() {
    this.epoch++;this.reference=null;this.motionReference=null;this.motionCurrent=null;this.reuse=null;
    this.observed=null;
    this.sampleKey=null;this.lastSample=null;
    this.confidence.reset();
  }
  isObsolete(sourceSequence) {
    return Number.isFinite(sourceSequence) && sourceSequence<this.latestSequence;
  }
  sample(frame,crop,sourceSequence) {
    // Order belongs to capture initiation, not media time or async delivery.
    // A newer backward seek is valid; a delayed old ImageBitmap is obsolete.
    if(this.isObsolete(sourceSequence))return null;
    if(Number.isFinite(sourceSequence))this.latestSequence=sourceSequence;
    const key=typeof frame.timestamp==='number'?`${frame.timestamp}:${crop.x}:${crop.y}:${crop.width}:${crop.height}`:null;
    let pixels;
    if(key&&key===this.sampleKey)pixels=this.lastSample.pixels;
    else {
      this.thumbContext.drawImage(frame,crop.x,crop.y,crop.width,crop.height,0,0,80,45);
      pixels=this.thumbContext.getImageData(0,0,80,45).data;
    }
    const mediaTime=Number.isFinite(frame.timestamp)?frame.timestamp:null;
    const geometry=`${crop.x}:${crop.y}:${crop.width}:${crop.height}`;
    const previous=this.observed;
    const discontinuity=previous && mediaTime!==null && previous.mediaTime!==null &&
      (mediaTime<previous.mediaTime || mediaTime-previous.mediaTime>250000);
    // Inference and presentation both observe the source. A capture may reach
    // a cut/seek before the presentation clock; tag it with the new epoch now.
    if(previous && (geometry!==previous.geometry || discontinuity))this.resetHistory();
    else if(previous && pixels!==previous.pixels && isSceneCut(previous.pixels,pixels)) {
      this.resetHistory();this.metrics.sceneCuts++;
    }
    this.sampleKey=key;
    // Cache pixels only. Each inference capture owns its age and epoch; an
    // in-flight result must never be relabelled by a later presentation/reset.
    return this.observed=this.lastSample={pixels,capturedAt:performance.now(),
      mediaTime,epoch:this.epoch,geometry};
  }
  resize(width,height) {
    if (this.base.width===width && this.base.height===height) return;
    for (const canvas of [this.output,this.base,this.ai,this.layer]) {canvas.width=width;canvas.height=height;}
    this.gpu?.resize(width,height);
    this.resetHistory();this.sampleKey=null;this.lastSample=null;
    this.uploadedImage=null;
    this.current=null; this.reference=null;
  }
  present(frame,crop,size,sourceSequence) {
    if(this.isObsolete(sourceSequence))return false;
    this.resize(size.outputWidth,size.outputHeight);
    const next=this.sample(frame,crop,sourceSequence);
    this.current=next;
    if(this.gpu)this.gpu.setFrame(frame,crop);
    else this.baseContext.drawImage(frame,crop.x,crop.y,crop.width,crop.height,0,0,this.base.width,this.base.height);
    this.draw();
    return true;
  }
  recordCoverage(reuse,strength,age,rawReuse=reuse) {
    const coverage=source=>{
      if(!source?.mask.length)return 0;
      let sum=0;for(let i=3;i<source.mask.length;i+=4)sum+=source.mask[i];
      return sum/(source.mask.length/4*255);
    };
    this.metrics.motionCoverage=coverage(rawReuse);
    this.metrics.confidenceCoverage=coverage(reuse);
    this.metrics.temporalWeight=strength;
    this.metrics.aiCoverage=this.metrics.confidenceCoverage*strength;
    this.metrics.historyAgeMs=Math.max(0,age);
    this.metrics.presentedFrames++;
    if(this.metrics.aiCoverage<.1)this.metrics.lowConfidenceFrames++;
  }
  unchangedSource(reference) {
    if (!reference || !this.current || reference.epoch!==this.epoch ||
        reference.geometry!==this.current.geometry) return false;
    const a=reference.pixels,b=this.current.pixels;
    if (!a || !b || a.length!==b.length) return false;
    if (a===b) return true;
    for(let i=0;i<a.length;i++)if(a[i]!==b[i])return false;
    return true;
  }
  accept(image,reference,repeated=false) {
    let reason;
    if (image.width!==this.ai.width || image.height!==this.ai.height) reason='size';
    else if(!reference || (reference.epoch!==undefined && reference.epoch!==this.epoch))reason='scene';
    else if((this.observed??this.current) && reference.geometry!==(this.observed??this.current).geometry)reason='geometry';
    else if(performance.now()-reference.capturedAt>=TEMPORAL_MAX_SOURCE_AGE_MS && !this.unchangedSource(reference))reason='expired';
    else if(this.reference && reference.capturedAt<this.reference.capturedAt)reason='out-of-order';
    if(reason) {this.metrics.rejectedResults++;this.metrics.lastRejected=reason;return false;}
    if(!repeated || this.uploadedImage!==image) {
      if(this.gpu)this.gpu.uploadAI(image);else if(image.data)this.aiContext.putImageData(image,0,0);else this.aiContext.drawImage(image,0,0);
      this.uploadedImage=image;
    }
    if(reference) {
      const latency=performance.now()-reference.capturedAt;
      this.latencyMs=this.latencyMs===undefined?latency:this.latencyMs*.8+latency*.2;
    }
    // Only the presentation clock draws. Rendering again on every inference
    // completion caused a second, irregular brightness/detail update per frame.
    this.reference=reference;
    this.metrics.acceptedResults++;return true;
  }
  draw() {
    if (!this.current) return;
    const width=this.base.width,height=this.base.height;
    if(!this.gpu)this.context.drawImage(this.base,0,0);
    // Age belongs to the source frame, not to when slow inference finished.
    const age=this.reference?performance.now()-this.reference.capturedAt:0;
    // A stopped source has no temporal displacement. Keep its validated AI
    // visible, retaining the real capture age in diagnostics. Motion and cuts
    // still use the original source-age budget, never a fabricated capture time.
    const unchanged=this.unchangedSource(this.reference);
    const strength=this.reference?temporalStrength(unchanged?0:age,this.latencyMs):0;
    this.metrics.unchangedSource=unchanged;
    if (!strength) {this.recordCoverage(null,0,age);this.confidence.clear(performance.now());this.gpu?.draw();return;}
    // Repeated video frames do not need another block search.
    if(this.motionReference!==this.reference.pixels||this.motionCurrent!==this.current.pixels) {
      this.reuse=motionReuse(this.reference.pixels,this.current.pixels,80,45);
      this.motionReference=this.reference.pixels;this.motionCurrent=this.current.pixels;
    }
    const reuse=this.confidence.filter(this.reuse,performance.now());
    this.recordCoverage(reuse,strength,age,this.reuse);
    if(this.gpu){this.gpu.draw(reuse,false,strength);return;}
    if (!reuse?.tiles.length) return;
    this.maskContext.putImageData(new ImageData(reuse.mask,80,45),0,0);
    this.layerContext.clearRect(0,0,width,height);
    const sx=width/80,sy=height/45;
    this.layerContext.imageSmoothingEnabled=true;
    for(const tile of reuse.tiles) {
      this.layerContext.drawImage(this.ai,(tile.x+tile.dx)*sx,(tile.y+tile.dy)*sy,tile.width*sx,tile.height*sy,
        tile.x*sx,tile.y*sy,tile.width*sx,tile.height*sy);
    }
    this.layerContext.globalCompositeOperation='destination-in'; this.layerContext.imageSmoothingEnabled=true;
    this.layerContext.drawImage(this.mask,0,0,width,height);
    this.layerContext.globalCompositeOperation='source-over';
    this.context.globalAlpha=strength;this.context.drawImage(this.layer,0,0);this.context.globalAlpha=1;
  }
}
