import { TemporalPresenter } from '../src/engine/temporal-presenter.js';
import { createNativeGpuRenderer } from '../src/engine/native-gpu-renderer.js';
document.querySelector('#run').onclick=async()=>{
  const previous=document.querySelector('#output'),output=previous.cloneNode(false);previous.replaceWith(output);
  const useGpu=document.querySelector('#gpu').checked;
  const gpu=useGpu?await createNativeGpuRenderer(output):null;
  if(useGpu&&!gpu){document.querySelector('#result').textContent='FAIL: compositor GPU indisponível';return;}
  const snapshot=new OffscreenCanvas(160,90),snapshotContext=snapshot.getContext('2d',{willReadFrequently:true});
  const context=gpu?null:output.getContext('2d');
  const source=new OffscreenCanvas(160,90), sourceContext=source.getContext('2d');
  const presenter=new TemporalPresenter(output,context,gpu), crop={x:0,y:0,width:160,height:90},size={outputWidth:160,outputHeight:90};
  const pixel=(x,y)=>{snapshotContext.drawImage(output,0,0);return Array.from(snapshotContext.getImageData(x,y,1,1).data).slice(0,3);};
  const same=(actual,expected)=>actual.every((v,i)=>v===expected[i]);
  sourceContext.fillStyle='black';sourceContext.fillRect(0,0,160,90);
  presenter.present(source,crop,size);
  const reference=presenter.sample(source,crop);
  const ai=new ImageData(160,90); for(let i=0;i<ai.data.length;i+=4){ai.data[i]=255;ai.data[i+3]=255;}
  presenter.accept(ai,reference);presenter.draw();
  const stableUsesAI=same(pixel(140,70),[255,0,0]);
  sourceContext.fillStyle='white';sourceContext.fillRect(20,20,20,20);
  presenter.present(source,crop,size);
  const motionUsesCurrent=same(pixel(30,30),[255,255,255]);
  const stableStillUsesAI=same(pixel(140,70),[255,0,0]);
  const captureTime=reference.capturedAt;
  reference.capturedAt=performance.now()-301; presenter.draw();
  const oldResultExpires=same(pixel(140,70),[0,0,0]);
  reference.capturedAt=captureTime;
  presenter.accept(ai,reference);presenter.draw();
  sourceContext.fillStyle='white';sourceContext.fillRect(0,0,160,90);presenter.present(source,crop,size);
  const sceneCutUsesCurrent=same(pixel(140,70),[255,255,255]);
  const small=new OffscreenCanvas(80,45), smallContext=small.getContext('2d');
  const pixels=new ImageData(80,45);
  for(let y=0;y<45;y++)for(let x=0;x<80;x++) {
    const i=(y*80+x)*4;
    pixels.data.set([(x*17+y*31+x*y*3)%220,(x*7+y*19)%220,(x*11+y*13)%220,255],i);
  }
  smallContext.putImageData(pixels,0,0);
  sourceContext.imageSmoothingEnabled=false;sourceContext.drawImage(small,0,0,160,90);
  presenter.present(source,crop,size);
  const movingReference=presenter.sample(source,crop);
  const detailed=sourceContext.getImageData(0,0,160,90);
  for(let y=38;y<42;y++)for(let x=54;x<58;x++)detailed.data.set([255,0,0,255],(y*160+x)*4);
  presenter.accept(detailed,movingReference);
  sourceContext.fillStyle='black';sourceContext.fillRect(0,0,160,90);
  sourceContext.drawImage(small,4,2,160,90);
  const started=performance.now();presenter.present(source,crop,size);
  const motionPresentationMs=performance.now()-started;
  const aiDetailFollowsMotion=same(pixel(59,41),[255,0,0]);
  const oldPositionHasNoTrail=!same(pixel(55,39),[255,0,0]);
  const checks={stableUsesAI,motionUsesCurrent,stableStillUsesAI,oldResultExpires,sceneCutUsesCurrent,aiDetailFollowsMotion,oldPositionHasNoTrail};
  sourceContext.fillStyle='rgb(100,100,100)';sourceContext.fillRect(0,0,160,90);
  presenter.present(source,crop,size);
  const steadyReference=presenter.sample(source,crop);
  const steadyAI=new ImageData(160,90);for(let i=0;i<steadyAI.data.length;i+=4)steadyAI.data.set([130,130,130,255],i);
  const beforeAccept=pixel(140,70);
  presenter.accept(steadyAI,steadyReference);
  checks.acceptDoesNotPresent=same(pixel(140,70),beforeAccept);
  const pulse=[];
  for(const age of [20,53,86,119,150]) {
    steadyReference.capturedAt=performance.now()-age;presenter.draw();pulse.push(pixel(140,70)[0]);
  }
  checks.stableCadenceHasNoPulse=Math.max(...pulse)-Math.min(...pulse)<=1;
  steadyReference.capturedAt=performance.now();
  sourceContext.fillStyle='rgb(105,105,105)';sourceContext.fillRect(0,0,160,90);presenter.present(source,crop,size);
  const thresholdBefore=pixel(140,70)[0];
  sourceContext.fillStyle='rgb(107,107,107)';sourceContext.fillRect(0,0,160,90);presenter.present(source,crop,size);
  checks.thresholdDoesNotFlash=Math.abs(pixel(140,70)[0]-thresholdBefore)<8;
  sourceContext.fillStyle='white';sourceContext.fillRect(0,0,160,90);presenter.present(source,crop,size);
  checks.lateResultAfterCutRejected=presenter.accept(steadyAI,steadyReference)===false;
  presenter.draw();checks.lateResultCannotOverwriteCut=same(pixel(140,70),[255,255,255]);
  const geometryReference=presenter.sample(source,crop);
  presenter.present(source,{...crop,x:1,width:159},size);
  checks.changedGeometryRejectsHistory=presenter.accept(steadyAI,geometryReference)===false;
  presenter.resetHistory();presenter.current=null;
  sourceContext.fillStyle='rgb(100,100,100)';sourceContext.fillRect(0,0,160,90);
  presenter.present(source,crop,size);
  const oscillationReference=presenter.sample(source,crop);
  presenter.accept(steadyAI,oscillationReference);presenter.draw();
  let rejectedMatchStaysQuiet=true;
  for(let frame=0;frame<8;frame++) {
    const value=frame%2?100:120;
    sourceContext.fillStyle=`rgb(${value},${value},${value})`;sourceContext.fillRect(0,0,160,90);
    presenter.present(source,crop,size);
    rejectedMatchStaysQuiet&&=Math.abs(pixel(140,70)[0]-value)<=1;
  }
  checks.alternatingMatchesDoNotPulse=rejectedMatchStaysQuiet;
  // Exercise transferable frame timestamps, including identical-color seeks.
  presenter.resetHistory();presenter.current=null;
  sourceContext.fillStyle='rgb(100,100,100)';sourceContext.fillRect(0,0,160,90);
  const frameA=new VideoFrame(source,{timestamp:1000000});
  presenter.present(frameA,crop,size);
  const oldEpochReference=presenter.sample(frameA,crop);
  const repeatReference=presenter.sample(frameA,crop);
  checks.repeatHasIndependentReference=repeatReference!==oldEpochReference && repeatReference.pixels===oldEpochReference.pixels;
  presenter.resetHistory();presenter.present(frameA,crop,size);
  checks.resetCannotRelabelOldResult=presenter.accept(steadyAI,oldEpochReference)===false;
  const freshReference=presenter.sample(frameA,crop);
  presenter.accept(steadyAI,freshReference);presenter.draw();
  const frameB=new VideoFrame(source,{timestamp:500000});
  presenter.present(frameB,crop,size);
  checks.sameColorSeekRejectsOldResult=presenter.accept(steadyAI,freshReference)===false;
  checks.sameColorSeekShowsCurrent=same(pixel(140,70),[100,100,100]);
  frameA.close();frameB.close();
  const frameC=new VideoFrame(source,{timestamp:250000});
  const beforeSeekPresentation=presenter.sample(frameC,crop);
  presenter.present(frameC,crop,size);
  checks.captureBeforeSeekPresentationAccepted=presenter.accept(steadyAI,beforeSeekPresentation);
  // Completion uploads only. A fresh presentation imports the VideoFrame
  // texture again; an external texture is valid for one draw, not a second.
  presenter.present(frameC,crop,size);
  checks.captureBeforeSeekPresentationUsesAI=same(pixel(140,70),[130,130,130]);
  const changedCrop={...crop,x:1,width:159};
  const beforeCropPresentation=presenter.sample(frameC,changedCrop);
  checks.oldCropRejectedAtCapture=presenter.accept(steadyAI,beforeSeekPresentation)===false;
  presenter.present(frameC,changedCrop,size);
  checks.captureBeforeCropPresentationAccepted=presenter.accept(steadyAI,beforeCropPresentation);
  presenter.present(frameC,changedCrop,size);
  checks.captureBeforeCropPresentationUsesAI=same(pixel(140,70),[130,130,130]);
  frameC.close();
  presenter.resetHistory();presenter.current=null;
  sourceContext.fillStyle='rgb(100,100,100)';sourceContext.fillRect(0,0,160,90);
  const delayedFrame=new VideoFrame(source,{timestamp:1000000});
  presenter.present(delayedFrame,crop,size,10);
  sourceContext.fillStyle='rgb(200,200,200)';sourceContext.fillRect(0,0,160,90);
  const newerFrame=new VideoFrame(source,{timestamp:1040000});
  const newerCapture=presenter.sample(newerFrame,crop,12),newerEpoch=presenter.epoch;
  checks.delayedCaptureCannotPresent=presenter.present(delayedFrame,crop,size,11)===false;
  checks.delayedCaptureCannotSample=presenter.sample(delayedFrame,crop,11)===null;
  checks.delayedCaptureKeepsEpoch=presenter.epoch===newerEpoch;
  checks.newerCaptureSurvivesDelayedFrame=presenter.accept(steadyAI,newerCapture);
  presenter.present(newerFrame,crop,size,13);
  checks.newerCaptureStillUsesAI=same(pixel(140,70),[130,130,130]);
  delayedFrame.close();newerFrame.close();
  if(gpu){
    const direct=new ImageData(160,90);for(let i=0;i<direct.data.length;i+=4)direct.data.set([20,60,180,255],i);
    gpu.presentAI(direct);checks.normalAIColorPreserved=same(pixel(70,30),[20,60,180]);
    const smallAI=new ImageData(80,45);for(let i=0;i<smallAI.data.length;i+=4)smallAI.data.set([100,50,25,255],i);
    gpu.presentAI(smallAI);checks.resizeColorPreserved=same(pixel(20,20),[100,50,25]);
    const rgb=new Uint8Array(80*45*3);for(let i=0;i<rgb.length;i+=3)rgb.set([19,61,179],i);
    gpu.presentAI({width:80,height:45,rgb});checks.rgbUnpackPreservesColor=same(pixel(20,20),[19,61,179]);
  }
  document.querySelector('#result').textContent=JSON.stringify({...checks,compositor:gpu?'webgpu':'canvas2d',motionPresentationMs,
    coverageComponents:{motionCoverage:presenter.metrics.motionCoverage,
      confidenceCoverage:presenter.metrics.confidenceCoverage,temporalWeight:presenter.metrics.temporalWeight,
      aiCoverage:presenter.metrics.aiCoverage,historyAgeMs:presenter.metrics.historyAgeMs}},null,2);
  gpu?.destroy();
  if(Object.values(checks).some(v=>!v))throw new Error('Temporal pixel validation failed');
};
