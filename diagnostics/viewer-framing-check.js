import {describeCaptureTarget} from '../src/content/capture-target.js';
import {renderStillComparison} from '../viewer/still-comparison.js';
const $ = id => document.getElementById(id);
$('run').onclick = async () => {
  const source = document.createElement('canvas');source.width=320;source.height=180;
  const ctx = source.getContext('2d'), video=$('video');
  ctx.fillStyle='rgb(180,40,70)';ctx.fillRect(0,0,320,180);
  const stream=source.captureStream(30);video.srcObject=stream;
  let tick=0;
  const pump=setInterval(()=>{ctx.fillStyle=`rgb(${tick++%255},0,0)`;ctx.fillRect(0,0,1,1);stream.getVideoTracks()[0].requestFrame?.();},33);
  try {
    await video.play();
    await new Promise(resolve=>video.requestVideoFrameCallback(resolve));
    const geometry=describeCaptureTarget(document,{frameId:0,viewport:{width:innerWidth,height:innerHeight}});
    const result={contentAspectCorrect:Math.abs(geometry.rect.width/geometry.rect.height-16/9)<0.001,
      barsExcluded:Math.abs(geometry.rect.height-337.5)<0.1};
    const output=$('output');output.width=640;output.height=360;
    // A deterministic stand-in isolates frame identity from model quality.
    const engine={currentSize:{outputWidth:640,outputHeight:360},async processFrame(frame,crop,{still}){
      result.stillRequested=still;
      ctx.fillStyle='rgb(20,200,40)';ctx.fillRect(0,0,320,180);
      await new Promise(resolve=>setTimeout(resolve,150));
      const out=output.getContext('2d');out.imageSmoothingQuality='high';out.drawImage(frame,0,0,640,360);
      return this.currentSize;
    }};
    await renderStillComparison({capture:video,original:$('original'),engine,crop:{x:0,y:0,width:320,height:180}});
    const a=$('original').getContext('2d').getImageData(0,0,640,360).data;
    const b=output.getContext('2d').getImageData(0,0,640,360).data;
    result.identicalFrozenPixels=a.every((value,index)=>value===b[index]);
    result.capturePaused=video.paused;
    await video.play();
    await new Promise(resolve=>video.requestVideoFrameCallback(resolve));
    result.resumed=!video.paused;
    $('result').textContent=JSON.stringify(result,null,2);
  } catch(error) {$('result').textContent=error.stack;}
  finally {clearInterval(pump);stream.getTracks().forEach(track=>track.stop());}
};
