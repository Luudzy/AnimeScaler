import { AnimeScalerController } from '../src/content/controller.js';

window.runPlayerRegression=async()=>{
  const preview=document.querySelector('#preview'),main=document.querySelector('#main'),checks=[];
  const check=(name,passed,details)=>checks.push({name,passed:Boolean(passed),details});
  const ready=video=>video.readyState>=2?Promise.resolve():new Promise((resolve,reject)=>{
    video.addEventListener('loadeddata',resolve,{once:true});video.addEventListener('error',()=>reject(new Error('Media load failed')),{once:true});
  });
  const seek=(video,time)=>new Promise(resolve=>{video.addEventListener('seeked',resolve,{once:true});video.currentTime=time;});
  await Promise.all([ready(preview),ready(main)]);await main.play();
  const previewAt=preview.currentTime;
  const c=new AnimeScalerController();
  const first=c.getCaptureGeometry({frameId:0},'player-regression');
  check('initial geometry selects main',first?.rect.left===380,first?.rect);
  await c.playerCommand({type:'toggle'},'player-regression');
  check('pause controls main',main.paused);
  const stoppedAt=main.currentTime;
  await c.playerCommand({type:'seek',seconds:10},'player-regression');
  check('seek controls main',Math.abs(main.currentTime-stoppedAt-10)<.01,{mainTime:main.currentTime,previewTime:preview.currentTime});
  check('preview is untouched',Math.abs(preview.currentTime-previewAt)<.01);
  check('paused geometry remains main',c.getCaptureGeometry({frameId:0},'player-regression')?.rect.left===380);
  await c.playerCommand({type:'toggle'},'player-regression');
  check('resume controls main',!main.paused);check('preview remains paused',preview.paused);
  main.pause();preview.pause();
  main.style.visibility='hidden';
  check('hidden source is unavailable',c.getCaptureGeometry({frameId:0},'player-regression')===null);
  main.style.visibility='visible';
  await seek(main,2);
  c.state='running';c.video=c.displayTarget=main;c.settings={...c.settings,maxFps:12,adaptive:false};
  const rendered=[];
  c.engine={needsReconfigure:()=>false,async processFrame(video){rendered.push(video.currentTime);return {};},destroy(){}};
  // Exercise real HTMLVideoElement state; inference remains a CPU stub.
  await c.onVideoFrame(0,{mediaTime:2});await seek(main,4);
  await c.onVideoFrame(10,{mediaTime:4-1/24});await c.onVideoFrame(16,{mediaTime:4});
  check('paused seek refreshes without repeated inference',rendered.length===2&&Math.abs(rendered[0]-2)<.01&&Math.abs(rendered[1]-4)<.01,rendered);
  await c.stop();
  return {checks,failed:checks.filter(x=>!x.passed).length,scope:'Real local HTMLVideoElements and source controller; inference stub, no extension tabCapture or final upscaled pixels.'};
};
