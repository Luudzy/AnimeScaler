document.querySelector('#run').onclick=async()=>{
  const old=document.querySelector('#output'),output=old.cloneNode();old.replaceWith(output);
  const worker=new Worker('./repeat-worker-fixture.js'+(document.querySelector('#canvas2d').checked?'?canvas2d=1':''),{type:'module'});
  let readyResolve;const ready=new Promise(resolve=>{readyResolve=resolve;});
  let id=0;const pending=new Map();
  worker.onmessage=({data})=>{if(data.fixtureReady){readyResolve();return;}const p=pending.get(data.id);if(!p)return;pending.delete(data.id);clearTimeout(p.timer);data.error?p.reject(Error(data.error)):p.resolve(data.result);};
  const call=(type,data,transfer=[])=>new Promise((resolve,reject)=>{
    const key=++id,timer=setTimeout(()=>{pending.delete(key);reject(Error('Timeout'));},10000);
    pending.set(key,{resolve,reject,timer});worker.postMessage({id:key,type,...data},transfer);
  });
  try {
    await ready;
    const canvas=output.transferControlToOffscreen();
    await call('init',{canvas,model:'adore-lite',backend:'rocm',transport:'rgb',connection:{url:'http://127.0.0.1:8766',token:'a'.repeat(64)}},[canvas]);
    const source=new OffscreenCanvas(40,40),ctx=source.getContext('2d');ctx.fillStyle='rgb(100,100,100)';ctx.fillRect(0,0,40,40);
    const crop={x:0,y:0,width:40,height:40},size={inputWidth:40,inputHeight:40,outputWidth:80,outputHeight:80};
    const send=async(type,fluidVideo=false)=>{
      const bitmap=await createImageBitmap(source);const result=await call(type,{bitmap,crop,size,fluidVideo},[bitmap]);
      // OffscreenCanvas commits to its placeholder on a rendering update, after
      // the worker response. Read pixels only once that update has occurred.
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      return result;
    };
    const snap=new OffscreenCanvas(80,80),context=snap.getContext('2d',{willReadFrequently:true});
    const pixel=()=>{context.drawImage(output,0,0);return context.getImageData(40,40,1,1).data[0];};
    await send('frame');const initial=pixel();
    const repeat=await send('frame');const repeated=pixel();
    await send('frame');const changed=pixel();
    await send('present');await send('frame',true);await send('present');const hybrid=pixel();
    const hybridRepeat=await send('frame',true);await send('present');const hybridAgain=pixel();
    const checks={initialCorrect:initial===120,repeatPreservesPixels:repeated===120,changedFrameUpdates:changed===200,
      repeatHasOnlyHeader:repeat.responseBytes===24&&repeat.repeated===true,
      hybridPreservesPixels:hybrid===120&&hybridAgain===120,hybridRepeatHasOnlyHeader:hybridRepeat.responseBytes===24};
    document.querySelector('#result').textContent=JSON.stringify({...checks,compositor:repeat.compositor,passed:Object.values(checks).every(Boolean)},null,2);
  } catch(error){document.querySelector('#result').textContent=String(error);}
  finally{worker.terminate();}
};
