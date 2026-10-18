import { ExclusiveSource } from '../src/content/exclusive-source.js';
const exclusive=new ExclusiveSource();
document.querySelector('#test').onclick=async()=>{
  const result=document.querySelector('#result'),frame=document.querySelector('#player');
  const target={frameId:1,url:frame.src};
  const originals=[document.querySelector('#wrapper'),frame,frame.contentDocument.querySelector('#clip'),frame.contentDocument.querySelector('video')].map(el=>({el,style:el.getAttribute('style')}));
  try{
    const parent=exclusive.enter('test',target);
    const child=frame.contentWindow.exclusive.enter('test',{frameId:0});
    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    const rect=frame.getBoundingClientRect();const video=frame.contentDocument.querySelector('video');
    const inner=video.getBoundingClientRect();
    const full=parent.ok&&child.ok&&Math.abs(rect.width-innerWidth)<1&&Math.abs(rect.height-innerHeight)<1&&Math.abs(inner.width-rect.width)<1&&Math.abs(inner.height-rect.height)<1&&frame.contentWindow.getComputedStyle(video).objectFit==='contain';
    frame.contentWindow.exclusive.exit('test');exclusive.exit('test');
    const restored=originals.every(({el,style})=>(el.getAttribute('style')||'')===(style||''));
    result.textContent=JSON.stringify({fullVideoFits:full,originalStylesRestored:restored,outer:[rect.width,rect.height],inner:[inner.width,inner.height]},null,2);
    if(!full||!restored)throw new Error('Falha na expansão ou restauração');
  }catch(error){result.textContent+='\n'+error.message;frame.contentWindow.exclusive?.exit();exclusive.exit();}
};
