"""Matched frozen frames: bilinear, official FSR3 reset and video-motion adaptation."""
import json
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'native'))
from fsr3_worker import Fsr3Renderer
from protocol import HEADER,MAGIC


def metrics(output,reference,previous,old_reference):
    delta=output-reference
    mse=float(np.mean(delta*delta))
    return {'mae':float(np.mean(np.abs(delta))), 'psnr':float(-10*np.log10(max(mse,1e-12))),
            'temporalMae':None if previous is None else float(np.mean(np.abs((output-previous)-(reference-old_reference))))}


def main():
    renderer=Fsr3Renderer(ROOT)
    report={'scope':'144 frozen real-video LR/HR pairs, 360p to 720p. Pillow bilinear; RGB MAE and reference-compensated temporal MAE, no flow warping. Not player FPS.',
            'provider':renderer.info,'fps':'24000/1001','clips':[],
            'workerSha256':hashlib.sha256(renderer.worker.exe.read_bytes()).hexdigest()}
    try:
        for seconds in (60,600,1200):
            folder=ROOT/'diagnostics/video-sequences/rezero-real-01'/f'clip-{seconds}s-360-to-720'
            files=sorted((folder/'lr').glob('*.png'))[:48]
            if len(files)!=48:raise ValueError('Incomplete frozen manifest')
            clip={'clip':folder.name,'frames':48,'metrics':{},'completedNativeMs':{}}
            for mode in ('bilinear','fsr3-reset','fsr3-video-motion'):
                rows=[];times=[];previous=None;old_reference=None
                renderer.reset_history();renderer.frame_context(False,1001/24)
                for index,path in enumerate(files):
                    with Image.open(path) as source:lr=source.convert('RGBA')
                    with Image.open(folder/'hr'/path.name) as source:hr=source.convert('RGB')
                    w,h=lr.size;ow,oh=hr.size
                    if mode=='bilinear':output=lr.resize((ow,oh),Image.Resampling.BILINEAR).convert('RGB')
                    else:
                        if mode=='fsr3-reset':renderer.reset_history()
                        body,duration=renderer.render(HEADER.pack(MAGIC,index,w,h,ow,oh)+lr.tobytes())
                        output=Image.frombytes('RGBA',(ow,oh),body[HEADER.size:]).convert('RGB');times.append(duration)
                        if index in (0,16,32):output.save(ROOT/'diagnostics'/f'fsr3-{seconds}s-{mode}-{index}.png')
                    a=np.asarray(output,dtype=np.float32)/255;r=np.asarray(hr,dtype=np.float32)/255
                    rows.append(metrics(a,r,previous,old_reference));previous=a;old_reference=r
                clip['metrics'][mode]={key:float(np.mean([row[key] for row in rows if row[key] is not None])) for key in rows[0]}
                if times:clip['completedNativeMs'][mode]={'scope':'Service worker pipe, upload, GPU completion, readback; not pure inference or player FPS','warmupMs':times[0],'median':float(np.median(times[1:])),'p95':float(np.percentile(times[1:],95))}
            report['clips'].append(clip)
            print(f'Completed {folder.name}',flush=True)
        target=ROOT/'diagnostics/fsr3-native-real-video-144.json'
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report['clips'],indent=2))
    finally:renderer.close()


if __name__=='__main__':main()
