"""Local held-out screenshot comparison through the actual binary service."""
import asyncio
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
from aiohttp import ClientSession
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'native'))
from protocol import HEADER, MAGIC

async def main():
    source=Image.open(sys.argv[1]).convert('RGB')
    w=640; h=int(w*source.height/source.width)//4*4
    low=source.resize((w,h),Image.Resampling.LANCZOS)
    rgba=np.asarray(low.convert('RGBA'))
    config=json.loads((ROOT/'native/connection.json').read_text())
    images={}
    async with ClientSession() as client:
        for index,model in enumerate(['adore','adore-lite','adore']):
            async with client.ws_connect('http://127.0.0.1:8766/stream?model='+model) as socket:
                await socket.send_str(config['token']); await socket.receive_json()
                await socket.send_bytes(HEADER.pack(MAGIC,index,w,h,w*2,h*2)+rgba.tobytes())
                timing=await socket.receive_json()
                if 'error' in timing: raise RuntimeError(timing['error'])
                response=await socket.receive()
                values=np.frombuffer(response.data,dtype=np.uint8,offset=HEADER.size).reshape(h*2,w*2,4).copy()
                if index==2:
                    assert np.array_equal(values,images['adore']), 'Switching models changed the restored Adore result'
                else: images[model]=values
    # View the held-out frame, not a training crop. Original PNG is not HR truth.
    comparison=Image.new('RGB',(1280,3*(h*2+30)),(24,24,24));draw=ImageDraw.Draw(comparison)
    rows=[('Bilinear - input 640px',low.resize((w*2,h*2),Image.Resampling.BILINEAR)),
          ('Adore FP16 - teacher',Image.fromarray(images['adore']).convert('RGB')),
          ('Adore Lite FP16 - experimental',Image.fromarray(images['adore-lite']).convert('RGB'))]
    for index,(label,image) in enumerate(rows):
        y=index*(h*2+30);draw.text((10,y+8),label,fill='white');comparison.paste(image,(0,y+30))
    folder=ROOT/'diagnostics/native-fixture';folder.mkdir(exist_ok=True)
    comparison.save(folder/'lite-holdout.png')
    report={'switchBackIdentical':True,'input':[w,h], 'output':[w*2,h*2],
      'quantizedTeacherMeanAbsoluteError':float(np.abs(images['adore'][:,:,:3].astype(float)-images['adore-lite'][:,:,:3]).mean()/255),
      'image':'diagnostics/native-fixture/lite-holdout.png',
      'scope':'One reserved screenshot; teacher agreement, not ground-truth quality.'}
    (ROOT/'diagnostics/lite-quality-results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': asyncio.run(main())
