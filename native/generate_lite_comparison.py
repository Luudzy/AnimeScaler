"""Same input, same resolution; reconstruction plots and amplified error maps."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image,ImageDraw
import torch
from lite_model import AdoreLite
from lite_v3 import AdoreLiteV3
from lite_evaluation import ROOT,load_patches,tensor
from lite_degradations import degrade
import random

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    saved=torch.load(args.candidate/'best_composite-master-fp32.pt',map_location='cpu',weights_only=True);c=saved['config']
    v3=AdoreLiteV3(c['channels'],c['blocks'],c['residualScale']).eval();v3.load_state_dict(saved['model'])
    v2=AdoreLite().eval()
    with np.load(ROOT/'models/adore-lite-experimental.npz',allow_pickle=False) as w:v2.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
    folder=args.candidate/'comparisons';folder.mkdir(exist_ok=True)
    patches=load_patches(dataset=Path(c.get('datasetPath',ROOT/'native/training-data/v3')));rng=random.Random(194)
    # Fixed indices before evaluating V3; avoid selecting only its best examples.
    for index in np.linspace(0,len(patches)-1,min(8,len(patches)),dtype=int):
        high=Image.fromarray(patches[index]);low=degrade(high,rng,'heavy')
        with torch.inference_mode():
            x=tensor(low,'cpu');outputs=[m(x).clamp(0,1)[0].permute(1,2,0).numpy() for m in [v2,v3]]
        images=[low.resize(high.size,Image.Resampling.NEAREST),low.resize(high.size,Image.Resampling.BILINEAR),low.resize(high.size,Image.Resampling.BICUBIC),
                *[Image.fromarray(np.uint8(np.round(value*255))) for value in outputs],high]
        labels=['Input (nearest)','Bilinear','Bicubic','Lite V2','Lite V3','HR reference']
        canvas=Image.new('RGB',(6*384,416),'#161920');draw=ImageDraw.Draw(canvas)
        for i,(label,image) in enumerate(zip(labels,images)):
            draw.text((i*384+8,8),label,fill='white');canvas.paste(image.resize((384,384),Image.Resampling.NEAREST),(i*384,32))
        canvas.save(folder/f'{index:02d}-comparison-200pct.png')
        # All maps have fixed 8x gain, no per-image normalization.
        target=np.array(high)/255;baseline=np.array(images[2])/255
        maps=[np.abs(outputs[0]-target),np.abs(outputs[1]-target),np.abs(outputs[1]-baseline)]
        error=Image.new('RGB',(576,216),'#161920');d=ImageDraw.Draw(error)
        for i,(label,value) in enumerate(zip(['V2-HR error x8','V3-HR error x8','V3-bicubic diff x8'],maps)):
            d.text((i*192+4,4),label,fill='white');error.paste(Image.fromarray(np.uint8(np.clip(value*8*255,0,255))),(i*192,24))
        error.save(folder/f'{index:02d}-difference-x8.png')
        for scale in [4,8]:
            strip=Image.new('RGB',(6*48*scale,48*scale+24),'#161920');d=ImageDraw.Draw(strip)
            for i,(label,image) in enumerate(zip(labels,images)):
                d.text((i*48*scale+4,4),label,fill='white');strip.paste(image.crop((72,72,120,120)).resize((48*scale,48*scale),Image.Resampling.NEAREST),(i*48*scale,24))
            strip.save(folder/f'{index:02d}-center-{scale*100}pct.png')
    print(folder)
if __name__=='__main__':main()
