"""Identified and blinded same-resolution comparisons of fixed validation images."""
import sys,json,argparse,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image,ImageDraw
import torch
from lite_v31 import AdoreLiteV31
from lite_v3 import AdoreLiteV3
from lite_model import AdoreLite

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
    saved=torch.load(args.candidate/'best-lpipsSqueeze.pt',map_location='cpu',weights_only=True);config=saved['config']
    model=AdoreLiteV31(config['variant'],config['residual_scale']).eval();model.load_state_dict(saved['model'])
    v2=AdoreLite().eval();v3=AdoreLiteV3().eval()
    for m,name in [(v2,'adore-lite-experimental.npz'),(v3,'adore-lite-v3-experimental.npz')]:
        with np.load(root/'models'/name) as w:m.load_state_dict({k:torch.from_numpy(w[k].copy()).float() for k in w.files})
    rows=[json.loads(s) for s in (Path(config['dataset'])/'manifest.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']=='validation' and r.get('domain')=='anime']
    out=args.candidate/'visual';out.mkdir(exist_ok=True);key={}
    for i in np.linspace(0,len(rows)-1,8,dtype=int):
        with Image.open(rows[i]['file']) as image:high=image.convert('RGB')
        high=high.crop((0,0,high.width//2*2,high.height//2*2))
        low=high.resize((high.width//2,high.height//2),Image.Resampling.LANCZOS)
        x=torch.from_numpy(np.array(low).transpose(2,0,1).copy()).float()[None]/255
        images={'LR nearest':low.resize(high.size,Image.Resampling.NEAREST),
                'Bilinear':low.resize(high.size,Image.Resampling.BILINEAR),'Bicubic':low.resize(high.size,Image.Resampling.BICUBIC)}
        with torch.inference_mode():
            for label,m in [('V2',v2),('V3',v3),('V3.1',model)]:
                value=m(x).clamp(0,1)[0].permute(1,2,0).numpy()
                images[label]=Image.fromarray(np.uint8(np.round(value*255)))
        images['HR']=high
        for name,image in images.items():image.save(out/f'{i:03d}-{name.replace(" ","-")}-full.png')
        names=list(images);blind=names.copy();random.Random(731+int(i)).shuffle(blind)
        key[str(i)]={chr(65+j):name for j,name in enumerate(blind)}
        for zoom in [2,4]:
            side=high.width//zoom;box=((high.width-side)//2,(high.height-side)//2,(high.width+side)//2,(high.height+side)//2)
            for identified,order in [(True,names),(False,blind)]:
                panel=Image.new('RGB',(384*len(order),416),'#171a20');draw=ImageDraw.Draw(panel)
                for j,name in enumerate(order):
                    draw.text((384*j+8,8),name if identified else chr(65+j),fill='white')
                    panel.paste(images[name].crop(box).resize((384,384),Image.Resampling.NEAREST),(384*j,32))
                panel.save(out/f'{i:03d}-{zoom}x-{"identified" if identified else "blind"}.png')
        target=np.asarray(high,dtype=np.float32)
        for name in ['V2','V3','V3.1']:
            value=np.asarray(images[name],dtype=np.float32)
            Image.fromarray(np.uint8(np.clip(np.abs(value-target)*8,0,255))).save(out/f'{i:03d}-{name}-error-gain8.png')
            gray=value.mean(2);edge=np.zeros_like(gray);edge[:-1,:-1]=(np.abs(np.diff(gray,axis=0)[:,:-1])+np.abs(np.diff(gray,axis=1)[:-1,:]))/2
            Image.fromarray(np.uint8(np.clip(edge*4,0,255))).save(out/f'{i:03d}-{name}-edges-gain4.png')
    (out/'blind-key.json').write_text(json.dumps(key,indent=2))
    (out/'scope.json').write_text(json.dumps({'checkpoint':saved['step'],'selection':'best LPIPS validation checkpoint, not release selection','cases':'8 evenly spaced anime validation images; full native images, synthetic clean 2x input','realVideo':'NOT EXECUTED'}))
if __name__=='__main__':main()
