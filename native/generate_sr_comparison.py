"""Registry-based full-image, blinded crop and fixed-gain difference exports."""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw


def write_comparison(images, folder, blind=False, seed=731):
    if 'reference' not in images or len({image.size for image in images.values()}) != 1:
        raise ValueError('Comparisons require one common resolution and a reference')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    names=list(images)
    if blind:random.Random(seed).shuffle(names)
    key={chr(65+i) if blind else name:name for i,name in enumerate(names)}
    target=np.asarray(images['reference'],dtype=np.float32)
    width,height=images['reference'].size
    for label,name in key.items():
        image=images[name];image.save(folder/f'{label}-full.png')
        value=np.asarray(image,dtype=np.float32)
        Image.fromarray(np.uint8(np.clip(np.abs(value-target)*8,0,255))).save(folder/f'{label}-difference-gain8.png')
        gray=value.mean(2);edge=np.zeros_like(gray)
        edge[:-1,:-1]=(np.abs(np.diff(gray,axis=0)[:,:-1])+np.abs(np.diff(gray,axis=1)[:-1,:]))/2
        Image.fromarray(np.uint8(np.clip(edge*4,0,255))).save(folder/f'{label}-edges-gain4.png')
    # Fixed 96px center crop magnified with nearest; no model gets extra sharpening.
    side=min(96,width,height);left=(width-side)//2;top=(height-side)//2
    for zoom in (2,4):
        cell=side*zoom;columns=min(3,len(key));rows=math.ceil(len(key)/columns)
        sheet=Image.new('RGB',(cell*columns,(cell+24)*rows),'#161920');draw=ImageDraw.Draw(sheet)
        for index,(label,name) in enumerate(key.items()):
            x,y=(index%columns)*cell,(index//columns)*(cell+24)
            draw.text((x+4,y+4),label,fill='white')
            crop=images[name].crop((left,top,left+side,top+side)).resize((cell,cell),Image.Resampling.NEAREST)
            sheet.paste(crop,(x,y+24))
        sheet.save(folder/f'{zoom}x-crop.png')
    return key


def main():
    from server import ROOT
    import torch
    from torch.nn import functional as F
    from model_registry import build_registry
    from lite_evaluation import tensor
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',action='append')
    parser.add_argument('--dataset',type=Path,default=ROOT/'native/training-data/v3-hfa2k')
    parser.add_argument('--output',type=Path,default=ROOT/'diagnostics/zenkai-baselines/full-comparisons')
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--blind',action='store_true')
    parser.add_argument('--zenkai-export',action='append',help='Explicit experimental entry: zenkai-ID=EXPORT_DIRECTORY')
    args=parser.parse_args()
    if args.count<1:parser.error('count must be positive')
    torch.set_num_threads(2)
    registry=build_registry(ROOT)
    from zenkai_artifacts import add_zenkai_exports
    add_zenkai_exports(registry,args.zenkai_export)
    names=args.model or ['adore-lite-v3','realesr-anime','hfa2k-cugan']
    models={name:registry.load(name,'torch',device='cuda',precision='fp16') for name in names}
    rows=[json.loads(line) for line in (args.dataset/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[row for row in rows if row['split']=='validation' and row.get('domain')=='anime']
    if not rows:raise ValueError('No anime validation images')
    args.output.mkdir(parents=True,exist_ok=True)
    records=[];keys={}
    for index in np.linspace(0,len(rows)-1,min(args.count,len(rows)),dtype=int):
        row=rows[index]
        with Image.open(row['file']) as image:high=image.convert('RGB')
        high=high.crop((0,0,high.width//2*2,high.height//2*2))
        low=high.resize((high.width//2,high.height//2),Image.Resampling.LANCZOS)
        x=tensor(low)
        def to_image(value):
            pixels=value[0].float().clamp(0,1).cpu().permute(1,2,0).numpy()
            return Image.fromarray(np.uint8(np.rint(pixels*255)))
        images={'input-nearest':low.resize(high.size,Image.Resampling.NEAREST),
                'bilinear':to_image(F.interpolate(x,size=(high.height,high.width),mode='bilinear',align_corners=False)),
                'bicubic':low.resize(high.size,Image.Resampling.BICUBIC)}
        with torch.inference_mode():
            for name,model in models.items():
                output=model.infer(x.half())
                if output.shape[-2:]!=(high.height,high.width):
                    output=F.interpolate(output,size=(high.height,high.width),mode='bilinear',align_corners=False)
                if not torch.isfinite(output).all():raise ValueError('Non-finite comparison output')
                images[name]=to_image(output)
        images['reference']=high
        key=write_comparison(images,args.output/f'{index:04d}',args.blind,731+int(index))
        keys[str(index)]=key
        records.append({'validationIndex':int(index),'source':row['file'],'sourceSha256':hashlib.sha256(Path(row['file']).read_bytes()).hexdigest(),
                        'inputSize':list(low.size),'targetSize':list(high.size)})
        print(f'Comparison {index} complete',flush=True)
    # Key is deliberately outside each blind image directory.
    (args.output/'blind-key.json').write_text(json.dumps(keys,indent=2),encoding='utf-8')
    (args.output/'scope.json').write_text(json.dumps({'scope':'Full anime validation images, clean synthetic x2; no hidden split or real video.',
        'backend':'ROCm','precision':'fp16','models':{name:model.artifact for name,model in models.items()},
        'externalTiling':'512px tiles above 1280x720 input; Real-CUGAN global SE makes tiling approximate.',
        'selection':'Evenly spaced validation entries, selected before inspection','records':records},indent=2),encoding='utf-8')


if __name__=='__main__':main()
