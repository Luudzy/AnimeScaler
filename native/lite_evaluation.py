"""Validation utilities: native-resolution crops, stronger interpolation baselines."""
import json,random,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lite_losses import metrics
from lite_degradations import degrade

ROOT=Path(__file__).resolve().parents[1]
def load_patches(split='validation',per_image=1,seed=193,size=192,dataset=None,domain=None):
    dataset=dataset or ROOT/'native/training-data/v3'
    rows=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text().splitlines()]
    rng=random.Random(seed);patches=[]
    for row in rows:
        if row['split']!=split:continue
        if domain and row.get('domain','illustration')!=domain:continue
        with Image.open(row['file']) as image:
            if min(image.size)<size:continue
            for _ in range(per_image):
                x=rng.randrange(image.width-size+1);y=rng.randrange(image.height-size+1)
                patches.append(np.array(image.crop((x,y,x+size,y+size)).convert('RGB')))
    return patches

def tensor(image,device='cuda'):
    return torch.from_numpy(np.array(image,copy=True).transpose(2,0,1)).to(device=device,dtype=torch.float32)[None]/255

def validate_codec_cache(folder,patches):
    records=json.loads((folder/'complete.json').read_text())['frames']
    if len(records)!=len(patches):raise ValueError('Codec cache and dataset counts differ; rebuild cache')
    for index,(record,patch) in enumerate(zip(records,patches)):
        if record['index']!=index or record['targetSha256']!=hashlib.sha256(patch.tobytes()).hexdigest():
            raise ValueError('Codec cache target mismatch; rebuild cache')

def evaluate(models,patches,device='cuda',cases=('clean','heavy'),codec_cache=None):
    report={};rng=random.Random(283)
    if 'codec' in cases:validate_codec_cache(codec_cache,patches)
    with torch.inference_mode():
        for case in cases:
            scores={key:[] for key in [*models,'bilinear','bicubic','lanczos']}
            for index,patch in enumerate(patches):
                high=Image.fromarray(patch)
                low=Image.open(codec_cache/f'{index:04d}.png').convert('RGB') if case=='codec' else degrade(high,rng,case)
                x=tensor(low,device);target=tensor(high,device)
                values={'bilinear':F.interpolate(x,size=target.shape[-2:],mode='bilinear',align_corners=False),
                        'bicubic':tensor(low.resize(high.size,Image.Resampling.BICUBIC),device),
                        'lanczos':tensor(low.resize(high.size,Image.Resampling.LANCZOS),device)}
                for name,model in models.items():
                    dtype=next(model.parameters()).dtype
                    values[name]=model(x.to(dtype)).float()
                for name,value in values.items():scores[name].append(metrics(value,target))
            report[case]={name:{key:float(np.mean([v[key] for v in values])) for key in values[0]} for name,values in scores.items()}
    return report

def validation_gate(report):
    # No test or final-hidden metrics are accepted here.
    checks={case:{'mae':s['candidate']['mae']<=s['previous']['mae'],
                  'edge':s['candidate']['edgeMae']<s['previous']['edgeMae'],
                  'ssim':s['candidate']['ssim']>=s['previous']['ssim'],
                  'beatsBicubicMae':s['candidate']['mae']<s['bicubic']['mae']}
            for case,s in report.items()}
    return {'byCase':checks,'passed':all(all(v.values()) for v in checks.values())}
