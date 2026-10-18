"""Fixed validation, domain-separated metrics, including perceptual scores."""
import os,sys,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'native/.tools'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from lite_evaluation import load_patches,tensor,validate_codec_cache
from lite_losses import metrics
from lite_degradations import degrade

class FixedEvaluator:
    def __init__(self,dataset,codec_cache,per_domain=32):
        import lpips
        self.feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
        self.patches=load_patches(dataset=dataset)
        self.rows=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text().splitlines()]
        self.rows=[r for r in self.rows if r['split']=='validation']
        if len(self.rows)!=len(self.patches):raise ValueError('Validation image smaller than patch')
        validate_codec_cache(codec_cache,self.patches)
        self.selected=[]
        for domain in ['anime','illustration']:
            indices=[i for i,r in enumerate(self.rows) if r.get('domain','illustration')==domain]
            positions=np.linspace(0,len(indices)-1,min(per_domain,len(indices)),dtype=int) if per_domain else range(len(indices))
            self.selected.extend(indices[p] for p in positions)
        self.cache=codec_cache
    def run(self,models,on_outputs=None):
        result={};rng=random.Random(283)
        with torch.inference_mode():
            for case in ['clean','heavy','codec_intra']:
                groups={d:{n:[] for n in [*models,'bilinear','bicubic']} for d in ['anime','illustration','combined']}
                for index in self.selected:
                    high=Image.fromarray(self.patches[index])
                    low=Image.open(self.cache/f'{index:04d}.png').convert('RGB') if case=='codec_intra' else degrade(high,rng,case)
                    x=tensor(low);target=tensor(high)
                    outputs={'bilinear':F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False),
                             'bicubic':tensor(low.resize(high.size,Image.Resampling.BICUBIC))}
                    outputs.update({name:model(x.to(next(model.parameters()).dtype)).float() for name,model in models.items()})
                    if on_outputs is not None:on_outputs(case,index,target,outputs)
                    for name,pred in outputs.items():
                        pred=pred.clamp(0,1)
                        score=metrics(pred,target)
                        score.update(lpipsSqueeze=float(self.feature(pred*2-1,target*2-1).mean()),
                                     msSsim=float(ms_ssim(pred,target,data_range=1,size_average=True)))
                        groups[self.rows[index].get('domain','illustration')][name].append(score)
                        groups['combined'][name].append(score)
                result[case]={d:{n:{k:float(np.mean([r[k] for r in values])) for k in values[0]} for n,values in names.items()} for d,names in groups.items()}
        return {'scope':'Fixed validation only; domain-balanced screening subset when limited; not final-hidden',
                'validationIndices':self.selected,'cases':result,'realVideoValidation':'REAL VIDEO VALIDATION NOT EXECUTED'}
