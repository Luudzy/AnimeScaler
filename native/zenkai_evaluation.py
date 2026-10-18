"""Deterministic model screening using the saved Zenkai image manifest."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
try:
    from pytorch_msssim import ms_ssim
except ImportError:  # Optional outside the ROCm training environment.
    ms_ssim = None
from lite_losses import metrics
from zenkai_losses import high_frequency, block_dct
from zenkai_validation_manifest import render_validation_sample
from zenkai_validation_manifest import APPROVED_ROOTS


def score_prediction(prediction,target):
    prediction=prediction.float().clamp(0,1);target=target.float()
    base=metrics(prediction,target)
    base['hfError']=float((high_frequency(prediction)-high_frequency(target)).abs().mean())
    base['dctError']=float((block_dct(prediction)-block_dct(target)).abs().mean())
    return base['mae'],base


class ZenkaiFixedEvaluator:
    def __init__(self,manifest,per_case=32,allowed_roots=APPROVED_ROOTS):
        manifest=Path(manifest).resolve();raw=manifest.read_bytes();self.manifest=json.loads(raw)
        self.manifest_path=manifest;self.sha256=hashlib.sha256(raw).hexdigest()
        if self.manifest.get('schemaVersion')!=1 or self.manifest.get('domain')!='anime':
            raise ValueError('Unsupported Zenkai validation manifest')
        source=Path(self.manifest['sourceManifest'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=self.manifest['sourceManifestSha256']:
            raise ValueError('Source manifest changed since fixed validation was created')
        if self.manifest.get('realVideoValidation')!='REAL VIDEO VALIDATION NOT EXECUTED':
            raise ValueError('Unexpected validation scope')
        self.per_case=per_case
        self.allowed_roots=allowed_roots
        self.codec_folder=None;self.codec_sha256=None
        if type(per_case) is not int or per_case<1:raise ValueError('per_case must be positive')
        samples=self.manifest.get('samples')
        if not samples or any(row.get('split')!='validation' for row in samples):
            raise ValueError('Empty or non-validation fixed manifest')
        self.samples={case:[] for case in self.manifest['cases']}
        for case in self.samples:
            rows=sorted((row for row in samples if row['case']==case),key=lambda row:row['sourceSha256'])
            if not rows:raise ValueError(f'Missing fixed samples for {case}')
            indices=np.linspace(0,len(rows)-1,min(per_case,len(rows)),dtype=int)
            self.samples[case]=[rows[int(index)] for index in indices]

    def attach_codec(self,path):
        path=Path(path).resolve();raw=path.read_bytes();report=json.loads(raw)
        if report.get('schemaVersion')!=1 or report.get('fixedManifestSha256')!=self.sha256:
            raise ValueError('Codec validation does not match frozen inputs')
        originals={json.dumps(row,sort_keys=True) for row in self.manifest['samples'] if row['case']=='clean'}
        rows=report.get('samples',[]);cases=report.get('cases',[])
        if not rows or not cases or len(set(cases))!=len(cases):raise ValueError('Empty or duplicated codec cases')
        if any(row.get('split')!='validation' or row.get('case') not in cases or
               not row['case'].startswith('codec-intra-') or
               json.dumps(row.get('base'),sort_keys=True) not in originals or
               row.get('sourceSha256')!=row['base']['sourceSha256'] for row in rows):
            raise ValueError('Codec validation contains unknown or non-validation samples')
        selected={}
        for case in cases:
            matching=sorted((row for row in rows if row['case']==case),key=lambda row:row['sourceSha256'])
            if not matching:raise ValueError('Missing codec case')
            indices=np.linspace(0,len(matching)-1,min(self.per_case,len(matching)),dtype=int)
            selected[case]=[matching[int(index)] for index in indices]
        self.samples.update(selected);self.codec_folder=path.parent
        self.codec_sha256=hashlib.sha256(raw).hexdigest()

    def render_sample(self,sample):
        if sample['case'].startswith('codec-intra-'):
            from prepare_zenkai_codec import render_codec_sample
            return render_codec_sample(sample,self.codec_folder,allowed_roots=self.allowed_roots)
        return render_validation_sample(sample,allowed_roots=self.allowed_roots)

    def evaluate(self,models,device='cuda',perceptual=None,on_outputs=None):
        if not models:raise ValueError('At least one model is required')
        from PIL import Image
        results={}
        with torch.inference_mode():
            for case,rows in self.samples.items():
                metrics_by_model={name:[] for name in [*models,'bilinear','bicubic']}
                for sample in rows:
                    low,high=self.render_sample(sample)
                    x=torch.from_numpy(np.array(low,copy=True).transpose(2,0,1)).float()[None].to(device)/255
                    target=torch.from_numpy(np.array(high,copy=True).transpose(2,0,1)).float()[None].to(device)/255
                    values={'bilinear':F.interpolate(x,size=target.shape[-2:],mode='bilinear',align_corners=False),
                            'bicubic':torch.from_numpy(np.array(low.resize(high.size,Image.Resampling.BICUBIC),copy=True).transpose(2,0,1)).float()[None].to(device)/255}
                    for name,model in models.items():
                        if isinstance(model,torch.nn.Module):
                            dtype=next(model.parameters(),torch.empty((),device=device)).dtype
                            value=model(x.to(dtype)).float()
                        else:value=model(x).float()
                        if value.shape!=target.shape:raise ValueError(f'{name} output shape does not match x2 target')
                        if not torch.isfinite(value).all():raise ValueError(f'{name} output is non-finite')
                        values[name]=value
                    if on_outputs is not None:on_outputs(case,sample,x,target,values)
                    for name,value in values.items():
                        _,score=score_prediction(value,target)
                        if perceptual is not None:
                            score['lpipsSqueeze']=float(perceptual(value.clamp(0,1)*2-1,target*2-1).mean())
                            if ms_ssim is None:raise RuntimeError('pytorch-msssim is required for perceptual evaluation')
                            score['msSsim']=float(ms_ssim(value.clamp(0,1),target,data_range=1,size_average=True))
                        metrics_by_model[name].append(score)
                results[case]={name:{key:float(np.mean([row[key] for row in rowscores]))
                                     for key in rowscores[0]} for name,rowscores in metrics_by_model.items()}
        return {'scope':'Fixed anime image validation, clean/heavy'+(' and codec-intra' if self.codec_folder else '')+'; never final-hidden.',
                'validationManifest':str(self.manifest_path),'validationManifestSha256':self.sha256,
                'sourceManifestSha256':self.manifest['sourceManifestSha256'],
                'codecManifestSha256':self.codec_sha256,
                'sampleCounts':{case:len(rows) for case,rows in self.samples.items()},
                'indices':{case:[row['sourceSha256'] for row in rows] for case,rows in self.samples.items()},
                'cases':results,'realVideoValidation':'REAL VIDEO VALIDATION NOT EXECUTED'}
