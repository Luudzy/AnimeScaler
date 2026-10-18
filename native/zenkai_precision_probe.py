"""ROCm FP32 versus scaled autocast backward, immutable screening checkpoints."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from export_zenkai import ROOT,load_checkpoint,sha256,configure_cache
from zenkai_evaluation import ZenkaiFixedEvaluator
from zenkai_losses import ZenkaiLoss


def summarize_gradients(model,scale):
    if scale<=0:raise ValueError('Positive scale required')
    groups={}
    for name,parameter in model.named_parameters():
        parts=name.split('.')
        group='.'.join(parts[:2]) if parts[0]=='blocks' else parts[0]
        row=groups.setdefault(group,{'elements':0,'missing':0,'zero':0,'squared':0.})
        row['elements']+=parameter.numel()
        if parameter.grad is None:
            row['missing']+=parameter.numel();continue
        gradient=parameter.grad.detach().cpu().double()/scale
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite gradient')
        row['zero']+=int((gradient==0).sum())
        row['squared']+=float(gradient.square().sum())
    return {name:{'gradientL2':row['squared']**.5,
        'zeroElementsPct':100*row['zero']/row['elements'],
        'missingElementsPct':100*row['missing']/row['elements']} for name,row in groups.items()}


def probe(model,x,target,criterion,amp,scale):
    clone=copy.deepcopy(model).cuda().float().train()
    clone.zero_grad(set_to_none=True)
    # Same accumulation normalization as training. No optimizer step.
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):prediction=clone(x)
    loss=criterion(prediction,target)[0]/2
    if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
    (loss*scale).backward()
    result={'lossBeforeAccumulationNormalization':float(loss.detach())*2,
        'scale':scale,'groups':summarize_gradients(clone,scale)}
    del clone
    return result


def main():
    configure_cache(ROOT)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    output=ROOT/'diagnostics/zenkai-gradient-precision.json'
    if output.exists():raise FileExistsError('Refusing to overwrite probe')
    torch.set_num_threads(1)
    evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=2)
    report={'scope':'Matched GPU backward only, no optimizer update; FP32 versus AMP FP16 with explicit loss scale',
        'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'hip':torch.version.hip,
        'validationManifestSha256':evaluator.sha256,'lossScales':[65536.,262144.],
        'scaleCaveat':'Fixed probe scales, not reconstructed historical A/B scaler history',
        'runs':{},'releaseApproved':False}
    convert=lambda im:torch.from_numpy(np.asarray(im).copy().transpose(2,0,1)).float()[None].cuda()/255
    for letter in ('a','b'):
        path=ROOT/f'models/zenkai/experiments/zenkai-lite-v1-{letter}-5k/step-5000-master-fp32.pt'
        digest=sha256(path);model,saved=load_checkpoint(path)
        if evaluator.sha256!=saved['config']['fixedValidationSha256']:raise ValueError('Manifest changed')
        criterion=ZenkaiLoss(saved['config']['loss']).cuda()
        cases=[]
        for case in ('clean','heavy'):
            sample=evaluator.samples[case][0];low,high=evaluator.render_sample(sample)
            x,target=convert(low),convert(high)
            fp32=probe(model,x,target,criterion,False,1.)
            amp={str(scale):probe(model,x,target,criterion,True,scale) for scale in report['lossScales']}
            cases.append({'case':case,'sourceSha256':sample['sourceSha256'],'fp32':fp32,'amp':amp})
        if sha256(path)!=digest:raise ValueError('Checkpoint changed during probe')
        report['runs'][letter]={'checkpointSha256':digest,'step':saved['step'],'cases':cases}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'state':'complete','output':str(output),'releaseApproved':False}))


if __name__=='__main__':main()
