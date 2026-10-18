"""Measure learned weight changes and correction relative to the same-size resize.

Correction strength is not a quality metric and never approves a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from artifact_io import atomic_json_create, atomic_torch_save
from statistics import mean
import numpy as np
import torch
from torch.nn import functional as F
from export_zenkai import ROOT,load_checkpoint
from zenkai_model import ZenkaiLite,SPAB
from zenkai_evaluation import ZenkaiFixedEvaluator


def correction_stats(prediction,baseline):
    if prediction.shape!=baseline.shape or prediction.ndim!=4 or prediction.shape[1]!=3:
        raise ValueError('Correction comparison requires the same RGB NCHW resolution')
    if not torch.isfinite(prediction).all() or not torch.isfinite(baseline).all():
        raise ValueError('Non-finite correction')
    visible=prediction.float().clamp(0,1);base=baseline.float().clamp(0,1)
    error=(visible-base).abs()
    changed=(torch.round(visible*255)!=torch.round(base*255)).any(dim=1)
    return {'visibleCorrectionMeanAbs':float(error.mean()),'visibleCorrectionMaxAbs':float(error.max()),
            'channelsChangedGtOneLsbPct':float((error>1/255).float().mean()*100),
            'quantizedPixelChangePct':float(changed.float().mean()*100)}


def weight_change_summary(trained,initial):
    if trained.keys()!=initial.keys():raise ValueError('Checkpoint/initialization keys differ')
    groups={}
    for name,value in trained.items():
        reference=initial[name]
        if value.shape!=reference.shape or not torch.isfinite(value).all() or not torch.isfinite(reference).all():
            raise ValueError('Invalid weight comparison')
        components=name.split('.');group='.'.join(components[:2]) if components[0]=='blocks' else components[0]
        record=groups.setdefault(group,{'elements':0,'deltaSquared':0.,'initialSquared':0.,'deltaAbs':0.,'changed':0})
        reference=reference.detach().cpu().double();delta=value.detach().cpu().double()-reference
        record['elements']+=value.numel();record['deltaSquared']+=float(delta.square().sum())
        record['initialSquared']+=float(reference.square().sum());record['deltaAbs']+=float(delta.abs().sum())
        record['changed']+=int((delta.abs()>1e-6).sum())
    return {group:{'elements':record['elements'],'deltaL2':record['deltaSquared']**.5,
                   'initialL2':record['initialSquared']**.5,
                   'deltaToInitialRatio':(record['deltaSquared']/record['initialSquared'])**.5 if record['initialSquared']>1e-24 else None,
                   'meanAbsChange':record['deltaAbs']/record['elements'],
                   'changedElementsPct':record['changed']/record['elements']*100} for group,record in groups.items()}


def gradient_flow_probe(model,value,target,loss_function):
    if value.dtype!=torch.float32 or target.dtype!=torch.float32:
        raise ValueError('Gradient diagnostic requires FP32 input and target')
    model.zero_grad(set_to_none=True)
    groups={};attention={};handles=[]
    for name,block in model.named_modules():
        if not isinstance(block,SPAB):continue
        def capture_gate(module,inputs,result,name=name):
            gate=torch.sigmoid(result.detach().float())-.5
            attention.setdefault(name,{}).update(meanAbsGate=float(gate.abs().mean()),maxAbsGate=float(gate.abs().max()))
        def capture_rms(module,inputs,result,name=name):
            attention.setdefault(name,{}).update(inputRms=float(inputs[0].detach().float().square().mean().sqrt()),
                                                outputRms=float(result.detach().float().square().mean().sqrt()))
        handles.append(block.convolutions[-1].register_forward_hook(capture_gate))
        handles.append(block.register_forward_hook(capture_rms))
    try:
        with torch.enable_grad(),torch.autocast(device_type=value.device.type,enabled=False):
            prediction=model(value)
            loss=loss_function(prediction,target)
            if not torch.isfinite(loss):raise ValueError('Non-finite gradient probe loss')
            loss.backward()
        for name,parameter in model.named_parameters():
            components=name.split('.');group='.'.join(components[:2]) if components[0]=='blocks' else components[0]
            record=groups.setdefault(group,{'elements':0,'squared':0.,'absSum':0.,'max':0.,'missing':0,'aboveThreshold':0})
            record['elements']+=parameter.numel()
            if parameter.grad is None:record['missing']+=parameter.numel();continue
            gradient=parameter.grad.detach().double()
            if not torch.isfinite(gradient).all():raise ValueError('Non-finite probe gradient')
            record['squared']+=float(gradient.square().sum());record['absSum']+=float(gradient.abs().sum())
            record['max']=max(record['max'],float(gradient.abs().max()))
            record['aboveThreshold']+=int((gradient.abs()>1e-12).sum())
        return {'loss':float(loss.detach()),'device':str(value.device),'precision':'fp32',
                'scope':'FP32 diagnostic backward only; no optimizer update; not a measured AMP underflow test',
                'attention':attention,'groups':{name:{'gradientL2':row['squared']**.5,'meanAbsGradient':row['absSum']/row['elements'],
                    'maxAbsGradient':row['max'],'missingGradientElementsPct':100*row['missing']/row['elements'],
                    'gradientElementsGt1eMinus12Pct':100*row['aboveThreshold']/row['elements']} for name,row in groups.items()}}
    finally:
        for handle in handles:handle.remove()
        model.zero_grad(set_to_none=True)


def model_code_hash():
    return hashlib.sha256(Path(__file__).with_name('zenkai_model.py').read_bytes()).hexdigest()


def load_initial_reference(path,config):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('torch')!=config['torch'] or saved.get('modelCodeSha256')!=model_code_hash():
        raise ValueError('Initialization reference used a different Torch version or model code')
    if any(saved.get('config',{}).get(key)!=config[key] for key in ('seed','channels','depth','nativeScale','torch')):
        raise ValueError('Initialization reference seed/configuration differs')
    return saved['model']


def write_initial_reference(checkpoint,output):
    _,saved=load_checkpoint(checkpoint);config=saved['config'];output=Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Initialization reference must be new and inside workspace')
    if str(torch.__version__)!=config['torch']:
        raise ValueError('Reconstruct initialization with the training interpreter')
    torch.manual_seed(config['seed'])
    initial=ZenkaiLite(channels=config['channels'],depth=config['depth'])
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch_save({'model':initial.state_dict(),'config':config,'torch':str(torch.__version__),
                'modelCodeSha256':model_code_hash(),
                'kind':'Reconstructed initialization with training seed/code/interpreter; not an archived step-zero snapshot'},output)
    return {'step':'initialization-reference','sourceTorch':str(torch.__version__),'output':str(output)}


def inspect_checkpoint(checkpoint,per_case=8,codec=None,gradient_probes=0,initial_reference=None):
    checkpoint=Path(checkpoint).resolve();raw=checkpoint.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
    model,saved=load_checkpoint(checkpoint);config=saved['config']
    if initial_reference:
        initial_state=load_initial_reference(initial_reference,config)
    else:
        if str(torch.__version__)!=config['torch']:
            raise ValueError('Different Torch version: provide --initial-reference made by the training interpreter')
        torch.manual_seed(config['seed'])
        initial_state=ZenkaiLite(channels=config['channels'],depth=config['depth']).state_dict()
    weight_groups=weight_change_summary(model.state_dict(),initial_state)
    deploy=model.deploy_copy().eval()
    evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=per_case)
    if evaluator.sha256!=config['fixedValidationSha256']:raise ValueError('Fixed validation changed since training')
    if codec:evaluator.attach_codec(codec)
    if type(gradient_probes) is not int or not 0<=gradient_probes<=4:raise ValueError('Use zero to four gradient probes')
    cases={};samples=[]
    with torch.inference_mode():
        for case,rows in evaluator.samples.items():
            scores=[]
            for row in rows:
                low,_=evaluator.render_sample(row)
                value=torch.from_numpy(np.array(low,copy=True).transpose(2,0,1)).float()[None]/255
                baseline=F.interpolate(value,scale_factor=2,mode='bilinear',align_corners=False)
                stats=correction_stats(deploy(value),baseline);scores.append(stats)
                samples.append({'case':case,'sourceSha256':row['sourceSha256'],'inputShape':list(value.shape),
                                'commonOutputShape':list(baseline.shape),'correction':stats})
            cases[case]={key:mean(score[key] for score in scores) for key in scores[0]}
            cases[case]['peakVisibleCorrectionMaxAbs']=max(score['visibleCorrectionMaxAbs'] for score in scores)
    probes=[]
    if gradient_probes:
        from zenkai_losses import ZenkaiLoss
        criterion=ZenkaiLoss(config['loss'])
        for index in range(gradient_probes):
            case=('clean','heavy')[index%2];rows=evaluator.samples[case]
            row=rows[0 if index<2 else -1]
            low,high=evaluator.render_sample(row)
            convert=lambda image:torch.from_numpy(np.array(image,copy=True).transpose(2,0,1)).float()[None]/255
            probe=gradient_flow_probe(model,convert(low),convert(high),lambda a,b:criterion(a,b)[0])
            probes.append({'case':case,'sourceSha256':row['sourceSha256'],**probe})
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=checksum:raise ValueError('Checkpoint changed during inspection')
    return {'schemaVersion':1,'checkpoint':str(checkpoint),'checkpointSha256':checksum,'step':saved['step'],
            'scope':'CPU FP32, learned correction compared with bilinear at identical output resolution',
            'qualityConclusion':'Correction magnitude demonstrates contribution, not improved quality',
            'initializationSeed':config['seed'],'weightChangeThreshold':1e-6,'weightGroups':weight_groups,
            'initializationTorch':config['torch'],'diagnosticTorch':str(torch.__version__),
            'initializationReferenceSha256':hashlib.sha256(Path(initial_reference).read_bytes()).hexdigest() if initial_reference else None,
            'validationManifestSha256':evaluator.sha256,'codecManifestSha256':evaluator.codec_sha256,
            'cases':cases,'samples':samples,'gradientProbes':probes,'releaseApproved':False,'realVideoValidation':'REAL VIDEO VALIDATION NOT EXECUTED'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-case',type=int,default=8)
    parser.add_argument('--gradient-probes',type=int,choices=range(5),default=0)
    parser.add_argument('--initial-reference',type=Path)
    parser.add_argument('--write-initial-reference',action='store_true',help='CPU-only reconstruction using the training interpreter; writes .pt')
    parser.add_argument('--codec',type=Path,default=ROOT/'native/training-data/zenkai/codec-validation-fixed/complete.json')
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():parser.error('Output must be new and inside workspace')
    torch.set_num_threads(2)
    if args.write_initial_reference:
        print(json.dumps(write_initial_reference(args.checkpoint,output)));return
    report=inspect_checkpoint(args.checkpoint,args.per_case,args.codec,args.gradient_probes,args.initial_reference)
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json_create(report,output)
    print(json.dumps({'step':report['step'],'output':str(output),'releaseApproved':False}))


if __name__=='__main__':main()
