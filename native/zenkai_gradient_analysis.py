"""Immutable CPU gradient diagnosis; simulated updates are not historical updates."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from export_zenkai import ROOT,load_checkpoint,sha256
from zenkai_activity import gradient_flow_probe,weight_change_summary,load_initial_reference
from zenkai_model import ZenkaiLite,GRepConv
from zenkai_losses import ZenkaiLoss
from zenkai_evaluation import ZenkaiFixedEvaluator
from train_zenkai import learning_rate


def simulated_update(model,x,target,loss_function,lr):
    clone=copy.deepcopy(model).float().train()
    before={key:value.detach().clone() for key,value in clone.state_dict().items()}
    optimizer=torch.optim.AdamW(clone.parameters(),lr=lr,betas=(.9,.99),weight_decay=1e-5)
    optimizer.zero_grad(set_to_none=True)
    loss=loss_function(clone(x),target)
    if not torch.isfinite(loss):raise ValueError('Nonfinite diagnostic loss')
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(clone.parameters(),1.)
    if not torch.isfinite(grad):raise ValueError('Nonfinite diagnostic gradient')
    optimizer.step()
    summary=weight_change_summary(clone.state_dict(),before)
    return {'scope':'Simulated fresh-state AdamW step on a copy; not an actual historical optimizer update',
            'lr':lr,'loss':float(loss.detach()),'groups':{name:{'weightL2':row['initialL2'],
            'updateL2':row['deltaL2'],'updateToWeightRatio':row['deltaToInitialRatio']} for name,row in summary.items()}}


def branch_activity(model,x):
    measurements={};handles=[]
    for name,module in model.named_modules():
        if not isinstance(module,GRepConv):continue
        for label,branch in [('direct3',module.direct3),('direct1',module.direct1),
                              *((f'seq13.{i}',item) for i,item in enumerate(module.seq13))]:
            def hook(module,inputs,result,name=name,label=label):
                measurements.setdefault(name,{})[label]=float(result.detach().float().square().mean().sqrt())
            handles.append(branch.register_forward_hook(hook))
    try:
        with torch.inference_mode():model(x)
    finally:
        for handle in handles:handle.remove()
    return measurements


def main():
    output=ROOT/'diagnostics/zenkai-gradient-analysis.json'
    if output.exists():raise FileExistsError('Refusing to overwrite gradient audit')
    torch.set_num_threads(1)
    evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=2)
    report={'scope':'CPU FP32 probes; initialization reconstructed with exact training interpreter/seed',
        'validationManifestSha256':evaluator.sha256,
        'historicalOptimizerUpdates':'NOT MEASURED: A/B checkpoints do not save optimizer/RNG',
        'runs':{},'releaseApproved':False,'causalConclusion':'Diagnostic observations, not proof of cause'}
    convert=lambda im:torch.from_numpy(np.asarray(im).copy().transpose(2,0,1)).float()[None]/255
    for letter in ('a','b'):
        folder=ROOT/f'models/zenkai/experiments/zenkai-lite-v1-{letter}-5k'
        _,metadata=load_checkpoint(folder/'step-5000-master-fp32.pt');config=metadata['config']
        if evaluator.sha256!=config['fixedValidationSha256']:raise ValueError('Gradient probe validation manifest changed')
        reference=ROOT/f'diagnostics/zenkai-initial-{letter}-seed929.pt'
        initial=load_initial_reference(reference,config)
        records=[]
        for step in (0,1000,3000,5000):
            if step:
                checkpoint=folder/f'step-{step}-master-fp32.pt';model,_=load_checkpoint(checkpoint)
                checkpoint_hash=sha256(checkpoint)
            else:
                model=ZenkaiLite(32,config['depth']).eval();model.load_state_dict(initial);checkpoint_hash=sha256(reference)
            criterion=ZenkaiLoss(config['loss'])
            probes=[]
            for case in ('clean','heavy'):
                sample=evaluator.samples[case][0];low,high=evaluator.render_sample(sample)
                x,target=convert(low),convert(high)
                loss_fn=lambda a,b:criterion(a,b)[0]
                gradient=gradient_flow_probe(model,x,target,loss_fn)
                update=simulated_update(model,x,target,loss_fn,learning_rate(max(1,step),5000))
                probes.append({'case':case,'sourceSha256':sample['sourceSha256'],
                    'gradient':gradient,'simulatedUpdate':update,
                    'branchRms':branch_activity(model,x) if step in (0,5000) else 'NOT MEASURED'})
            records.append({'step':step,'checkpointSha256':checkpoint_hash,
                'accumulatedWeightChange':weight_change_summary(model.state_dict(),initial),'probes':probes})
        report['runs'][letter]=records
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Zenkai gradient analysis','','CPU FP32 diagnostic. No A/B checkpoint modified.',
           '',report['historicalOptimizerUpdates'],'',
           'Initialization is reconstructed, not an archived step-zero snapshot.',
           'Update ratios below are simulated fresh-state AdamW steps, not observed optimizer history.',
           '','| Run | Step | Case | Block | Gradient norm | Weight norm | Simulated update norm | Update/weight |',
           '|---|---:|---|---|---:|---:|---:|---:|']
    for run,records in report['runs'].items():
        for record in records:
            for probe in record['probes']:
                for group,gradient in probe['gradient']['groups'].items():
                    update=probe['simulatedUpdate']['groups'][group]
                    lines.append(f"| {run} | {record['step']} | {probe['case']} | {group} | {gradient['gradientL2']:.3g} | {update['weightL2']:.3g} | {update['updateL2']:.3g} | {update['updateToWeightRatio']} |")
    lines+=['','Gate magnitudes, input/output RMS, GRep branch RMS and accumulated changes are retained in JSON.',
            'Small FP32 gradients do not by themselves prove AMP underflow, dead blocks or attention saturation.',
            'Per-step historical update norms cannot be recovered from sparse weights-only checkpoints.']
    (ROOT/'docs/zenkai-gradient-analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'state':'complete','output':str(output)}))


if __name__=='__main__':main()
