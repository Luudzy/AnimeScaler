"""New V2 dataset smoke and paired baseline1k, same FP32/scaledAMP backward diagnostics."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
import numpy as np
import torch
from export_zenkai import configure_cache,sha256
from zenkai_probe_evaluation import load_probe
from zenkai_precision_probe import probe
from zenkai_activity import weight_change_summary
from zenkai_recipe_loss import make_experiment_loss
from zenkai_evaluation import ZenkaiFixedEvaluator


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):raise ValueError('New workspace report required')
    configure_cache(ROOT);torch.set_num_threads(1)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=2)
    convert=lambda im:torch.from_numpy(np.asarray(im).copy().transpose(2,0,1)).float()[None].cuda()/255
    runs=('zenkai-ablation-spab-identity-skip-1k','zenkai-ablation-skip-v2-dataset-1k')
    report={'scope':'Checkpoint-local1k FP32/actual-scaledAMP backward; no optimizer update; accumulated init→1k changes',
            'validationManifestSha256':evaluator.sha256,'runs':{},'releaseApproved':False}
    for run in runs:
        folder=ROOT/'models/zenkai/experiments'/run;path=folder/'step-1000-master-fp32.pt'
        model,saved=load_probe(path);snapshot=torch.load(folder/'step-1000-training-state.pt',map_location='cpu',weights_only=True)
        initial=folder/'initial-training-state.pt';origin=torch.load(initial,map_location='cpu',weights_only=True)
        if saved['config']['fixedValidationSha256']!=evaluator.sha256:raise ValueError('Fixed manifest changed')
        scale=snapshot['scaler']['scale'];criterion=make_experiment_loss('zenkai_frequency').cuda();cases=[]
        for case in ('clean','heavy'):
            sample=evaluator.samples[case][0];low,high=evaluator.render_sample(sample);x,target=convert(low),convert(high)
            cases.append({'case':case,'sourceSha256':sample['sourceSha256'],
                          'fp32':probe(model,x,target,criterion,False,1.),'amp':probe(model,x,target,criterion,True,scale)})
        report['runs'][run]={'checkpointSha256':sha256(path),'trainingStateSha256':sha256(folder/'step-1000-training-state.pt'),
            'initialSha256':sha256(initial),'actualSnapshotLossScale':scale,'cases':cases,
            'accumulatedWeightChanges':weight_change_summary(model.state_dict(),origin['model'])}
    report['state']='PASS';args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
