"""Matched final1k backward diagnostics, using actual stored AMP scales/init weights."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
import numpy as np
import torch
from export_zenkai import configure_cache,sha256
from zenkai_probe_evaluation import load_probe,load_reference_checkpoint
from zenkai_precision_probe import probe
from zenkai_activity import weight_change_summary
from zenkai_recipe_loss import make_experiment_loss
from zenkai_evaluation import ZenkaiFixedEvaluator

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--factor',action='append',choices=('dataset-sampling','crop96','codec50','scheduler-constant'))
 parser.add_argument('--quality-exit',type=Path,default=ROOT/'diagnostics/zenkai-training/factor-screens-1k-fixed-quality.exit.json')
 parser.add_argument('--output',type=Path,default=ROOT/'diagnostics/zenkai-factor-gradient-analysis.json')
 args=parser.parse_args()
 if args.factor and len(set(args.factor))!=len(args.factor):parser.error('Distinct factors required')
 for path in (args.quality_exit,args.output):
  if not path.resolve().is_relative_to(ROOT):parser.error('Workspace paths required')
 status=json.loads(args.quality_exit.read_text(encoding='utf-8-sig'))
 if status['exitCode']!=0:raise RuntimeError('Factor quality must finish first')
 output=args.output
 if output.exists():raise FileExistsError('Preserving existing analysis')
 configure_cache(ROOT)
 if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
 torch.set_num_threads(1)
 evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=2)
 convert=lambda im:torch.from_numpy(np.asarray(im).copy().transpose(2,0,1)).float()[None].cuda()/255
 report={'scope':'Matched1k checkpoint-local GPU backward, no optimizer step; accumulated weight changes are observed init→1k, not per-step updates',
  'validationManifestSha256':evaluator.sha256,'torch':str(torch.__version__),'hip':torch.version.hip,
  'gpu':torch.cuda.get_device_name(),'runs':{},'releaseApproved':False}
 from zenkai_factor_screens import FACTOR_SPECS
 for run in ['zenkai-ablation-spab-identity-skip-1k']+['zenkai-ablation-skip-factor-'+factor+'-1k' for factor in (args.factor or FACTOR_SPECS)]:
  folder=ROOT/'models/zenkai/experiments'/run;path=folder/'step-1000-master-fp32.pt'
  model,saved=load_probe(path)
  config=saved['config']
  if evaluator.sha256!=config['fixedValidationSha256']:raise ValueError('Manifest changed')
  digest=sha256(path);initial=folder/'initial-training-state.pt';state=folder/'step-1000-training-state.pt'
  start=torch.load(initial,map_location='cpu',weights_only=True);snapshot=torch.load(state,map_location='cpu',weights_only=True)
  scale=snapshot['scaler']['scale']
  criterion=make_experiment_loss(config['loss']).cuda();cases=[]
  for case in ('clean','heavy'):
   sample=evaluator.samples[case][0];low,high=evaluator.render_sample(sample);x,target=convert(low),convert(high)
   cases.append({'case':case,'sourceSha256':sample['sourceSha256'],
    'fp32':probe(model,x,target,criterion,False,1.),'amp':probe(model,x,target,criterion,True,scale)})
  assert sha256(path)==digest
  report['runs'][run]={'checkpointSha256':digest,'initialSha256':sha256(initial),'trainingStateSha256':sha256(state),
   'actualSnapshotLossScale':scale,'lossProfile':config['loss'],'config':config,'cases':cases,
   'accumulatedWeightChanges':weight_change_summary(model.state_dict(),start['model']),
   'currentWeightL2':{group:row['initialL2'] for group,row in weight_change_summary(model.state_dict(),model.state_dict()).items()}}
 output.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'state':'complete','output':str(output),'releaseApproved':False}))
if __name__=='__main__':main()
