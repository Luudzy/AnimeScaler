"""Matched final3k backward diagnostics, using actual stored AMP scales/init weights."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
import numpy as np
import torch
from export_zenkai import configure_cache,sha256
from zenkai_probe_evaluation import load_probe,load_reference_checkpoint
from zenkai_precision_probe import probe
from zenkai_activity import weight_change_summary
from zenkai_losses import ZenkaiLoss
from zenkai_evaluation import ZenkaiFixedEvaluator

def main():
 status=json.loads((ROOT/'diagnostics/zenkai-training/ablations3k-evaluation-recovery5.status.json').read_text(encoding='utf-8-sig'))
 if status['state']!='complete':raise RuntimeError('Wait for all GPU stages to finish')
 output=ROOT/'diagnostics/zenkai-ablation-gradient-analysis-3k.json'
 if output.exists():raise FileExistsError('Preserving existing analysis')
 configure_cache(ROOT)
 if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
 torch.set_num_threads(1)
 evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=2)
 convert=lambda im:torch.from_numpy(np.asarray(im).copy().transpose(2,0,1)).float()[None].cuda()/255
 report={'scope':'Matched3k checkpoint-local GPU backward, no optimizer step; accumulated weight changes are observed init→3k, not per-step updates',
  'validationManifestSha256':evaluator.sha256,'torch':str(torch.__version__),'hip':torch.version.hip,
  'gpu':torch.cuda.get_device_name(),'runs':{},'releaseApproved':False}
 for run in ['zenkai-ablation-frequency-3k-continuation','zenkai-ablation-spab-identity-skip-3k-continuation']:
  folder=ROOT/'models/zenkai/experiments'/run;path=folder/'step-3000-master-fp32.pt'
  if run.endswith('charbonnier-1k'):
   from export_zenkai import load_checkpoint
   model,saved=load_checkpoint(path)
  else:model,saved=load_reference_checkpoint(path,3000) if run=='zenkai-ablation-frequency-3k-continuation' else load_probe(path,3000)
  config=saved['config']
  if evaluator.sha256!=config['fixedValidationSha256']:raise ValueError('Manifest changed')
  digest=sha256(path);initial=Path(config['continuedFrom']).parent/'initial-training-state.pt';state=folder/'step-3000-training-state.pt'
  start=torch.load(initial,map_location='cpu',weights_only=True);snapshot=torch.load(state,map_location='cpu',weights_only=True)
  scale=snapshot['scaler']['scale']
  criterion=ZenkaiLoss(config['loss']).cuda();cases=[]
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
