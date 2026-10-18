"""Read-only CPU attention/gradient snapshots using the existing tested diagnostic."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
import numpy as np
import torch
from zenkai_activity import gradient_flow_probe
from zenkai_recipe_loss import make_experiment_loss
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_probe_evaluation import load_probe
from zenkai_evaluation import ZenkaiFixedEvaluator
from zenkai_domain_sampler import INITIAL_SHA
from artifact_provenance import validate_checkpoint_file

def main():
 out=ROOT/'diagnostics/zenkai-domain10-1k/attention-cpu.json'
 if out.exists():raise FileExistsError('Preserve priordiagnostic')
 torch.set_num_threads(2)
 source=ROOT/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/initial-training-state.pt';validate_checkpoint_file(source,INITIAL_SHA)
 initial=torch.load(source,map_location='cpu',weights_only=True)
 specs=[('actual-initialization',source,0),('uniform1k',ROOT/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/step-1000-master-fp32.pt',1000),('domain10-1k',ROOT/'models/zenkai/experiments/zenkai-ablation-skip-v2-domain10-1k/step-1000-master-fp32.pt',1000),('C5k-context-only',ROOT/'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k/step-5000-master-fp32.pt',5000)]
 evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=1)
 if evaluator.sha256!=initial['config']['fixedValidationSha256']:raise ValueError('Wrong fixed inputs')
 criterion=make_experiment_loss('zenkai_frequency')
 r={'scope':'Additional CPU FP32 diagnostic of actual archived initialization and weights;not AMP/HIP performance/training updates','torch':str(torch.__version__),'fixedValidationSha256':evaluator.sha256,'checkpoints':{},'releaseApproved':False,'limitations':['C5k differs in budget/codec/dataset;context only,not sampler causal comparison','CoreSPAB output excludes separate added SkipBlock identity path;core output/input ratio is not wholeblock attenuation','Mean/max centered gate does not measure saturation fraction','Two local crops do not establish dataset-wide behavior']}
 for name,path,step in specs:
  pin=hashlib.sha256(path.read_bytes()).hexdigest()
  if step==0:
   model=IdentitySkipProbe().eval();model.load_state_dict(initial['model'],strict=True)
  else:model,_=load_probe(path,step)
  records=[]
  for case in ('clean','heavy'):
   sample=evaluator.samples[case][0];low,high=evaluator.render_sample(sample)
   convert=lambda im:torch.from_numpy(np.asarray(im).copy().transpose(2,0,1)).float()[None]/255
   stats=gradient_flow_probe(model,convert(low),convert(high),lambda p,t:criterion(p,t)[0])
   records.append({'case':case,'sourceSha256':sample['sourceSha256'],**stats})
  validate_checkpoint_file(path,pin)
  r['checkpoints'][name]={'path':str(path),'sha256':pin,'step':step,'probes':records}
 out.write_text(json.dumps(r,indent=2),encoding='utf-8')
 print('Actual0/1k/C5k read-only CPU attention probes saved;no reconstructed state or optimizer updates')
if __name__=='__main__':main()
