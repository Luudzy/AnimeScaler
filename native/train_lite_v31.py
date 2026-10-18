"""From-scratch bounded V3.1 screening; never publishes a model automatically."""
import sys,json,time,random,math,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT
from artifact_io import atomic_json_write,atomic_torch_save,atomic_torch_save_new
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler
from lite_v31 import AdoreLiteV31,VARIANTS,receptive_field
from lite_v3 import AdoreLiteV3
from lite_model import AdoreLite
from lite_online_dataset import OnlineAnimeDataset
from lite_v31_losses import reconstruction_loss,PRESETS
from lite_v31_evaluation import FixedEvaluator

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'native/training-data/v31')
    p.add_argument('--variant',choices=VARIANTS,default='b');p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--seed',type=int,default=929);p.add_argument('--patch-lr',type=int,choices=[96,128],default=128)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--loss',choices=PRESETS,default='detail');p.add_argument('--residual-scale',type=float,default=.5)
    p.add_argument('--edge-weight',type=float,default=0);p.add_argument('--interval',type=int,default=1000)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if min(args.steps,args.batch,args.accumulation,args.interval)<1:p.error('Positive counts required')
    if not (args.dataset/'codec-full/complete.json').exists():raise RuntimeError('Full codec pairs not ready')
    if ((args.output/'last.pt').exists() or any(args.output.glob('step-*.pt'))
            or any(args.output.glob('best-*.pt'))):
        raise RuntimeError('Use a new experiment directory; no implicit resume or overwrite')
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed)
    if not torch.cuda.is_available() or not torch.version.hip:raise RuntimeError('ROCm required')
    args.output.mkdir(parents=True,exist_ok=True)
    model=AdoreLiteV31(args.variant,args.residual_scale).cuda()
    v2=AdoreLite().cuda().eval();v3=AdoreLiteV3().cuda().eval()
    for baseline,file in [(v2,'adore-lite-experimental.npz'),(v3,'adore-lite-v3-experimental.npz')]:
        with np.load(ROOT/'models'/file) as weights:baseline.load_state_dict({k:torch.from_numpy(weights[k].copy()).float() for k in weights.files})
    data=OnlineAnimeDataset(args.dataset,args.patch_lr,args.seed)
    generator=torch.Generator().manual_seed(args.seed)
    sampler=RandomSampler(data,replacement=True,num_samples=args.steps*args.batch*args.accumulation,generator=generator)
    # One worker owns RNG and unique-crop counters. No duplicated Windows workers.
    batches=iter(DataLoader(data,batch_size=args.batch,sampler=sampler,num_workers=0))
    evaluator=FixedEvaluator(args.dataset,ROOT/'native/training-data/v3-hfa2k/codec-validation')
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda');started=time.perf_counter();history=[];best={}
    config={**vars(args),'architecture':'AdoreLiteV31','channels':VARIANTS[args.variant][0],
            'dilations':VARIANTS[args.variant][1],'receptiveFieldLR':receptive_field(VARIANTS[args.variant][1]),
            'parameters':sum(v.numel() for v in model.parameters()),'lossWeights':PRESETS[args.loss],
            'datasetReport':json.loads((args.dataset/'dataset.json').read_text()),'effectiveBatch':args.batch*args.accumulation,
            'initialization':'from scratch; zero tail','promoted':False,'gpu':torch.cuda.get_device_name(),
            'realVideoValidation':'REAL VIDEO VALIDATION NOT EXECUTED'}
    config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    atomic_json_write(config,args.output/'config.json')
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        lr=(1e-5+(2e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/max(1,args.steps-1))))*min(1,step/300)
        for group in optimizer.param_groups:group['lr']=lr
        for _ in range(args.accumulation):
            x,target=next(batches);x=x.cuda();target=target.cuda()
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x)
            loss,terms=reconstruction_loss(pred,target,args.loss,args.edge_weight,evaluator.feature)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss/args.accumulation).backward()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'terms':terms,'lr':lr,**data.unique_crops.report_fields(),'samples':data.samples}),flush=True)
        if step%args.interval==0 or step==args.steps:
            model.eval();scores=evaluator.run({'v31':model,'v3':v3,'v2':v2})
            summary={key:float(np.mean([case['anime']['v31'][key] for case in scores['cases'].values()])) for key in ['mae','edgeMae','laplacianMae','ssim','lpipsSqueeze','msSsim']}
            summary['composite']=float(np.mean([case['anime']['v31']['mae']/case['anime']['v3']['mae']+.3*case['anime']['v31']['edgeMae']/case['anime']['v3']['edgeMae'] for case in scores['cases'].values()]))
            checkpoint={'model':{k:v.detach().float().cpu().clone() for k,v in model.state_dict().items()},'step':step,'config':config}
            # Retain every screening checkpoint for top-three review, not one winner.
            atomic_torch_save_new(checkpoint,args.output/f'step-{step}.pt');atomic_torch_save(checkpoint,args.output/'last.pt')
            for key,value in summary.items():
                lower=-value if key in ('ssim','msSsim') else value
                if key not in best or lower<best[key]:best[key]=lower;atomic_torch_save(checkpoint,args.output/f'best-{key}.pt')
            history.append({'step':step,'summaryAnime':summary,'evaluation':scores})
            report={**config,'stepsExecuted':step,'trainingSeconds':time.perf_counter()-started,**data.unique_crops.report_fields(),
                    'samples':data.samples,'codecSamples':data.codec_samples,'hardSamples':data.hard_samples,'history':history,
                    'perceptual':{case:{d:{m:{k:v for k,v in metrics.items() if k in ('lpipsSqueeze','msSsim')} for m,metrics in models.items()} for d,models in domains.items()} for case,domains in scores['cases'].items()},
                    'releaseDecision':'PENDING visual/temporal/cost review; no automatic promotion'}
            atomic_json_write(report,args.output/'report.json');print(json.dumps({'validationStep':step,'anime':summary}),flush=True)
    print(json.dumps({'finished':True,'promoted':False}),flush=True)
if __name__=='__main__':main()
