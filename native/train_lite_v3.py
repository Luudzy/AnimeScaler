"""Bounded V3 experiment. No automatic promotion and no test-based selection."""
import sys,json,time,random,hashlib,argparse,subprocess,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT  # Project-local MIOpen caches, before torch initialization.
from artifact_io import atomic_json_write,atomic_torch_save
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lite_model import AdoreLite
from lite_v3 import AdoreLiteV3
from lite_losses import LOSS_WEIGHTS,loss_terms,metrics
from lite_degradations import degrade
from lite_evaluation import ROOT,load_patches,tensor,evaluate,validation_gate,validate_codec_cache

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=5000);p.add_argument('--channels',type=int,default=24)
    p.add_argument('--seed',type=int,default=929);p.add_argument('--resume',type=Path);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--interval',type=int,default=1000);p.add_argument('--output',type=Path,default=ROOT/'models/lite-v3-24-experiment')
    p.add_argument('--patience',type=int,default=10000)
    p.add_argument('--dataset',type=Path,default=ROOT/'native/training-data/v3')
    p.add_argument('--patches-per-image',type=int,default=4)
    args=p.parse_args()
    if min(args.steps,args.batch,args.interval,args.patches_per_image)<1:p.error('Positive counts required')
    args.dataset=args.dataset.resolve();args.output=args.output.resolve()
    if not torch.cuda.is_available() or not torch.version.hip:raise RuntimeError('ROCm required')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=random.Random(args.seed)
    model=AdoreLiteV3(args.channels).cuda();baseline=AdoreLite().cuda().eval()
    with np.load(ROOT/'models/adore-lite-experimental.npz',allow_pickle=False) as w:baseline.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
    saved=None;offset=0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True);model.load_state_dict(saved['model'])
        offset=saved['step']
    patches=load_patches('train',per_image=args.patches_per_image,seed=929,dataset=args.dataset);val=load_patches('validation',dataset=args.dataset)
    if len(patches)<32 or len(val)<8:raise RuntimeError('Dataset too small')
    codec=args.dataset/'codec-validation';cases=['clean','heavy']
    if (codec/'complete.json').exists():cases.append('codec')
    train_codec=args.dataset/'codec-train'
    codec_files=sorted(train_codec.glob('*.png')) if (train_codec/'complete.json').exists() else []
    if codec_files:validate_codec_cache(train_codec,patches)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda');started=time.perf_counter();best={};history=[];last_improvement=offset;best_composite=float('inf')
    if saved and 'optimizer' in saved:
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        rng.setstate(saved['pythonRng']);torch.set_rng_state(saved['torchRng']);torch.cuda.set_rng_state_all(saved['cudaRng'])
    snapshot=lambda:{k:v.detach().float().cpu().clone() for k,v in model.state_dict().items()}
    config={'architecture':'AdoreLiteV3','channels':args.channels,'blocks':4,'residualScale':.2,'scale':2,
            'parameters':sum(v.numel() for v in model.parameters()),'seed':args.seed,'loss':LOSS_WEIGHTS,
            'optimizer':'AdamW','betas':[.9,.99],'weightDecay':1e-5,'lr':[2e-4,1e-5],'batch':args.batch,
            'patchLR':96,'patchHR':192,'patchSeed':929,'patchesPerImage':args.patches_per_image,'datasetPath':str(args.dataset),
            'dataset':json.loads((args.dataset/'dataset.json').read_text()),
            'torch':str(torch.__version__),'hip':torch.version.hip,'gpu':torch.cuda.get_device_name(),
            'gitCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'workingTreeDirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'resume':str(args.resume) if args.resume else None,'selectionSplit':'validation','releaseApproved':False,
            'resumeStep':offset,'resumeKind':'optimizer-and-rng' if saved and 'optimizer' in saved else 'weights-only' if saved else 'from-scratch',
            'postTrainingMetrics':['LPIPS-Squeeze','MS-SSIM'],'realVideoValidation':'pending: no sequences supplied'}
    atomic_json_write(config,args.output/'config.json')
    for step in range(args.steps):
        global_step=offset+step
        model.train();pairs=[]
        for _ in range(args.batch):
            index=rng.randrange(len(patches));high=Image.fromarray(patches[index])
            if codec_files and index<len(codec_files) and rng.random()<.25:
                low=Image.open(train_codec/f'{index:04d}.png').convert('RGB')
            else:low=degrade(high,rng)
            if rng.random()<.5:high=high.transpose(Image.Transpose.FLIP_LEFT_RIGHT);low=low.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pairs.append((tensor(low),tensor(high)))
        x=torch.cat([v[0] for v in pairs]);target=torch.cat([v[1] for v in pairs])
        lr=1e-5+(2e-4-1e-5)*.5*(1+np.cos(np.pi*global_step/max(1,offset+args.steps-1)))
        lr*=min(1,(global_step+1)/300)
        for group in optimizer.param_groups:group['lr']=float(lr)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):pred=model(x)
        terms=loss_terms(pred,target);loss=sum(LOSS_WEIGHTS[k]*v for k,v in terms.items())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update()
        if (step+1)%100==0:
            print(json.dumps({'step':step+1,'globalStep':global_step+1,'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},'lr':float(lr),
                              'samplesPerSecond':(step+1)*args.batch/(time.perf_counter()-started),'allocatedMiB':torch.cuda.memory_allocated()/1048576}),flush=True)
        if (step+1)%args.interval==0 or step+1==args.steps:
            model.eval();report=evaluate({'candidate':model,'previous':baseline},val,cases=cases,codec_cache=codec)
            aggregate={k:float(np.mean([s['candidate'][k] for s in report.values()])) for k in ['mae','edgeMae','ssim','laplacianMae']}
            aggregate['composite']=float(np.mean([s['candidate']['mae']/s['previous']['mae']+.3*s['candidate']['edgeMae']/s['previous']['edgeMae']+.2*s['candidate']['laplacianMae']/s['previous']['laplacianMae']+.1*(1-s['candidate']['ssim'])/max(1e-6,1-s['previous']['ssim']) for s in report.values()]))
            saved={'model':snapshot(),'step':global_step+1,'config':config,'optimizer':optimizer.state_dict(),
                   'scaler':scaler.state_dict(),'pythonRng':rng.getstate(),'torchRng':torch.get_rng_state(),'cudaRng':torch.cuda.get_rng_state_all()}
            atomic_torch_save(saved,args.output/'last-master-fp32.pt')
            for metric,value in aggregate.items():
                score=-value if metric=='ssim' else value
                if metric not in best or score<best[metric]:best[metric]=score;atomic_torch_save(saved,args.output/f'best_{metric}-master-fp32.pt')
            entry={'step':global_step+1,'metrics':report,'aggregate':aggregate,'gate':validation_gate(report)};history.append(entry)
            atomic_json_write(history,args.output/'validation.json')
            print(json.dumps({'validationStep':step+1,'aggregate':aggregate,'gate':entry['gate']['passed']}),flush=True)
            if aggregate['composite']<best_composite*.999:
                best_composite=aggregate['composite'];last_improvement=global_step+1
            if args.patience>0 and global_step+1-last_improvement>=args.patience:
                print('Early stopping: validation plateau',flush=True);break
    saved=torch.load(args.output/'best_composite-master-fp32.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model']);model.eval()
    fp32=evaluate({'candidate':model,'previous':baseline},val,cases=cases,codec_cache=codec)
    deploy=copy.deepcopy(model).half()
    fp16=evaluate({'candidate':deploy,'previous':baseline},val,cases=cases,codec_cache=codec)
    np.savez_compressed(args.output/'deploy-fp16.npz',**{k:v.detach().cpu().numpy() for k,v in deploy.state_dict().items()})
    delta={case:{k:fp16[case]['candidate'][k]-fp32[case]['candidate'][k] for k in fp32[case]['candidate']} for case in cases}
    result={**config,'selectedStep':saved['step'],'stepsExecuted':step+1,'cumulativeExecutedSteps':offset+step+1,'trainingSeconds':time.perf_counter()-started,
            'fp32':fp32,'fp16':fp16,'fp16Delta':delta,'fp16Approved':all(abs(v['mae'])<1e-4 and abs(v['ssim'])<1e-3 for v in delta.values()),
            'validationGate':validation_gate(fp16),'weightsSha256':hashlib.sha256((args.output/'deploy-fp16.npz').read_bytes()).hexdigest()}
    atomic_json_write(result,args.output/'report.json');print(json.dumps({'finished':True,'gate':result['validationGate']['passed']}),flush=True)
if __name__=='__main__':main()
