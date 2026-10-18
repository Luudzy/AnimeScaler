"""Clean-start long run on the full real-anime pair split."""
import argparse,json,random,time,sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader,RandomSampler
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(HERE))
from artifact_io import atomic_json_write,atomic_torch_save_new
from lite_online_dataset import OnlineAnimeDataset
from zenkai_losses import ZenkaiLoss
from zenkai_realesr_lite import ZenkaiRealESRLite
from train_zenkai_realesr import snapshot
from train_zenkai import learning_rate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=10000); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if a.steps<1000: raise ValueError('clean run needs at least 1000 steps')
    data=OnlineAnimeDataset(a.dataset,patch_lr=128,seed=929,codec_fraction=0,codec_cache=None)
    config={'run':a.output.name,'family':'Zenkai','architecture':'SRVGG-inspired PReLU','channels':32,'blocks':6,'nativeScale':2,'steps':a.steps,'seed':929,'batch':4,'accumulation':2,'effectiveBatch':8,'loss':'zenkai_frequency','trainingDataset':str(a.dataset.resolve()),'datasetManifestSha256':data.manifest_sha256,'holdoutManifestSha256':data.holdout_manifest_sha256,'trainPairs':len(data.rows),'releaseApproved':False}
    a.output.mkdir(parents=True); atomic_json_write(config,a.output/'config.json')
    torch.set_num_threads(2); torch.manual_seed(929); random.seed(929); np.random.seed(929)
    model=ZenkaiRealESRLite().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5); scaler=torch.amp.GradScaler('cuda'); criterion=ZenkaiLoss('zenkai_frequency').cuda()
    sampler=RandomSampler(data,replacement=True,num_samples=a.steps*8,generator=torch.Generator().manual_seed(929)); indices=list(sampler); batches=iter(DataLoader(data,batch_size=4,sampler=indices,num_workers=0)); history=[]; start=time.perf_counter(); log=(a.output/'training.log').open('w')
    try:
        for step in range(1,a.steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
            for _ in range(2):
                low,target=next(batches); low=low.cuda(); target=target.cuda()
                with torch.autocast('cuda',dtype=torch.float16): pred=model(low); loss,_=criterion(pred,target)
                scaler.scale(loss/2).backward(); losses.append(float(loss.detach()))
            scaler.unscale_(optimizer); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(optimizer); scaler.update()
            lr=learning_rate(step,a.steps)
            for group in optimizer.param_groups: group['lr']=lr
            if step%100==0: print(json.dumps({'step':step,'loss':float(np.mean(losses)),'gradientNorm':float(grad),'lr':lr}),file=log,flush=True); print(json.dumps({'step':step,'loss':float(np.mean(losses)),'gradientNorm':float(grad),'lr':lr}),flush=True)
            if step%1000==0 or step==a.steps:
                state=snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices); atomic_torch_save_new(state,a.output/f'step-{step}-training-state.pt'); atomic_torch_save_new({'model':state['model'],'config':config,'step':step},a.output/f'step-{step}-master-fp32.pt'); history.append({'step':step,'trainingLoss':float(np.mean(losses))}); atomic_json_write({**config,'stepsExecuted':step,'trainingSeconds':time.perf_counter()-start,'history':history},a.output/'report.json')
    finally: log.close()
    print(json.dumps({'finished':True,'stepsExecuted':a.steps,'output':str(a.output)}))
if __name__=='__main__': main()
