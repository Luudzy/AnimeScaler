"""Continue the real-pair Zenkai run with its complete optimizer/RNG state."""
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
import sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(HERE))
from artifact_io import atomic_json_write,atomic_torch_save_new
from lite_online_dataset import OnlineAnimeDataset
from zenkai_losses import ZenkaiLoss
from zenkai_realesr_lite import ZenkaiRealESRLite
from train_zenkai_realesr import snapshot

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-run',type=Path,required=True); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--total-steps',type=int,default=3000); p.add_argument('--output',type=Path,required=True); p.add_argument('--augment-paired',action='store_true'); p.add_argument('--degradation-profile',choices=['legacy','anime-real','anime-balanced'],default='legacy'); p.add_argument('--loss',choices=['zenkai_frequency','zenkai_frequency_edge','zenkai_animevideo'],default='zenkai_frequency'); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--accumulation',type=int,default=2); a=p.parse_args()
    if a.total_steps<=500 or a.output.exists(): raise ValueError('new total and output required')
    source=a.source_run.resolve(); state_path=max(source.glob('step-*-training-state.pt'), key=lambda p:int(p.stem.split('-')[1])); state=torch.load(state_path,map_location='cpu',weights_only=False); start_step=int(state['step'])+1
    config=dict(state['config']); config['run']=a.output.name; config['steps']=a.total_steps; config['continuationOf']=str(source); config['resumeStep']=start_step-1; config['trainingDataset']=str(a.dataset.resolve())
    data=OnlineAnimeDataset(a.dataset,patch_lr=128,seed=929,codec_fraction=0,codec_cache=None,degradation_profile=a.degradation_profile,augment_paired=a.augment_paired)
    a.output.mkdir(parents=True)
    torch.set_num_threads(2); torch.manual_seed(929); random.seed(929); np.random.seed(929)
    model=ZenkaiRealESRLite().cuda(); model.load_state_dict(state['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.99),weight_decay=1e-5); optimizer.load_state_dict(state['optimizer'])
    scaler=torch.amp.GradScaler('cuda'); scaler.load_state_dict(state['scaler'])
    config['datasetManifestSha256']=data.manifest_sha256; config['holdoutManifestSha256']=data.holdout_manifest_sha256
    config['augmentPaired']=a.augment_paired; config['degradationProfile']=a.degradation_profile; config['loss']=a.loss; config['batch']=a.batch_size; config['accumulation']=a.accumulation; config['effectiveBatch']=a.batch_size*a.accumulation
    atomic_json_write(config,a.output/'config.json')
    data.rng.setstate(state['datasetRng']); indices=state['samplerIndices'].tolist(); batches=iter(DataLoader(data,batch_size=a.batch_size,sampler=indices,num_workers=0)); criterion=ZenkaiLoss(a.loss).cuda()
    history=[]; started=time.perf_counter(); log=(a.output/'training.log').open('w',encoding='utf-8')
    try:
        for step in range(start_step,a.total_steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
            for _ in range(a.accumulation):
                try: low,target=next(batches)
                except StopIteration: batches=iter(DataLoader(data,batch_size=4,sampler=indices,num_workers=0)); low,target=next(batches)
                low=low.cuda(); target=target.cuda()
                with torch.autocast('cuda',dtype=torch.float16): pred=model(low); loss,_=criterion(pred,target)
                scaler.scale(loss/a.accumulation).backward(); losses.append(float(loss.detach()))
            scaler.unscale_(optimizer); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(optimizer); scaler.update()
            if step%100==0: print(json.dumps({'step':step,'loss':float(np.mean(losses)),'gradientNorm':float(grad)}),file=log,flush=True); print(json.dumps({'step':step,'loss':float(np.mean(losses)),'gradientNorm':float(grad)}),flush=True)
            if step%500==0 or step==a.total_steps:
                current=snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices); atomic_torch_save_new(current,a.output/f'step-{step}-training-state.pt'); atomic_torch_save_new({'model':current['model'],'config':config,'step':step},a.output/f'step-{step}-master-fp32.pt'); history.append({'step':step,'trainingLoss':float(np.mean(losses))}); atomic_json_write({**config,'stepsExecuted':step,'history':history,'releaseApproved':False},a.output/'report.json')
    finally: log.close()
    print(json.dumps({'finished':True,'stepsExecuted':a.total_steps,'output':str(a.output)}))
if __name__=='__main__': main()
