"""New domain sampling10%1k from real uniform-union initialization; immutable math and recoveries."""
import argparse
import collections
import copy
import json
import os
from pathlib import Path
import shutil
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/'.tools'))
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler
from hip_operation import save_json
from artifact_io import atomic_copy_file,atomic_torch_save,atomic_torch_save_new,publish_training_checkpoint
from train_zenkai import ROOT,_sha,learning_rate,update_composite,_LegacyAdapter
from zenkai_experiments import state_snapshot,require_codec_cache
from zenkai_smoke_resume import restore_training_state
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_domain_sampler import SOURCE,RUN,MIX_ROOT,MIX_SHA,build_config,validate_snapshot,validate_initial_pair,sample_indices
from zenkai_v2_dataset_protocol import BASE_SHA
from zenkai_recipe_loss import make_experiment_loss
from lite_online_dataset import OnlineAnimeDataset
from zenkai_evaluation import ZenkaiFixedEvaluator


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_data(source):
    base=ROOT/'native/training-data/zenkai/manifest.jsonl';mix=MIX_ROOT/'manifest.jsonl'
    if _sha(base)!=BASE_SHA or _sha(mix)!=MIX_SHA or source['datasetManifestSha256']!=MIX_SHA:
        raise ValueError('Frozen dataset identity changed')
    rows=lambda path:[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    old,new=rows(base),rows(mix)
    if new[:len(old)]!=old or len(new)!=len(old)+602 or any(r['split']!='train' for r in new[len(old):]):
        raise ValueError('V2 union changes original holdouts or wrong addition count')
    audit=ROOT/'native/training-data/zenkai-v2-audit/pixel-verification.json';verified=read(audit)
    if verified['state']!='PASS' or verified['decodedTrainSources']!=602 or verified['decodedTrainCodecPairs']!=602 or verified['heldOutPixelsOpened']:
        raise ValueError('TRAIN content verification incomplete')
    expected={r['sourceSha256'] for r in verified['records']}
    if {r['sha256'] for r in new[len(old):]}!=expected:raise ValueError('Union includes unverified identity')
    return {'v2PreparationSha256':_sha(MIX_ROOT/'preparation.json'),'v2PixelVerificationSha256':_sha(audit)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--attempt',type=int,choices=(1,2,3),required=True);args=p.parse_args()
    origin=ROOT/'models/zenkai/experiments'/SOURCE;folder=ROOT/'models/zenkai/experiments'/RUN
    initial_path=origin/'initial-training-state.pt';pins=read(origin/'metadata.json')['checkpointsSha256']
    if _sha(initial_path)!=pins[initial_path.name]:raise ValueError('Original initial state changed')
    original=torch.load(initial_path,map_location='cpu',weights_only=True);source=original['config'];path=args.checkpoint.resolve()
    if path==initial_path:digest=pins[initial_path.name]
    elif path.is_relative_to(folder/'attempts'):
        digest=read(folder/'checkpoint-index.json')['checkpoints'].get(str(path.relative_to(folder)))
    elif path==folder/'initial-training-state.pt':digest=read(folder/'checkpoint-index.json')['initialSha256']
    else:raise ValueError('Unexpected source/recovery')
    if not digest or _sha(path)!=digest:raise ValueError('Full-state SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True);validate_snapshot(saved)
    if not torch.version.hip or not torch.cuda.is_available() or str(torch.__version__)!=source['torch'] or torch.version.hip!=source['hip']:
        raise RuntimeError('Actual ROCm training stack changed')
    for name,pin in source['sourceFilesSha256'].items():
        if name!='zenkai_v2_dataset_protocol.py' and _sha(Path(__file__).with_name(name))!=pin:raise ValueError('Original math dependency changed:'+name)
    if _sha(Path(__file__).with_name('zenkai_ablation_skip.py'))!=source['sourceFilesSha256']['zenkai_ablation_skip.py']:raise ValueError('Original skip forward changed')
    hashes={name:_sha(Path(__file__).with_name(name)) for name in
        (*source['sourceFilesSha256'],'zenkai_domain_screen.py','zenkai_domain_sampler.py','zenkai_smoke_resume.py','zenkai_ablation_skip.py','zenkai_recipe_loss.py')}
    config={**build_config(source,pins[initial_path.name],hashes),**verify_data(source)}
    if saved['config']['run']==RUN and saved['config']!=config:raise ValueError('Immutable recovery config/source changed')
    cache=require_codec_cache(config)
    if _sha(cache/'complete.json')!=source['codecCacheCompleteSha256']:raise ValueError('Codec cache identity changed')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    model=IdentitySkipProbe().cuda().float()
    data=OnlineAnimeDataset(MIX_ROOT,patch_lr=128,seed=929,codec_fraction=.25,codec_cache=cache)
    if len(data)!=2394 or collections.Counter(r['domain'] for r in data.rows)!={'anime':1792,'illustration':602}:raise ValueError('Wrong TRAIN union')
    # Verify the sampler algorithm in the actual Torch/HIP stack, not only on CPU preflight.
    baseline=list(RandomSampler(range(2394),replacement=True,num_samples=40000,generator=torch.Generator().manual_seed(929)))
    if not torch.equal(torch.tensor(baseline),original['samplerIndices']):raise ValueError('Original uniform sampler cannot be reproduced')
    indices=sample_indices([r['domain'] for r in data.rows]).tolist()
    if saved['config']['run']==RUN and not torch.equal(torch.tensor(indices),saved['samplerIndices']):raise ValueError('Domain weighted sampler stream changed')
    batches=iter(DataLoader(data,batch_size=4,sampler=indices[saved['nextSampleOffset']:],num_workers=0))
    evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=32)
    if evaluator.sha256!=source['fixedValidationSha256']:raise ValueError('Frozen validation changed')
    os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    from model_registry import build_registry
    registry=build_registry(ROOT);legacy=registry.load('adore-lite-v3',device='cuda',precision='fp16')
    if legacy.artifact['sha256']!=source['legacyV3Sha256']:raise ValueError('Historical V3 changed')
    criterion=make_experiment_loss('zenkai_frequency').cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5);scaler=torch.amp.GradScaler('cuda')
    restore_training_state(model,optimizer,scaler,saved,data)
    if not folder.exists():
        initial=state_snapshot(model,optimizer,scaler,0,config,data.rng.getstate(),indices)
        pairing=validate_initial_pair(initial,original);pairing.update(referenceSha256=pins[initial_path.name],reference=str(initial_path),baselineSamplerReproducedInActualStack=True)
        folder.mkdir();save_json(folder/'config.json',config);save_json(folder/'initial-pairing.json',pairing)
        atomic_torch_save_new(initial,folder/'initial-training-state.pt');archive=folder/'source';archive.mkdir()
        for name in hashes:shutil.copyfile(Path(__file__).with_name(name),archive/name)
        save_json(folder/'checkpoint-index.json',{'checkpoints':{},'lastCompleteStep':0,'initialSha256':_sha(folder/'initial-training-state.pt')})
        print(json.dumps({'initialPairing':'matched except weighted sampler indices','run':RUN,'trainImages':len(data)}),flush=True)
    elif read(folder/'config.json')!=config:raise ValueError('Run configuration changed')
    out=folder/'attempts'/f'attempt-{args.attempt}';out.mkdir(parents=True)
    progress=saved.get('resumeProgress',{'trainingSeconds':0.,'samples':0,'codecSamples':0,'history':[],'best':{}})
    started=time.perf_counter();history=copy.deepcopy(progress['history']);best=copy.deepcopy(progress['best'])
    save_json(out/'resume.json',{'step':saved['step'],'sha256':digest,'source':str(path),'stateRestored':True})
    for step in range(saved['step']+1,1001):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[];lr=learning_rate(step,5000)
        for group in optimizer.param_groups:group['lr']=lr
        for _ in range(2):
            low,target=next(batches);low=low.cuda();target=target.cuda()
            with torch.autocast('cuda',dtype=torch.float16):prediction=model(low)
            loss,_=criterion(prediction,target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            losses.append(float(loss.detach()));scaler.scale(loss/2).backward()
        scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old_scale:raise RuntimeError('AMP update skipped')
        if step==1000:
            model.eval();deploy=model.deploy_copy().cuda().half().eval()
            scores=evaluator.evaluate({'candidate':lambda x:deploy(x.half()).float(),'adore-lite-v3':_LegacyAdapter(legacy)},device='cuda',perceptual=feature)
            summary=update_composite(scores['cases']);best={'step':1000,'composite':summary['composite']}
            history.append({'step':step,'screening':scores,'summary':summary,'trainingLoss':float(np.mean(losses))});del deploy
        if step%100==0:
            state=state_snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices)
            domains=dict(collections.Counter(data.rows[i]['domain'] for i in indices[:step*8]))
            state['resumeProgress']={'trainingSeconds':progress['trainingSeconds']+time.perf_counter()-started,
                'samples':progress['samples']+data.samples,'codecSamples':progress['codecSamples']+data.codec_samples,
                'sampledTrainDomainCounts':domains,'distinctTrainImagesSampled':len(set(indices[:step*8])),
                'history':history,'best':best}
            checkpoint=out/f'step-{step}-training-state.pt'
            index_path=folder/'checkpoint-index.json';index=read(index_path)
            report={**config,**state['resumeProgress'],'stepsExecuted':step,'optimizerUpdates':step,
                'uniqueSpatialCrops':'NOT MEASURED across recovery attempts','promoted':False}
            index=publish_training_checkpoint(state,checkpoint,index_path,index,
                str(checkpoint.relative_to(folder)),step,str(checkpoint),folder/'report.json',report)
            print(json.dumps({'step':step,'loss':float(np.mean(losses)),'lr':lr,'gradientNorm':float(grad),'updateSkipped':False}),flush=True)
            del state
    final_path=Path(read(folder/'checkpoint-index.json')['lastCheckpoint'])
    final=torch.load(final_path,map_location='cpu',weights_only=True);validate_snapshot(final)
    if final['step']!=1000:raise ValueError('Missing complete final state')
    master={'model':final['model'],'config':config,'step':1000}
    atomic_torch_save(master,folder/'step-1000-master-fp32.pt');atomic_torch_save(master,folder/'best-composite-master-fp32.pt')
    atomic_copy_file(final_path,folder/'step-1000-training-state.pt')
    if _sha(path)!=digest:raise ValueError('Source state changed')
    print(json.dumps({'state':'PASS','stepsExecuted':1000,'run':RUN}),flush=True)


if __name__=='__main__':main()
