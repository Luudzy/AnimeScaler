import hashlib,json,sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'native'))
import torch
from zenkai_v2_dataset_protocol import validate_snapshot,validate_initial_pair
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
f=root/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k';idx=read(f/'checkpoint-index.json')
records=[]
for name,pin in idx['checkpoints'].items():
 assert sha(f/name)==pin
 s=torch.load(f/name,map_location='cpu',weights_only=True);validate_snapshot(s)
 records.append({'path':name,'sha256':pin,'step':s['step'],'nextSampleOffset':s['nextSampleOffset'],'state':'PASS'})
a=torch.load(f/'initial-training-state.pt',map_location='cpu',weights_only=True)
b=torch.load(root/'models/zenkai/experiments/zenkai-ablation-spab-identity-skip-1k/initial-training-state.pt',map_location='cpu',weights_only=True)
validate_initial_pair(a,b)
assert sha(f/'initial-training-state.pt')==idx['initialSha256']
final=torch.load(f/'step-1000-training-state.pt',map_location='cpu',weights_only=True);validate_snapshot(final)
master=torch.load(f/'step-1000-master-fp32.pt',map_location='cpu',weights_only=True)
assert final['step']==master['step']==1000
assert all(torch.equal(v,master['model'][k]) for k,v in final['model'].items())
best=torch.load(f/'best-composite-master-fp32.pt',map_location='cpu',weights_only=True)
assert best['step']==1000 and all(torch.equal(v,best['model'][k]) for k,v in master['model'].items())
cfg=read(f/'config.json')
for n,pin in cfg['sourceFilesSha256'].items():assert sha(f/'source'/n)==pin
for n,pin in read(root/'diagnostics/zenkai-c5k/final-verification.json')['selectedArtifactsSha256'].items():assert sha(root/n)==pin
assert sha(root/'models/legacy/adore-lite-v3/frozen-20260930/adore-lite-v3-experimental.npz')==cfg['legacyV3Sha256']
assert sha(root/'native/training-data/zenkai-v2-mix/manifest.jsonl')==cfg['datasetManifestSha256']
assert sha(root/'native/training-data/zenkai/manifest.jsonl')==cfg['sourceDatasetManifestSha256']
r={'state':'PASS','indexedRecoveryStates':records,'initialPair':True,'allArchivedSourceHashes':True,'masterEqualsFinalState':True,'bestEqualsFinalMaster':True,'previousC5kPinnedArtifactsPreserved':True,'HistoricalV3Preserved':True,'frozenOriginalAndMixManifestHashesPreserved':True,'finalStep':1000,'releaseApproved':False,'finalHiddenOpened':False}
(root/'diagnostics/zenkai-v2-dataset-1k/integrity.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print('PASS:10 indexed full recoveries,real initial pair,FP32 master equality,all archived sources,C5k pins,V3/manifests preserved')
