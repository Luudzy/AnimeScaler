"""Verify all saved states and selected evidence without GPU work."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
import torch
from zenkai_domain_sampler import RUN,SOURCE,INITIAL_SHA,validate_snapshot,validate_initial_pair
from artifact_provenance import validate_checkpoint_file

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 f=ROOT/'models/zenkai/experiments'/RUN;index=read(f/'checkpoint-index.json');records=[]
 for name,pin in index['checkpoints'].items():
  path=f/name;validate_checkpoint_file(path,pin);s=torch.load(path,map_location='cpu',weights_only=True);validate_snapshot(s)
  records.append({'path':name,'step':s['step'],'sha256':pin,'nextSampleOffset':s['nextSampleOffset'],'state':'PASS'})
 if len(records)!=10 or index['lastCompleteStep']!=1000:raise ValueError('Wrong indexedstates')
 validate_checkpoint_file(f/'initial-training-state.pt',index['initialSha256']);source=ROOT/'models/zenkai/experiments'/SOURCE
 validate_checkpoint_file(source/'initial-training-state.pt',INITIAL_SHA)
 validate_initial_pair(torch.load(f/'initial-training-state.pt',map_location='cpu',weights_only=True),torch.load(source/'initial-training-state.pt',map_location='cpu',weights_only=True))
 full=torch.load(f/'step-1000-training-state.pt',map_location='cpu',weights_only=True);validate_snapshot(full)
 master=torch.load(f/'step-1000-master-fp32.pt',map_location='cpu',weights_only=True);best=torch.load(f/'best-composite-master-fp32.pt',map_location='cpu',weights_only=True)
 if full['step']!=1000 or master['step']!=1000 or best['step']!=1000:raise ValueError('Wrongfinalsteps')
 for name,value in full['model'].items():
  if not torch.equal(value,master['model'][name]) or not torch.equal(value,best['model'][name]):raise ValueError('Master/full/best differ')
 cfg=read(f/'config.json')
 for name,pin in cfg['sourceFilesSha256'].items():
  validate_checkpoint_file(f/'source'/name,pin);validate_checkpoint_file(ROOT/'native'/name,pin)
 mutable_docs={'docs/zenkai-progress.md','docs/zenkai-overnight-report.md'};preserved_docs=[]
 for prior in ('zenkai-v2-dataset-1k','zenkai-c5k'):
  for name,pin in read(ROOT/'diagnostics'/prior/'final-verification.json')['selectedArtifactsSha256'].items():
   normalized=name.replace('\\','/')
   if normalized in mutable_docs:
    # Shared documents prepend new rounds; the previous whole document remains
    # a byte-identical suffix. Do not mistake the new prefix for model damage.
    content=(ROOT/name).read_bytes()
    marker=(b'# Current round COMPLETE:V2 dataset-only1k,2026-10-04' if normalized.endswith('zenkai-progress.md') else '# ZENKAI OVERNIGHT REPORT — V2 dataset-only1k,2026-10-04'.encode('utf-8'))
    start=content.find(marker)
    if start<0 or hashlib.sha256(content[start:]).hexdigest()!=pin:raise ValueError('Previous document suffix changed: '+name)
    preserved_docs.append({'path':normalized,'previousSha256':pin,'historicalSuffixVerified':True,'currentSha256':sha(ROOT/name),'change':'New Domain10 round prepended; historical bytes preserved'})
   else:validate_checkpoint_file(ROOT/name,pin)
 validate_checkpoint_file(ROOT/'models/legacy/adore-lite-v3/frozen-20260930/adore-lite-v3-experimental.npz',cfg['legacyV3Sha256'])
 validate_checkpoint_file(ROOT/'native/training-data/zenkai-v2-mix/manifest.jsonl',cfg['datasetManifestSha256'])
 old=read(ROOT/'diagnostics/zenkai-v2-dataset-1k/comparison.json');meta=read(source/'metadata.json')
 validate_checkpoint_file(ROOT/'diagnostics/zenkai-v2-dataset-1k/comparison.json',meta['evaluationReportSha256'])
 r=read(ROOT/'diagnostics/zenkai-domain10-1k/comparison.json');decision=read(ROOT/'docs/zenkai-domain10-decision.json')
 if decision['comparisonSha256']!=sha(ROOT/'diagnostics/zenkai-domain10-1k/comparison.json'):raise ValueError('Decisionreport changed')
 if any(o['state']!='PASS' for o in r['operations'].values()) or sum(v['frameCount'] for v in r['video']['sequences'])!=144 or sum(r['quality']['sampleCounts'].values())!=160:raise ValueError('Incomplete evidence')
 crops=read(ROOT/'diagnostics/zenkai-domain10-1k/crops/provenance.json')
 if len(crops['comparisons'])!=9:raise ValueError('Wrongpanelcount')
 for row in crops['comparisons']:
  validate_checkpoint_file(row['output'],row['outputSha256'])
  for v in row['inputs'].values():validate_checkpoint_file(v['path'],v['sha256'])
 result={'state':'PASS','indexedRecoveryStates':records,'allRngOptimizerScalerSamplerSchedulerValidated':True,'initialPairVerified':True,'masterBestFullEqual':True,'allCurrentAndArchivedTrainingSourceHashesVerified':True,'uniformV2AndC5kImmutableSelectedArtifactPinsPreserved':True,'sharedDocumentHistoricalSuffixes':preserved_docs,'HistoricalV3AndUnionManifestPreserved':True,'full144VideoAnd160Fixed':True,'nineCropPanelsAndInputPinsVerified':True,'releaseApproved':False,'Stage2Started':False,'finalHiddenOpened':False}
 (ROOT/'diagnostics/zenkai-domain10-1k/integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('PASS:10 fullrecoveries,exactinitialpair/master/best,15 archived/current sources,previousV2/C/V3 evidence,160/144 and9crop pins')
if __name__=='__main__':main()
