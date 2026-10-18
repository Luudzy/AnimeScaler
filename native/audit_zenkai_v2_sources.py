"""Metadata-only future V2 selection; never opens held-out image pixels or changes a run."""
import collections
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'native/training-data/zenkai-v2-audit'
    if folder.exists():raise FileExistsError('Preserve previous source audit')
    inputs={'v2':ROOT/'native/training-data/v2/manifest.jsonl','v31':ROOT/'native/training-data/v31/manifest.jsonl',
            'v3hfa':ROOT/'native/training-data/v3-hfa2k/manifest.jsonl','current':ROOT/'native/training-data/zenkai/manifest.jsonl'}
    rows={name:[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] for name,path in inputs.items()}
    protected={r['sha256'] for rs in rows.values() for r in rs if r['split']!='train'}
    v31={r['sha256']:r for r in rows['v31']};current={r['sha256'] for r in rows['current']}
    protected_groups={r['cluster'] for r in rows['v31'] if r['split']!='train'}
    eligible=[];reasons=collections.Counter();seen=set()
    for r in rows['v2']:
        prior=v31.get(r['sha256'])
        if r['sha256'] in seen:reasons['duplicate manifest hash']+=1;continue
        seen.add(r['sha256'])
        if r['sha256'] in protected:reasons['held out by at least one frozen historical/current manifest']+=1;continue
        if r['sha256'] in current:reasons['already current source']+=1;continue
        if prior is None or prior['split']!='train' or prior['cluster'] in protected_groups:reasons['not confirmed historical train-only group']+=1;continue
        path=(ROOT/'native/training-data/v2'/r['file']).resolve()
        if not path.is_relative_to(ROOT/'native/training-data/v2') or not path.is_file():reasons['missing or invalid local path']+=1;continue
        eligible.append({'file':str(path),'sha256':r['sha256'],'domain':'illustration','split':'train',
            'cluster':prior['cluster'],'pHash':prior['pHash'],'dHash':prior['dHash'],'historicalSplitV2':r['split'],
            'historicalSplitV31':prior['split'],'sourceDataset':'user-authorized-local-v2',
            'contentVerification':'NOT EXECUTED: metadata-only audit'})
    physical=list((ROOT/'native/training-data/v2').glob('*.jpg'))
    known={r['file'] for r in rows['v2']}
    unknown=[p.name for p in physical if p.name not in known]
    report={'state':'METADATA AUDIT COMPLETE','authorizedSource':'native/training-data/v2',
        'inputs':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'records':len(rows[name]),'splits':dict(collections.Counter(r['split'] for r in rows[name]))} for name,path in inputs.items()},
        'eligibleTrainOnly':len(eligible),'excludedReasons':dict(reasons),'physicalJpgCount':len(physical),
        'unlistedFilesQuarantined':unknown,'currentExactHashOverlap':len(current&seen),
        'finalHiddenPixelsOpened':False,'imagePixelsOpened':False,'newRunStarted':False,
        'scope':'Future dataset-only ablation; does not mutate frozen TRAIN/current manifests or any active run',
        'limitations':'Declared hashes and existing heuristic groups only. Content corruption/near-duplicate review not executed. Unlisted files not read or admitted.'}
    folder.mkdir();(folder/'eligible-train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in eligible),encoding='utf-8')
    report['eligibleManifestSha256']=hashlib.sha256((folder/'eligible-train.jsonl').read_bytes()).hexdigest()
    (folder/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'eligibleTrainOnly':len(eligible),'excludedReasons':dict(reasons),'unlistedQuarantined':len(unknown)}))


if __name__=='__main__':main()
