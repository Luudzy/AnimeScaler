"""Create Zenkai-only split metadata; never open held-out image pixels."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def partition_rows(rows,development_target=192,seed=929):
    if development_target<1:raise ValueError('Development target must be positive')
    approved=Path('F:/treino').resolve()
    result=[{**row,'previousSplit':row['split']} for row in rows
            if row.get('domain')=='anime' and Path(row['file']).resolve().is_relative_to(approved)]
    groups={}
    for row in result:
        if not row.get('cluster'):raise ValueError('Missing grouping cluster')
        groups.setdefault(row['cluster'],[]).append(row)
    if any(len({row['split'] for row in group})!=1 for group in groups.values()):
        raise ValueError('Existing cluster crosses splits; resolve metadata before partitioning')
    training=[group for group in groups.values() if group[0]['split']=='train']
    rank=lambda group:hashlib.sha256(f"{seed}:{min(row['sha256'] for row in group)}".encode()).hexdigest()
    reserved=0
    for group in sorted(training,key=rank):
        if reserved>=development_target:break
        for row in group:
            row['split']='development-test'
            row['historicalExposure']='Previously in Adore V3 training; not an independent V3 holdout'
        reserved+=len(group)
    if reserved<development_target or not any(row['split']=='train' for row in result):
        raise ValueError('Insufficient train groups to reserve development-test')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'native/training-data/v31')
    parser.add_argument('--output',type=Path,default=ROOT/'native/training-data/zenkai')
    parser.add_argument('--development-target',type=int,default=192)
    parser.add_argument('--seed',type=int,default=929)
    args=parser.parse_args()
    original=args.source/'manifest.jsonl'
    rows=[json.loads(line) for line in original.read_text(encoding='utf-8').splitlines() if line]
    result=partition_rows(rows,args.development_target,args.seed)
    splits={split:sum(row['split']==split for row in result) for split in ['train','validation','development-test','final-hidden']}
    if not all(splits.values()):raise ValueError('Every Zenkai split must contain anime')
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/'manifest.jsonl'
    data=''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in result)
    with output.open('x',encoding='utf-8') as stream:stream.write(data)
    metadata={'family':'Zenkai','sources':['F:/treino/hfa2k','F:/treino'],
        'sourceManifest':str(original),'sourceManifestSha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'manifestSha256':hashlib.sha256(output.read_bytes()).hexdigest(),'seed':args.seed,
        'images':len(result),'splits':splits,'groups':len({row['cluster'] for row in result}),
        'pixelsRead':False,'realVideoValidation':'NOT EXECUTED',
        'codecCache':str(args.source/'codec-full'),
        'limitations':['Development-test comes from prior Adore V3 training groups; not an independent V3 comparison.',
                      'Final-hidden and validation memberships preserved from v31; final-hidden pixels not inspected.',
                      'Clusters use old near-duplicate and adjacent-file heuristics, not verified episodes/scenes.',
                      'Only image paths physically within F:/treino are included; external illustration caches excluded.']}
    (args.output/'dataset.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
