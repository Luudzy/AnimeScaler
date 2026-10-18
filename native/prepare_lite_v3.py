"""Cluster before splitting; preserve historical exposure and reserve new images."""
import hashlib,json,argparse
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def fingerprints(image):
    gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    x=np.arange(32);k=np.arange(8)[:,None]
    dct=np.cos(np.pi*(x+.5)*k/32);coeff=dct@gray@dct.T
    bits=coeff.flatten()[1:]>np.median(coeff.flatten()[1:])
    small=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    dh=small[:,1:]>small[:,:-1]
    pack=lambda values:sum(int(v)<<i for i,v in enumerate(values.flatten()))
    return pack(bits),pack(dh)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'native/training-data/v3')
    parser.add_argument('--anime-source',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[];parents=[];hashes=[]
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for folder in ['v2','v3-final']:
        base=ROOT/'native/training-data'/folder
        if not (base/'manifest.jsonl').exists():continue
        for line in (base/'manifest.jsonl').read_text().splitlines():
            row=json.loads(line);path=base/row['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Checksum mismatch')
            with Image.open(path) as image:p,d=fingerprints(image)
            i=len(records);parents.append(i)
            for j,(pj,dj) in enumerate(hashes):
                if row['sha256']==records[j]['sha256'] or ((p^pj).bit_count()<=6 and (d^dj).bit_count()<=8):parents[find(i)]=find(j)
            hashes.append((p,d));records.append({**row,'file':str(path),'pHash':hex(p),'dHash':hex(d),'historicalSplit':row['split']})
    if args.anime_source:
        # Adjacent numbered crops can originate from one frame/scene. Keep
        # conservative windows together, then merge perceptual near-duplicates.
        windows={}
        for path in sorted(args.anime_source.glob('*.png')):
            data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
            with Image.open(path) as image:p,d=fingerprints(image)
            i=len(records);parents.append(i)
            window=(int(path.stem)-1)//64 if path.stem.isdigit() else path.stem
            if window in windows:parents[find(i)]=find(windows[window])
            else:windows[window]=i
            for j,(pj,dj) in enumerate(hashes):
                if digest==records[j]['sha256'] or ((p^pj).bit_count()<=6 and (d^dj).bit_count()<=8):parents[find(i)]=find(j)
            hashes.append((p,d));records.append({'file':str(path),'sha256':digest,'pHash':hex(p),'dHash':hex(d),
                'historicalSplit':'new-anime','domain':'anime','sourceWindow':str(window)})
    groups={}
    for i,row in enumerate(records):groups.setdefault(find(i),[]).append(row)
    counts={};conflicts=0
    # Former test data remain development-test. Never call them independent.
    priority=['train','test','validation','final-hidden']
    for cluster,rows in groups.items():
        splits={r['historicalSplit'] for r in rows};conflicts+=len(splits)>1
        split=next((s for s in priority if s in splits),None)
        if split is None:
            bucket=int(hashlib.sha256(min(r['sha256'] for r in rows).encode()).hexdigest()[:8],16)%100
            split='train' if bucket<70 else 'validation' if bucket<85 else 'test'
        for row in rows:row.update(split=split,cluster=f'cluster-{cluster:04d}');counts[split]=counts.get(split,0)+1
    manifest=args.output/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    report={'version':'anime-v3-hfa2k' if args.anime_source else 'anime-v3-original-crops','images':len(records),'clusters':len(groups),'groupedMembers':len(records)-len(groups),
            'crossSplitClustersReassigned':conflicts,'splits':counts,'manifestSha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'source':str(args.anime_source) if args.anime_source else None,
            'domainSplits':{domain:{split:sum(r.get('domain','illustration')==domain and r['split']==split for r in records) for split in counts} for domain in ['anime','illustration']},
            'limitations':'pHash+dHash and adjacent 64-file windows are conservative heuristics, not verified episode/scene IDs. Existing test is development-test. No known temporal order.'}
    (args.output/'dataset.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
