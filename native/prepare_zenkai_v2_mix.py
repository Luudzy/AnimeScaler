"""Prepare a separate future dataset-only manifest, keeping every holdout unchanged."""
import collections
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def merge_train_only(base,extra):
    seen={r['sha256'] for r in base};protected={r['cluster'] for r in base if r['split']!='train'}
    result=[dict(r) for r in base]
    for row in extra:
        if row['split']!='train' or row['sha256'] in seen or row['cluster'] in protected:
            raise ValueError('V2 addition overlaps source identity/protected group or is not TRAIN')
        seen.add(row['sha256']);result.append(dict(row))
    return result


def main():
    original=ROOT/'native/training-data/zenkai/manifest.jsonl';add=ROOT/'native/training-data/zenkai-v2-audit/verified-train.jsonl'
    folder=ROOT/'native/training-data/zenkai-v2-mix'
    if folder.exists():raise FileExistsError('Preserve existing mix manifest')
    read=lambda path:[json.loads(line) for line in path.read_text().splitlines() if line]
    base=read(original);extra=read(add);rows=merge_train_only(base,extra)
    cache=ROOT/'native/training-data/v31/codec-full';codec=collections.Counter()
    for row in rows:
        if row['split']!='train':continue
        stem=cache/row['sha256'];meta=json.loads(stem.with_suffix('.json').read_text())
        if meta['sourceSha256']!=row['sha256'] or meta['scale']!=2 or not stem.with_suffix('.png').is_file():raise ValueError('Missing aligned codec pair for future TRAIN identity')
        codec[str(meta.get('codec',meta.get('encoder','UNLABELED')))]+=1
    folder.mkdir();manifest=folder/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report={'state':'PREPARED NOT TRAINED','baseManifestSha256':digest(original),'extraManifestSha256':digest(add),
        'manifestSha256':digest(manifest),'splits':dict(collections.Counter(r['split'] for r in rows)),
        'trainDomains':dict(collections.Counter(r.get('domain','UNLABELED') for r in rows if r['split']=='train')),
        'uniformImageSamplingIllustrationFraction':len(extra)/(len(extra)+sum(r['split']=='train' for r in base)),
        'codecCache':str(cache),'codecCacheCompleteSha256':digest(cache/'complete.json'),'trainCodecMetadataCounts':dict(codec),
        'scope':'Future dataset-only ablation; same frozen validation/holdout identities,602 train-only illustrations added. Cache metadata/existence checked only for TRAIN identities; no held-out pixels opened.',
        'newRunStarted':False,'activeDatasetModified':False}
    (folder/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
