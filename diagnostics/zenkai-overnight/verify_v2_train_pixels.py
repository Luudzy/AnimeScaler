"""Decode only already-admitted TRAIN identities and their aligned codec pairs."""
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT/'native/training-data/zenkai-v2-audit'
    output = folder/'pixel-verification.json'
    if output.exists():
        raise FileExistsError('Preserve prior pixel verification')
    manifest = folder/'verified-train.jsonl'
    if sha(manifest) != 'ecd69554a652d8d857ce1c77f08a841664272cc9f3aa53e1b3d7484b11416b25':
        raise ValueError('Admitted TRAIN identities changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    protected = set()
    for name in ('v2', 'v31', 'v3-hfa2k', 'zenkai'):
        for line in (ROOT/'native/training-data'/name/'manifest.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] != 'train':
                protected.add(row['sha256'])
    cache = ROOT/'native/training-data/v31/codec-full'
    results = []
    for row in rows:
        path = Path(row['file']).resolve()
        if row['split'] != 'train' or row['sha256'] in protected or not path.is_relative_to(ROOT/'native/training-data/v2'):
            raise ValueError('Held-out or foreign content rejected before opening pixels')
        if sha(path) != row['sha256']:
            raise ValueError('Source bytes changed')
        with Image.open(path) as image:
            image.load()
            rgb = image.convert('RGB')
            high_size = rgb.size
        stem = cache/row['sha256']
        meta_path = stem.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        if meta['sourceSha256'] != row['sha256'] or meta['scale'] != 2:
            raise ValueError('Foreign codec pair rejected')
        low_path = stem.with_suffix('.png')
        with Image.open(low_path) as image:
            image.load()
            low_size = image.convert('RGB').size
        if min(high_size) < 256 or list(low_size) != meta['lrSize'] or low_size != (high_size[0]//4*2, high_size[1]//4*2):
            raise ValueError('Training crop/codec geometry mismatch')
        results.append({'sourceSha256': row['sha256'], 'sourceSize': high_size,
            'codecSha256': sha(low_path), 'codecSize': low_size, 'codecMetadataSha256': sha(meta_path)})
    if len(results) != 602:
        raise ValueError('Incomplete admitted TRAIN set')
    output.write_text(json.dumps({'state': 'PASS', 'manifestSha256': sha(manifest),
        'decodedTrainSources': len(results), 'decodedTrainCodecPairs': len(results),
        'heldOutPixelsOpened': False, 'activeDatasetModified': False,
        'scope': 'Full decoding, identities and aligned geometry only; not visual quality/near-duplicate certification',
        'records': results}, indent=2), encoding='utf-8')
    print('PASS:602 TRAIN sources and602 aligned codec pairs decoded; no held-out pixels opened')


if __name__ == '__main__':
    main()
