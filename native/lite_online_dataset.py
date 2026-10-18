"""Path-based TRAIN dataset. Codec pairs stay aligned at exactly 2x."""
import hashlib
import json
import math
import os
import random
from array import array
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from lite_degradations import degrade
from lite_degradations_anime_real import degrade as degrade_anime_real, degrade_balanced

def detail_score(image):
    gray = np.asarray(image.convert('L'),dtype=np.float32)
    return float(np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())

def _sha256_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def _read_jsonl(path, content):
    try:
        return [json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Invalid JSONL manifest: {path}') from error

def _manifest_path_key(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError('Train/holdout rows require a file path')
    path = Path(value)
    if not path.is_absolute(): path = root / path
    return os.path.normcase(os.path.normpath(str(path)))

def _manifest_hash(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdefABCDEF' for char in value):
        raise ValueError(f'Train/holdout rows require a valid {field}')
    return value.lower()

def _validate_train_holdout(root, train_rows, train_bytes):
    holdout_path = root / 'holdout.jsonl'
    if not holdout_path.is_file(): return None
    holdout_bytes = holdout_path.read_bytes()
    holdout_rows = _read_jsonl(holdout_path, holdout_bytes)
    metadata_path = root / 'metadata.json'
    if not metadata_path.is_file():
        raise ValueError('Split dataset requires metadata.json with train/holdout hashes')
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid split metadata.json') from error
    if metadata.get('state') != 'complete': raise ValueError('Split dataset metadata is not complete')
    if metadata.get('sourceSha256') != _sha256_bytes(train_bytes): raise ValueError('Training manifest hash differs from metadata')
    if metadata.get('holdoutSha256') != _sha256_bytes(holdout_bytes): raise ValueError('Holdout manifest hash differs from metadata')
    if metadata.get('trainPairs') != len(train_rows) or metadata.get('holdoutPairs') != len(holdout_rows):
        raise ValueError('Train/holdout pair counts differ from metadata')
    if metadata.get('splitStrategy') != 'whole source clips; frames from one clip never cross splits':
        raise ValueError('Split metadata does not certify source-clip separation')

    def identities(rows, expected_split):
        groups=set(); paths=set(); hashes=set()
        for row in rows:
            if not isinstance(row, dict) or row.get('split') != expected_split:
                raise ValueError(f'{expected_split} manifest contains a row with the wrong split label')
            group=row.get('splitGroup')
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f'{expected_split} rows require splitGroup')
            groups.add(group)
            paths.add(_manifest_path_key(root, row.get('file')))
            hashes.add(_manifest_hash(row.get('sha256'), 'sha256'))
            paired=row.get('pairedLr')
            if paired is not None:
                paths.add(_manifest_path_key(root, paired))
                hashes.add(_manifest_hash(row.get('pairedLrSha256'), 'pairedLrSha256'))
        if not groups: raise ValueError(f'{expected_split} split is empty')
        return groups, paths, hashes

    train=identities(train_rows, 'train')
    holdout=identities(holdout_rows, 'holdout')
    if train[0] & holdout[0]: raise ValueError('Train/holdout leakage: splitGroup is shared')
    if train[1] & holdout[1]: raise ValueError('Train/holdout leakage: file path is shared')
    if train[2] & holdout[2]: raise ValueError('Train/holdout leakage: file SHA-256 is shared')
    source_groups=train[0] | holdout[0]
    if metadata.get('sourceGroups') != len(source_groups): raise ValueError('Source group count differs from metadata')
    return _sha256_bytes(holdout_bytes)

class BoundedUniqueCropCounter:
    """Fixed-size HyperLogLog estimate for crop identities."""
    def __init__(self, precision=12):
        if not 4 <= precision <= 16: raise ValueError('HyperLogLog precision must be between 4 and 16')
        self.precision=precision; self.register_count=1<<precision
        self.registers=array('B',[0])*self.register_count
    @property
    def method(self): return f'hyperloglog-p{self.precision}'
    @property
    def memory_bytes(self): return len(self.registers)*self.registers.itemsize
    def report_fields(self,name='uniqueSpatialCrops'):
        return {name:len(self),f'{name}CountType':'estimate',
                f'{name}Method':self.method,f'{name}CounterMemoryBytes':self.memory_bytes}
    def add(self, identity):
        encoded=json.dumps(identity,separators=(',',':'),ensure_ascii=True).encode('utf-8')
        value=int.from_bytes(hashlib.blake2b(encoded,digest_size=8).digest(),'big')
        index=value>>(64-self.precision); remaining=value&((1<<(64-self.precision))-1)
        rank=(64-self.precision)-remaining.bit_length()+1 if remaining else 65-self.precision
        if rank>self.registers[index]: self.registers[index]=rank
    def __len__(self):
        m=self.register_count
        alpha=0.7213/(1+1.079/m)
        estimate=alpha*m*m/sum(2.0**-rank for rank in self.registers)
        zeroes=self.registers.count(0)
        if estimate<=2.5*m and zeroes: estimate=m*math.log(m/zeroes)
        return max(0,round(estimate))

class OnlineAnimeDataset(Dataset):
    def __init__(self, dataset, patch_lr=128, seed=929, hard_fraction=.3, codec_fraction=.25, codec_cache=None, degradation_profile='legacy', augment_paired=False):
        self.root = Path(dataset)
        manifest_path=self.root/'manifest.jsonl'
        manifest_bytes=manifest_path.read_bytes()
        all_rows=_read_jsonl(manifest_path,manifest_bytes)
        self.manifest_sha256=_sha256_bytes(manifest_bytes)
        self.holdout_manifest_sha256=_validate_train_holdout(self.root,all_rows,manifest_bytes)
        self.rows = [row for row in all_rows if isinstance(row,dict) and row.get('split')=='train']
        self.patch_lr=patch_lr; self.rng=random.Random(seed)
        if degradation_profile not in ('legacy','anime-real','anime-balanced'): raise ValueError('Unknown degradation profile')
        self.degradation_profile=degradation_profile
        self.augment_paired=bool(augment_paired)
        self.hard_fraction=hard_fraction; self.codec_fraction=codec_fraction
        self.codec_cache=Path(codec_cache) if codec_cache is not None else self.root/'codec-full'
        self.unique_crops=BoundedUniqueCropCounter(); self.samples=0; self.codec_samples=0; self.hard_samples=0
        if not isinstance(patch_lr,int) or isinstance(patch_lr,bool) or patch_lr<16 or not self.rows:
            raise ValueError('Invalid patch size or empty training set')
        for name,value in (('hard',hard_fraction),('codec',codec_fraction)):
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError(f'Invalid {name} fraction; expected a finite value from 0 to 1')
        self._preflight_rows()
    def _preflight_rows(self):
        required=self.patch_lr*2
        for row in self.rows:
            source=Path(row.get('file',''))
            if not source.is_absolute(): source=self.root/source
            row['file']=str(source)
            expected=row.get('sha256','')
            if len(expected)!=64 or any(char not in '0123456789abcdefABCDEF' for char in expected):
                raise ValueError(f'Invalid source SHA-256: {source}')
            if not source.is_file() or _sha256_file(source).lower()!=expected.lower():
                raise ValueError(f'Source SHA-256 mismatch: {source}')
            with Image.open(source) as image: width,height=image.size
            if min(width,height)<required: raise ValueError(f'Image too small for native crop: {source}')
            fixed_pair=row.get('pairedLr')
            if fixed_pair is not None:
                pair=Path(fixed_pair)
                if not pair.is_absolute(): pair=self.root/pair
                row['pairedLr']=str(pair)
                with Image.open(pair) as image: pair_size=image.size
                if pair_size!=(width//2,height//2): raise ValueError('Paired LR geometry mismatch')
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]; rng=self.rng; size=self.patch_lr*2
        with Image.open(row['file']) as image: high=image.convert('RGB')
        if min(high.size)<size: raise ValueError(f'Image too small for native crop: {row["file"]}')
        fixed_pair = row.get('pairedLr')
        paired = fixed_pair is not None or (self.codec_cache.is_dir() and rng.random()<self.codec_fraction)
        uses_paired_lr=paired and not self.augment_paired
        low_full=None
        if fixed_pair is not None:
            with Image.open(fixed_pair) as image: low_full=image.convert('RGB')
            if low_full.size != (high.width // 2, high.height // 2): raise ValueError('Paired LR geometry mismatch')
        elif paired:
            stem=self.codec_cache/row['sha256']
            metadata=json.loads(stem.with_suffix('.json').read_text())
            if metadata['sourceSha256']!=row['sha256'] or metadata['scale']!=2: raise ValueError('Codec pair identity mismatch')
            with Image.open(stem.with_suffix('.png')) as image: low_full=image.convert('RGB')
            if list(low_full.size)!=metadata['lrSize']: raise ValueError('Codec pair dimensions mismatch')
            if low_full.size!=(high.width//4*2,high.height//4*2): raise ValueError('Codec and HR geometry mismatch')
            high=high.crop((0,0,low_full.width*2,low_full.height*2))
        hard=rng.random()<self.hard_fraction
        candidates=[]
        for _ in range(4 if hard else 1):
            x=2*rng.randrange((high.width-size)//2+1); y=2*rng.randrange((high.height-size)//2+1)
            crop=high.crop((x,y,x+size,y+size))
            candidates.append((detail_score(crop) if hard else 0,x,y,crop))
        _,x,y,target=max(candidates,key=lambda v:v[0])
        if not uses_paired_lr:
            low = degrade_anime_real(target,rng) if self.degradation_profile=='anime-real' else degrade_balanced(target,rng) if self.degradation_profile=='anime-balanced' else degrade(target,rng)
        else:
            low = low_full.crop((x//2,y//2,(x+size)//2,(y+size)//2))
        if rng.random()<.5: target=ImageOps.mirror(target); low=ImageOps.mirror(low)
        if rng.random()<.5: target=ImageOps.flip(target); low=ImageOps.flip(low)
        self.unique_crops.add((row['sha256'],x,y,size)); self.samples+=1
        self.codec_samples+=uses_paired_lr; self.hard_samples+=hard
        convert=lambda image:torch.from_numpy(np.array(image,copy=True).transpose(2,0,1)).float()/255
        return convert(low),convert(target)
