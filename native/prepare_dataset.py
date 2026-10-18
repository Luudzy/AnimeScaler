"""Bounded, resumable download of safe-rated static training images from Parquet."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_IMAGE_BYTES = 12 * 1024 * 1024


def eligible(row):
    url = row.get('file_url') or ''
    parsed = urlparse(url)
    return (row.get('rating') in ('s', 'g') and row.get('file_ext') in ('jpg','jpeg','png','webp')
            and min(row.get('image_width') or 0, row.get('image_height') or 0) >= 720
            and max(row.get('image_width') or 0, row.get('image_height') or 0) <= 10000
            and parsed.scheme == 'https' and parsed.hostname == 'cdn.donmai.us'
            and not parsed.username and not parsed.password and parsed.port in (None,443))


def split_for(digest):
    bucket = int(digest[:8],16) % 100
    return 'test' if bucket < 10 else 'validation' if bucket < 20 else 'train'


def main():
    import pyarrow.parquet as pq
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('F:/treino'))
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'training-data/v2')
    parser.add_argument('--count',type=int,default=256)
    parser.add_argument('--max-mb',type=int,default=512)
    parser.add_argument('--max-attempts',type=int,default=1024)
    parser.add_argument('--exclude-manifest',type=Path)
    parser.add_argument('--split',choices=['train','validation','test','final-hidden'])
    args=parser.parse_args()
    if min(args.count,args.max_mb,args.max_attempts)<=0: parser.error('Positive limits required')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'manifest.jsonl'
    records=[]
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            path=args.output/row['file']
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']:
                records.append(row)
    seen={row['sha256'] for row in records}
    ids={row['id'] for row in records}
    excluded=set()
    if args.exclude_manifest:
        for line in args.exclude_manifest.read_text().splitlines():
            record=json.loads(line);ids.add(record['id']);excluded.add(record['sha256'])
    total=sum((args.output/row['file']).stat().st_size for row in records)
    # Rewrite verified records only, atomically; an interrupted file cannot be trusted.
    temp=manifest.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    os.replace(temp,manifest)
    attempts=0
    columns=['id','rating','file_ext','file_url','image_width','image_height']
    with manifest.open('a',encoding='utf-8') as out:
        for shard in sorted(args.source.glob('*.parquet')):
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=2048,columns=columns):
                for row in batch.to_pylist():
                    if len(seen)>=args.count or attempts>=args.max_attempts or total>=args.max_mb*1024*1024:
                        print(json.dumps({'images':len(seen),'bytes':total,'attempts':attempts}),flush=True); return
                    if row['id'] in ids or not eligible(row): continue
                    attempts+=1
                    try:
                        request=Request(row['file_url'],headers={'User-Agent':'AnimeScaler-local-training/1.0'})
                        with urlopen(request,timeout=20) as response:
                            if urlparse(response.url).hostname!='cdn.donmai.us': raise ValueError('Unexpected redirect')
                            data=response.read(min(MAX_IMAGE_BYTES,args.max_mb*1024*1024-total)+1)
                        if len(data)>MAX_IMAGE_BYTES or total+len(data)>args.max_mb*1024*1024: continue
                        digest=hashlib.sha256(data).hexdigest()
                        if digest in seen or digest in excluded: continue
                        with Image.open(io.BytesIO(data)) as image:
                            if image.width*image.height>40_000_000 or getattr(image,'n_frames',1)!=1: continue
                            image.verify()
                        name=f'{digest}.{row["file_ext"]}'
                        path=args.output/name
                        temporary=path.with_suffix('.download'); temporary.write_bytes(data); os.replace(temporary,path)
                        record={**row,'file':name,'sha256':digest,'split':args.split or split_for(digest)}
                        out.write(json.dumps(record)+'\n');out.flush()
                        seen.add(digest);ids.add(row['id']);total+=len(data)
                        if len(seen)%16==0: print(f'Images: {len(seen)}, MiB: {total/1048576:.1f}',flush=True)
                    except Exception as error:
                        print(f'Row {row["id"]}: {type(error).__name__}',flush=True)
                    time.sleep(.15)
    print(json.dumps({'images':len(seen),'bytes':total,'attempts':attempts}),flush=True)


if __name__=='__main__': main()
