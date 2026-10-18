"""Prepare complete LR codec pairs once, outside the training loop."""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_codec_cache import roundtrip

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True);args=parser.parse_args()
    rows=[json.loads(s) for s in (args.dataset/'manifest.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']=='train']
    folder=args.dataset/'codec-full';folder.mkdir(exist_ok=True);rng=random.Random(1931)
    for index,row in enumerate(rows):
        stem=folder/row['sha256'];codec=['libx264','libx265','libaom-av1'][index%3];crf=rng.choices([18,24,30],[.3,.5,.2])[0]
        if stem.with_suffix('.json').exists() and stem.with_suffix('.png').exists():continue
        source=Path(row['file'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source changed')
        with Image.open(source) as image:
            w=image.width//4*4;h=image.height//4*4
            low=image.convert('RGB').crop((0,0,w,h)).resize((w//2,h//2),Image.Resampling.LANCZOS)
        decoded=roundtrip([np.array(low)],codec,crf)[0]
        Image.fromarray(decoded).save(stem.with_suffix('.png'))
        stem.with_suffix('.json').write_text(json.dumps({'sourceSha256':row['sha256'],'scale':2,'lrSize':list(low.size),'codec':codec,'crf':crf,'scope':'codec_intra; complete image pair, not video'}))
        if (index+1)%25==0:print(json.dumps({'completed':index+1,'total':len(rows)}),flush=True)
    (folder/'complete.json').write_text(json.dumps({'images':len(rows),'scope':'TRAIN only; aligned complete pairs; no hidden images accessed'}))
if __name__=='__main__':main()
