"""Compare a continued Lite candidate at 2x and 1.5x, including JPEG inputs."""
import io
import argparse
import json
import random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lite_model import AdoreLite
from train_lite_v2 import edges

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--previous',default='lite-backups/pre-12000-20260927/adore-lite-experimental.npz')
    parser.add_argument('--candidate',default='adore-lite-v2-16.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'diagnostics/lite-12000-additional-evaluation.json')
    subset=parser.add_mutually_exclusive_group()
    subset.add_argument('--exclude-manifest',type=Path)
    subset.add_argument('--include-manifest',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    models = {}
    for name, path in [('previous',args.previous),('candidate',args.candidate)]:
        model = AdoreLite().cuda().eval()
        with np.load(ROOT / 'models' / path, allow_pickle=False) as weights:
            model.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files})
        models[name] = model
    dataset = ROOT / 'native/training-data/v2'
    rows = [json.loads(row) for row in (dataset/'manifest.jsonl').read_text().splitlines()]
    subset_path=args.exclude_manifest or args.include_manifest
    if subset_path:
        hashes={json.loads(row)['sha256'] for row in subset_path.read_text().splitlines()}
        rows=[row for row in rows if (row['sha256'] in hashes)==bool(args.include_manifest)]
    test_count=sum(row['split']=='test' for row in rows)
    if not test_count:raise ValueError('No test images selected')
    rng = random.Random(9027)
    scores = {case: {name: [] for name in models} for case in ['2x', '1.5x', '2x-jpeg', '1.5x-jpeg']}
    with torch.inference_mode():
        for row in rows:
            if row['split'] != 'test':
                continue
            with Image.open(dataset/row['file']) as source:
                image = source.convert('RGB')
                image.thumbnail((1024,1024), Image.Resampling.LANCZOS)
                x=rng.randrange(image.width-96+1); y=rng.randrange(image.height-96+1)
                high=image.crop((x,y,x+96,y+96))
            target=torch.from_numpy(np.asarray(high).copy().transpose(2,0,1)).float().cuda()[None]/255
            for case in scores:
                size=64 if case.startswith('1.5') else 48
                low=high.resize((size,size),Image.Resampling.LANCZOS)
                if case.endswith('jpeg'):
                    buffer=io.BytesIO(); low.save(buffer,format='JPEG',quality=80); buffer.seek(0)
                    low=Image.open(buffer).convert('RGB')
                value=torch.from_numpy(np.asarray(low).copy().transpose(2,0,1)).float().cuda()[None]/255
                for name, model in models.items():
                    output=model(value)
                    if size==64:
                        output=F.interpolate(output,size=(96,96),mode='bilinear',align_corners=False)
                    output=output.clamp(0,1)
                    a,b=edges(output),edges(target)
                    scores[case][name].append([F.l1_loss(output,target).item(),
                        ((F.l1_loss(a[0],b[0])+F.l1_loss(a[1],b[1]))/2).item()])
    report={case:{name:dict(zip(['mae','edgeMae'],np.mean(values,axis=0).tolist()))
                  for name,values in values_by_model.items()} for case,values_by_model in scores.items()}
    report['evaluation']=f'{test_count} held-out images, fixed crops; synthetic scaling/JPEG, not video validation'
    report['subset']='new images' if args.exclude_manifest else 'original images' if args.include_manifest else 'all images'
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
