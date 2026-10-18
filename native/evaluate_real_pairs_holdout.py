"""Evaluate a real-pair checkpoint only on held-out frame indices."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'native'))
from model_registry import build_registry
from zenkai_realesr_lite import ZenkaiRealESRLite

def tensor(path):
    with Image.open(path) as im: a=np.asarray(im.convert('RGB'),dtype=np.float32).copy()
    return torch.from_numpy(a).permute(2,0,1)[None]/255

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--holdout',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model=ZenkaiRealESRLite().eval(); model.load_state_dict(payload['model'],strict=True)
    rows=[json.loads(s) for s in a.holdout.read_text().splitlines() if s]; groups=defaultdict(list)
    for row in rows: groups[row['source']].append(row)
    reg=build_registry(ROOT); v3=reg.load('adore-lite-v3',device='cpu',precision='fp32')
    result={'state':'complete','pairs':len(rows),'models':{},'sequences':[]}
    try:
        with torch.inference_mode():
            for source, seq in sorted(groups.items()):
                seq=sorted(seq,key=lambda r:int(Path(r['file']).stem)); scores=defaultdict(list); prev={}
                for row in seq:
                    x=tensor(row['pairedLr']); y=tensor(row['file']); values={'bilinear':F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False),'adore-lite-v3':v3.infer(x),'real-pairs-candidate':model(x)}
                    for name,val in values.items():
                        scores[name].append(float((val.float()-y).abs().mean()))
                        if name in prev: scores[name+'-temporal'].append(float(((val-prev[name])-(y-prev['target'])).abs().mean()))
                        prev[name]=val
                    prev['target']=y
                result['sequences'].append({'source':source,'frames':len(seq),'mae':{k:float(np.mean(v)) for k,v in scores.items() if not k.endswith('-temporal')},'temporalMae':{k[:-9]:float(np.mean(v)) for k,v in scores.items() if k.endswith('-temporal')}})
    finally: reg.unload('adore-lite-v3')
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({'state':'complete','output':str(a.output)}))
if __name__=='__main__': main()
