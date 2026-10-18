"""Controlled temporal test with a known moving HR reference, not real footage."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lite_model import AdoreLite
from lite_v3 import AdoreLiteV3
from lite_evaluation import load_patches,tensor

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(args.candidate/'best_composite-master-fp32.pt',map_location='cpu',weights_only=True)
    config=saved['config'];v3=AdoreLiteV3(config['channels'],config['blocks'],config['residualScale']).cuda().eval();v3.load_state_dict(saved['model'])
    v2=AdoreLite().cuda().eval()
    with np.load(ROOT/'models/adore-lite-experimental.npz',allow_pickle=False) as w:v2.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
    from evaluate_sr_temporal import evaluate_synthetic_temporal
    patches=load_patches(per_image=1,size=224,dataset=Path(config.get('datasetPath',ROOT/'native/training-data/v3')))
    result=evaluate_synthetic_temporal({'v2':v2,'v3':v3},patches)
    (args.candidate/'temporal.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
