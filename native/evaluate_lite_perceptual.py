"""LPIPS-Squeeze and MS-SSIM on fixed validation crops, CPU only."""
import sys,os,json,argparse,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,str(ROOT/'native/.tools'))
os.environ['TORCH_HOME']=str(ROOT/'diagnostics/native-cache/torch')
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import lpips
from pytorch_msssim import ms_ssim
from lite_model import AdoreLite
from lite_v3 import AdoreLiteV3
from lite_evaluation import load_patches,tensor,validate_codec_cache
from lite_degradations import degrade

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    metric=lpips.LPIPS(net='squeeze').eval()
    saved=torch.load(args.candidate/'best_composite-master-fp32.pt',map_location='cpu',weights_only=True);c=saved['config']
    v3=AdoreLiteV3(c['channels'],c['blocks'],c['residualScale']).eval();v3.load_state_dict(saved['model'])
    v2=AdoreLite().eval()
    with np.load(ROOT/'models/adore-lite-experimental.npz',allow_pickle=False) as w:v2.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
    dataset=Path(c.get('datasetPath',ROOT/'native/training-data/v3'))
    report={};patches=load_patches(dataset=dataset);cache=dataset/'codec-validation';validate_codec_cache(cache,patches)
    rng=random.Random(283)
    with torch.inference_mode():
        for case in ['clean','heavy','codec']:
            scores={name:[] for name in ['bilinear','bicubic','v2','v3']}
            for index,patch in enumerate(patches):
                high=Image.fromarray(patch);low=Image.open(cache/f'{index:04d}.png').convert('RGB') if case=='codec' else degrade(high,rng,case)
                target=tensor(high,'cpu');x=tensor(low,'cpu')
                values={'bilinear':F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False),
                        'bicubic':tensor(low.resize(high.size,Image.Resampling.BICUBIC),'cpu'),'v2':v2(x).clamp(0,1),'v3':v3(x).clamp(0,1)}
                for name,value in values.items():scores[name].append([float(metric(value*2-1,target*2-1)),float(ms_ssim(value,target,data_range=1,size_average=True))])
            report[case]={name:dict(zip(['lpipsSqueeze','msSsim'],np.mean(values,axis=0).tolist())) for name,values in scores.items()}
            print(json.dumps({case:report[case]}),flush=True)
    (args.candidate/'perceptual.json').write_text(json.dumps({'scope':'Validation only; LPIPS-Squeeze (not AlexNet) and standard MS-SSIM, no training selection.',**report},indent=2))
if __name__=='__main__':main()
