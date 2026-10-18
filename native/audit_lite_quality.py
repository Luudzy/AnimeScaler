"""Measure V2 against interpolation at identical resolution; never trains."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT
import torch,numpy as np
from torch.nn import functional as F
from PIL import Image,ImageDraw
from lite_model import AdoreLite
from lite_evaluation import ROOT,load_patches,tensor,evaluate
from lite_losses import metrics

torch.set_num_threads(2)
model=AdoreLite().cuda().eval()
with np.load(ROOT/'models/adore-lite-experimental.npz',allow_pickle=False) as w:model.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
patches=load_patches();report=evaluate({'v2':model},patches,cases=['clean','heavy','codec'],codec_cache=ROOT/'native/training-data/v3/codec-validation')
report['scale1_5']={};residual=[]
with torch.inference_mode():
    for patch in patches:
        high=Image.fromarray(patch);low=high.resize((128,128),Image.Resampling.LANCZOS);x=tensor(low);target=tensor(high);raw=model(x)
        for mode,aa in [('bilinear',False),('bicubic',False),('bicubic',True),('area',False)]:
            options={'size':(192,192),'mode':mode}
            if mode!='area':options.update(align_corners=False,antialias=aa)
            key=mode+('-aa' if aa else '')
            report['scale1_5'].setdefault(key,[]).append(metrics(F.interpolate(raw,**options),target))
        out=raw[0].permute(1,2,0).cpu().numpy()
        # Float-mode Pillow avoids quantizing/clipping only the Lanczos arm.
        lanczos=np.stack([np.asarray(Image.fromarray(out[:,:,c]).resize((192,192),Image.Resampling.LANCZOS)) for c in range(3)])
        report['scale1_5'].setdefault('lanczos',[]).append(metrics(torch.from_numpy(lanczos.copy()).cuda()[None],target))
        residual.append(float((raw-F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)).abs().mean())*255)
for key,values in report['scale1_5'].items():report['scale1_5'][key]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
report['meanAbsoluteNeuralResidual8bit']=float(np.mean(residual))
report['scope']='88 native-resolution validation crops; synthetic degradation, no independent final-test or real-video quality claim'
(ROOT/'diagnostics/lite-v2-quality-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
