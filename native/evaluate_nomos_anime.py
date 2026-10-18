"""External static anime validation from visually reviewed user-provided PNGs."""
import sys,json,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import torch
from lite_model import AdoreLite
from lite_v3 import AdoreLiteV3
from lite_evaluation import ROOT,evaluate

# Reviewed in diagnostics/nomos-contact-sheet.jpg before seeing V3 results.
# Keep the whole selection out of training; some images share the same scene.
ANIME=['0001','0121','0181','0481','0841','0901','0961','1021','1321','1381','1441','1681']
DIFFICULT={'0001':'small background structures','0901':'thin sketch lineart','1021':'faces and hair','1681':'dark foliage'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('F:/treino/nomos_uni_gt'));p.add_argument('--candidate',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);patches=[];hard=[];manifest=[]
    for name in ANIME:
        path=args.source/f'{name}.png'
        with Image.open(path) as image:
            for x,y in [(0,0),(image.width-192,0),(0,image.height-192),(image.width-192,image.height-192)]:
                patch=np.array(image.crop((x,y,x+192,y+192)).convert('RGB'));patches.append(patch)
                if name in DIFFICULT:hard.append(patch)
        manifest.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':'external-validation',
                         'category':'anime','difficultTag':DIFFICULT.get(name),'temporalOrderKnown':False})
    saved=torch.load(args.candidate/'best_composite-master-fp32.pt',map_location='cpu',weights_only=True);c=saved['config']
    v3=AdoreLiteV3(c['channels'],c['blocks'],c['residualScale']).eval();v3.load_state_dict(saved['model'])
    v2=AdoreLite().eval()
    with np.load(ROOT/'models/adore-lite-experimental.npz',allow_pickle=False) as w:v2.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
    models={'candidate':v3,'previous':v2}
    result={'scope':'12 visually reviewed anime PNGs, 4 fixed native crops each; existing scenes may be correlated. Static anime images with synthetic degradation, not real streaming sequences.',
            'manifest':manifest,'validation_anime_real':evaluate(models,patches,device='cpu'),
            'validation_difficult':evaluate(models,hard,device='cpu')}
    (args.candidate/'nomos-anime.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='manifest'},indent=2))
if __name__=='__main__':main()
