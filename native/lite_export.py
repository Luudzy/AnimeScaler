"""Export a reviewed V3 candidate without replacing the V2 deployment."""
import sys,json,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from lite_v3 import AdoreLiteV3
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--experimental',action='store_true');args=p.parse_args()
    report=json.loads((args.candidate/'report.json').read_text())
    if not report['fp16Approved']:raise ValueError('FP16 validation failed')
    if not report['validationGate']['passed'] and not args.experimental:raise ValueError('Validation gate failed; experimental export requires --experimental')
    saved=torch.load(args.candidate/'best_composite-master-fp32.pt',map_location='cpu',weights_only=True)
    model=AdoreLiteV3(report['channels'],report['blocks'],report['residualScale']).eval();model.load_state_dict(saved['model'])
    # ONNX has FP32 I/O/master values; TensorRT independently chooses FP16 kernels.
    onnx=args.candidate/'model.onnx'
    torch.onnx.export(model,torch.zeros(1,3,40,40),str(onnx),input_names=['input'],output_names=['output'],opset_version=17,
                      dynamic_axes={'input':{2:'height',3:'width'},'output':{2:'out_height',3:'out_width'}},dynamo=False)
    import onnxruntime as ort
    session=ort.InferenceSession(str(onnx),providers=['CPUExecutionProvider']);errors=[]
    for h,w in [(40,40),(48,64),(72,96)]:
        x=torch.rand(1,3,h,w)
        with torch.inference_mode():expected=model(x).numpy()
        actual=session.run(None,{'input':x.numpy()})[0];errors.append(float(np.max(np.abs(actual-expected))))
    if max(errors)>2e-5:raise ValueError(f'ONNX parity failed: {errors}')
    # Publishing this separate option never changes the V2 files or defaults.
    path=ROOT/'models/adore-lite-v3-experimental.npz';path.write_bytes((args.candidate/'deploy-fp16.npz').read_bytes())
    path.with_suffix('.onnx').write_bytes(onnx.read_bytes())
    metadata={**report,'architecture':'AdoreLiteV3','experimental':True,'releaseApproved':False,'onnxMaxErrors':errors,
              'weightsSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'masterCheckpoint':str(args.candidate/'best_composite-master-fp32.pt')}
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print(json.dumps({'exported':str(path),'onnxMaxErrors':errors}))
if __name__=='__main__':main()
