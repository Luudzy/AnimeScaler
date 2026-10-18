"""Export optimized user models with dynamic-shape CPU ONNX parity checks."""
import sys,json,hashlib
from pathlib import Path
import torch,torchvision
# Reuse only optional Python architecture modules; torch comes from this interpreter.
sys.path.append('F:/comfyui-rocm-071926/python_env/Lib/site-packages')
from user_models import USER_MODELS,load_user_model
import onnxruntime as ort
import numpy as np

class SingleInput(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x)

def main():
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(2);torch.manual_seed(929);results={}
    for name,spec in USER_MODELS.items():
        model,metadata=load_user_model(root,name)
        raw,_=load_user_model(root,name,optimize=False)
        x=torch.rand(1,3,40,48)
        with torch.inference_mode():fusion_error=float((model(x)-raw(x)).abs().max())
        if fusion_error>2e-5:raise ValueError(f'Fusion parity failed: {name}: {fusion_error}')
        path=root/'models'/f'{name}.onnx'
        torch.onnx.export(SingleInput(model),x,str(path),input_names=['input'],output_names=['output'],opset_version=17,
                          dynamic_axes={'input':{2:'height',3:'width'},'output':{2:'out_height',3:'out_width'}},dynamo=False)
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);errors=[]
        for h,w in [(40,48),(48,64),(72,80)]:
            x=torch.rand(1,3,h,w)
            with torch.inference_mode():expected=model(x).numpy()
            actual=session.run(None,{'input':x.numpy()})[0]
            assert actual.shape==(1,3,h*spec['scale'],w*spec['scale'])
            errors.append(float(np.abs(actual-expected).max()))
        if max(errors)>1e-4:raise ValueError(f'ONNX parity failed: {name}: {errors}')
        results[name]={**spec,**metadata,'parametersAfterFusion':sum(p.numel() for p in model.parameters()),'fusionMaxError':fusion_error,
                       'onnxMaxErrors':errors,'sourceSha256':hashlib.sha256((root/'models'/spec['file']).read_bytes()).hexdigest(),
                       'onnxSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tensorRtHardwareValidated':False}
        (root/'models'/f'{name}.json').write_text(json.dumps(results[name],indent=2));print(json.dumps({name:results[name]}),flush=True)
    (root/'diagnostics/user-model-export.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
