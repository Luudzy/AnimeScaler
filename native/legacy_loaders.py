"""Legacy checkpoint validation and preprocessing, kept out of the server."""
import hashlib,json
import numpy as np

def read_weights(path,expected,model):
    import torch
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f'Invalid checksum: {path.name}')
    with np.load(path,allow_pickle=False) as weights:
        model.load_state_dict({k:torch.from_numpy(weights[k].copy()).float() for k in weights.files},strict=True)
    return model

def register_legacy(registry,root):
    from model_registry import ModelMetadata
    definitions=[('adore','Adore','RealCUGAN','adore-native.npz','2x_Adore_renarchi_webgpu_fp16.onnx'),
                 ('adore-lite','Adore Lite V2','AdoreLite16-v1','adore-lite-experimental.npz','adore-lite-experimental.onnx'),
                 ('animejanai','AnimeJaNai SuperUltraCompact','AnimeJaNai','animejanai-native.npz','2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu_fp16.onnx')]
    for name,label,architecture,file,onnx in definitions:
        path=root/'models'/file
        if not path.exists():continue
        def factory(n=name,p=path):
            if n=='adore':
                import torch
                from spandrel.architectures.RealCUGAN.__arch.upcunet_v3 import UpCunet2x_fast
                model=read_weights(p,(p.parent/'adore-native.sha256').read_text().strip(),UpCunet2x_fast())
                class NormalizedAdore(torch.nn.Module):
                    def __init__(self,model):super().__init__();self.model=model
                    def forward(self,x):return self.model(x*.7+.15).sub(.15).div(.7)
                return NormalizedAdore(model)
            report=json.loads(p.with_suffix('.json').read_text())
            if n=='adore-lite':
                from lite_model import AdoreLite
                if report['architecture']!='AdoreLite16-v1' or not report.get('improvesOverBilinear'):raise ValueError('Invalid legacy Lite metadata')
                model=AdoreLite()
            else:
                from animejanai_model import AnimeJaNai
                model=AnimeJaNai()
            return read_weights(p,report['weightsSha256'],model)
        registry.register(ModelMetadata(name,label,'External' if name=='animejanai' else 'Adore',architecture,2,'legacy-baseline',legacy=True,weights_path=str(path)),factory,root/'models'/onnx)
