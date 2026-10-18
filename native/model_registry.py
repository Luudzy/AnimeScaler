"""Generic metadata/lifecycle foundation for legacy, external and future Zenkai models."""
from dataclasses import dataclass,asdict
from pathlib import Path
import importlib.util
import hashlib,json
from inference_backends import TorchBackend,OnnxBackend,TensorRTBackend

@dataclass(frozen=True)
class ModelMetadata:
    id:str
    name:str
    family:str
    architecture:str
    native_scale:int
    status:str
    legacy:bool=False
    anime_specific:bool=True
    provider:str='AnimeScaler'
    license:str='NOT VERIFIED'
    code_license:str='NOT VERIFIED'
    source_url:str=''
    weights_path:str=''
    capture_graph:bool=True
    capture_graph_max_pixels:int=0
    def __post_init__(self):
        if not self.id or type(self.native_scale) is not int or not 1<=self.native_scale<=8:raise ValueError('Invalid model identity/scale')
    def to_dict(self):return asdict(self)

class SRModel:
    def __init__(self,metadata,backend,source):self.metadata=metadata;self.backend=backend;self.source=source
    def load(self):self.backend.load(self.source);return self
    def warmup(self,frame):return self.backend.warmup(frame)
    def infer(self,frame):return self.backend.infer(frame)
    def infer_to(self,frame,output_size):
        if hasattr(self.backend,'infer_to'):return self.backend.infer_to(frame,output_size)
        import torch
        result=self.infer(frame)
        is_tensor=isinstance(result,torch.Tensor)
        value=result if is_tensor else torch.from_numpy(result)
        if tuple(value.shape[-2:])!=tuple(output_size):
            value=torch.nn.functional.interpolate(value,size=output_size,mode='bilinear',align_corners=False)
        return value if is_tensor else value.numpy()
    def __call__(self,frame):return self.infer(frame)
    def prepare(self,shape):
        if hasattr(self.backend,'prepare'):self.backend.prepare(shape)
    def close(self):self.backend.close()

class ModelRegistry:
    def __init__(self):self.entries={};self.aliases={};self.loaded={};self.artifact_validators={}
    def register(self,metadata,torch_factory=None,onnx_path=None,aliases=(),artifact_validator=None):
        if metadata.id in self.entries or metadata.id in self.aliases:raise ValueError('Duplicate model')
        if any(a in self.entries or a in self.aliases or a==metadata.id for a in aliases):raise ValueError('Duplicate alias')
        self.entries[metadata.id]=(metadata,torch_factory,onnx_path)
        if artifact_validator is not None:self.artifact_validators[metadata.id]=artifact_validator
        self.aliases.update({a:metadata.id for a in aliases})
    def get(self,name):return self.entries[self.aliases.get(name,name)][0]
    def list_models(self):return [entry[0] for entry in self.entries.values()]
    def availability(self,name,backend,root=None):
        name=self.get(name).id;metadata,factory,path=self.entries[name]
        if backend=='torch':
            if factory is None:return {'available':False,'reason':'PyTorch provider unavailable','artifactPath':None}
            artifact=Path(metadata.weights_path) if metadata.weights_path else None
            if artifact is None:return {'available':False,'reason':'Model weights are not registered','artifactPath':None}
            if not artifact.is_file():return {'available':False,'reason':f'Model weights missing: {artifact.name}','artifactPath':str(artifact)}
        elif backend in ('onnx','tensorrt'):
            if path is None:return {'available':False,'reason':'ONNX artifact is not registered','artifactPath':None}
            artifact=Path(path)
            if not artifact.is_file():return {'available':False,'reason':f'Model artifact missing: {artifact.name}','artifactPath':str(artifact)}
            if backend=='tensorrt':
                if root is None:return {'available':False,'reason':'TensorRT runtime root is unavailable','artifactPath':str(artifact)}
                try:provider=importlib.util.find_spec('tensorrt') is not None
                except (ImportError,ValueError):provider=False
                if not provider:return {'available':False,'reason':'TensorRT provider is unavailable','artifactPath':str(artifact)}
        else:return {'available':False,'reason':f'Unsupported backend: {backend}','artifactPath':None}
        if not artifact.is_file():return {'available':False,'reason':f'Model artifact missing: {artifact.name}','artifactPath':str(artifact)}
        return {'available':True,'reason':None,'artifactPath':str(artifact)}
    def load(self,name,backend='torch',device='cpu',precision='fp32',memory_format=None,root=None):
        name=self.get(name).id;metadata,factory,path=self.entries[name]
        if backend not in ('torch','onnx','tensorrt'):raise ValueError(f'Unsupported backend for {name}: {backend}')
        if backend in ('onnx','tensorrt') and path is None:raise ValueError(f'Unsupported backend for {name}: {backend}')
        status=self.availability(name,backend,root)
        if not status['available']:raise ValueError(status['reason'])
        artifact_path=Path(metadata.weights_path) if backend=='torch' and metadata.weights_path else (Path(path) if backend!='torch' else None)
        artifact=None
        if artifact_path is not None:
            digest=hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if name in self.artifact_validators:self.artifact_validators[name](artifact_path,digest)
            artifact={'path':str(artifact_path),'sha256':digest,'kind':'weights' if backend=='torch' else 'onnx'}
        if backend=='torch' and factory is not None:engine=TorchBackend(device,precision,memory_format);source=factory
        elif backend=='onnx' and path is not None:engine=OnnxBackend();source=path
        elif backend=='tensorrt' and path is not None and root is not None:engine=TensorRTBackend();source=(root,name,path,metadata.native_scale)
        else:raise ValueError(f'Unsupported backend for {name}: {backend}')
        self.unload(name);model=SRModel(metadata,engine,source).load();model.artifact=artifact;self.loaded[name]=model;return model
    def unload(self,name):
        name=self.get(name).id
        if name in self.loaded:self.loaded.pop(name).close()

def build_registry(root):
    from user_models import USER_MODELS,load_user_model
    registry=ModelRegistry();root=Path(root)
    aliases={'hfa2k-cugan':['realcugan-x2'],'realesr-anime':['realesr-animevideov3']}
    for name,spec in USER_MODELS.items():
        if not (root/'models'/spec['file']).exists():continue
        architecture='SRVGGNetCompact' if spec['architecture']=='Compact' else spec['architecture']
        sources={'hfa2k-cugan':('MIT','https://github.com/bilibili/ailab/tree/main/Real-CUGAN'),
                 'realesr-anime':('BSD-3-Clause','https://github.com/xinntao/Real-ESRGAN'),
                 'span-mssim':('Apache-2.0','https://github.com/hongyuanyu/SPAN')}
        code_license,source=sources[name]
        import os
        optimize_realesr=os.environ.get('ANIMESCALER_REALESR_OPTIMIZE','1')!='0'
        meta=ModelMetadata(name,spec['label'],'External',architecture,spec['scale'],'external-baseline',provider='User-supplied checkpoint',code_license=code_license,source_url=source,weights_path=str(root/'models'/spec['file']),capture_graph=False,
                           capture_graph_max_pixels=640*360 if name=='realesr-anime' and optimize_realesr else 0)
        def external(n=name,optimize_realesr=optimize_realesr):
            from tiled_model import TiledModel
            model,metadata=load_user_model(root,n)
            if not metadata['supportsHalf']:raise ValueError('FP16 unsupported')
            if n=='realesr-anime' and optimize_realesr:
                from realesr_optimized import RealESRAnimeVideo
                return RealESRAnimeVideo(model)
            return TiledModel(model,metadata['scale'])
        registry.register(meta,external,root/'models'/f'{name}.onnx',aliases.get(name,()))
    frozen=root/'models/legacy/adore-lite-v3/frozen-20260930'
    if (frozen/'baseline-lock.json').exists():
        def validate_frozen(path,digest):
            lock=json.loads((frozen/'baseline-lock.json').read_text())
            expected=next((r['sha256'] for r in lock['files'] if r['snapshot']==path.name),None)
            if digest!=expected:raise ValueError(f'Frozen V3 changed: {path.name}')
        def legacy_v3():
            import numpy as np,torch
            from lite_v3 import AdoreLiteV3
            weights=frozen/'adore-lite-v3-experimental.npz'
            model=AdoreLiteV3()
            with np.load(weights,allow_pickle=False) as values:model.load_state_dict({k:torch.from_numpy(values[k].copy()).float() for k in values.files})
            return model
        registry.register(ModelMetadata('adore-lite-v3','Adore Lite V3','Adore','AdoreLiteV3',2,'legacy-baseline',legacy=True,weights_path=str(frozen/'adore-lite-v3-experimental.npz')),legacy_v3,frozen/'adore-lite-v3-experimental.onnx',artifact_validator=validate_frozen)
    from legacy_loaders import register_legacy
    register_legacy(registry,root)
    from zenkai_1_0 import register_zenkai_1_0
    register_zenkai_1_0(registry,root)
    from zenkai_realesr_deploy import register_zenkai_realesr
    register_zenkai_realesr(registry,root)
    from anime4k_registry import register_anime4k
    register_anime4k(registry,root)
    return registry
