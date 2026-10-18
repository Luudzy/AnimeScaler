import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tensorrt_model import TensorRTModel


class Runtime:
    def deserialize_cuda_engine(self, payload):
        return None


class CacheTests(unittest.TestCase):
    def make_model(self, root):
        root=Path(root);artifact=root/'model.onnx';artifact.write_bytes(b'onnx')
        class Logger:
            WARNING=1
            def __init__(self,level):pass
        trt=types.SimpleNamespace(
            __version__='10.0', Logger=Logger,
            Runtime=lambda logger:Runtime())
        torch=types.SimpleNamespace(
            version=types.SimpleNamespace(cuda='12.0'),
            cuda=types.SimpleNamespace(get_device_name=lambda:'Fake GPU',get_device_capability=lambda:(8,0)))
        with patch.dict(sys.modules,{'tensorrt':trt,'user_models':types.SimpleNamespace(USER_MODELS={})}):
            model=TensorRTModel(root,'test',torch,path=artifact,scale=2)
        return model,trt

    def cache_path(self, model, trt, torch, shape):
        identity=[hashlib.sha256(model.path.read_bytes()).hexdigest(),trt.__version__,
                  torch.version.cuda,torch.cuda.get_device_name(),torch.cuda.get_device_capability(),shape]
        key=hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        return model.cache/(key+'.engine')

    def test_corrupt_cache_is_rejected_without_build_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            model,trt=self.make_model(folder)
            shape=(1,3,40,40);cache=self.cache_path(model,trt,model.torch,shape)
            cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(b'corrupt')
            with patch.dict(sys.modules,{'tensorrt':trt}),patch.dict(os.environ,{},clear=False):
                os.environ.pop('ANIMESCALER_TRT_BUILD',None)
                with self.assertRaisesRegex(RuntimeError,'Engine TensorRT ausente'):
                    model.prepare(shape)
            self.assertEqual(cache.read_bytes(),b'corrupt')

    def test_invalid_serialized_engine_does_not_replace_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            model,trt=self.make_model(folder)
            shape=(1,3,40,40);cache=self.cache_path(model,trt,model.torch,shape)
            cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(b'corrupt')
            network=types.SimpleNamespace(num_inputs=1,num_outputs=1,get_input=lambda index:types.SimpleNamespace(name='image'))
            class Config:
                def set_memory_pool_limit(self,*args):pass
                def add_optimization_profile(self,*args):pass
            class Builder:
                platform_has_fast_fp16=False
                def __init__(self,logger):pass
                def create_network(self,flags):return network
                def create_builder_config(self):return Config()
                def create_optimization_profile(self):return types.SimpleNamespace(set_shape=lambda *args:True)
                def build_serialized_network(self,*args):return b'invalid-new-engine'
            trt.NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0)
            trt.OnnxParser=lambda *args:types.SimpleNamespace(parse=lambda payload:True,num_errors=0)
            trt.Builder=Builder
            trt.MemoryPoolType=types.SimpleNamespace(WORKSPACE=0)
            trt.BuilderFlag=types.SimpleNamespace(FP16=0)
            with patch.dict(sys.modules,{'tensorrt':trt}),patch.dict(os.environ,{'ANIMESCALER_TRT_BUILD':'1'}):
                with self.assertRaisesRegex(RuntimeError,'TensorRT engine load failed'):
                    model.prepare(shape)
            self.assertEqual(cache.read_bytes(),b'corrupt')
            self.assertEqual(list(model.cache.glob('*.tmp')),[])


if __name__=='__main__':unittest.main()
