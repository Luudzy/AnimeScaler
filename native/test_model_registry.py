import unittest
import tempfile,json
from pathlib import Path
import torch
from unittest.mock import patch
import hashlib
from model_registry import ModelMetadata,ModelRegistry
from inference_backends import TorchBackend,OnnxBackend

class RegistryTests(unittest.TestCase):
    def test_backend_availability_reports_missing_weights_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing=Path(tmp)/'missing.npz';registry=ModelRegistry()
            registry.register(ModelMetadata('missing','Missing','Test','Identity',2,'test',weights_path=str(missing)),
                torch_factory=lambda:torch.nn.Identity())
            status=registry.availability('missing','torch')
            self.assertFalse(status['available'])
            self.assertIn('weights',status['reason'].lower())
            self.assertEqual(registry.loaded,{})

    def test_tensorrt_availability_requires_artifact_and_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifact=root/'model.onnx';artifact.write_bytes(b'onnx')
            registry=ModelRegistry()
            registry.register(ModelMetadata('test','Test','Test','Identity',2,'test'),onnx_path=artifact)
            with patch('model_registry.importlib.util.find_spec',return_value=None):
                missing=registry.availability('test','tensorrt',root)
            self.assertFalse(missing['available'])
            self.assertIn('provider',missing['reason'].lower())
            with patch('model_registry.importlib.util.find_spec',return_value=object()):
                available=registry.availability('test','tensorrt',root)
            self.assertTrue(available['available'])

    def test_frozen_export_verified_before_backend_load(self):
        from model_registry import build_registry
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'models/legacy/adore-lite-v3/frozen-20260930';folder.mkdir(parents=True)
            export=folder/'adore-lite-v3-experimental.onnx';export.write_bytes(b'changed')
            (folder/'baseline-lock.json').write_text(json.dumps({'files':[{'snapshot':export.name,'sha256':'0'*64}]}))
            registry=build_registry(root)
            for backend, cls in [('onnx','OnnxBackend'),('tensorrt','TensorRTBackend')]:
                with patch('model_registry.importlib.util.find_spec',return_value=object()), patch('model_registry.'+cls+'.load') as loader:
                    with self.assertRaisesRegex(ValueError,'Frozen V3 changed'):
                        registry.load('adore-lite-v3',backend=backend,root=root)
                    loader.assert_not_called()
            (folder/'baseline-lock.json').write_text(json.dumps({'files':[{'snapshot':export.name,'sha256':hashlib.sha256(b'changed').hexdigest()}]}))
            for backend, cls in [('onnx','OnnxBackend'),('tensorrt','TensorRTBackend')]:
                with patch('model_registry.importlib.util.find_spec',return_value=object()), patch('model_registry.'+cls+'.load') as loader:
                    model=registry.load('adore-lite-v3',backend=backend,root=root)
                    loader.assert_called_once()
                    self.assertEqual(model.artifact['path'],str(export))

    def test_unlisted_frozen_export_rejected(self):
        from model_registry import build_registry
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'models/legacy/adore-lite-v3/frozen-20260930';folder.mkdir(parents=True)
            (folder/'adore-lite-v3-experimental.onnx').write_bytes(b'export')
            (folder/'baseline-lock.json').write_text(json.dumps({'files':[]}))
            with patch('model_registry.OnnxBackend.load') as loader:
                with self.assertRaisesRegex(ValueError,'Frozen V3 changed'):
                    build_registry(root).load('adore-lite-v3',backend='onnx')
                loader.assert_not_called()

    def test_provenance_identifies_executed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights=Path(tmp)/'weights.npz';weights.write_bytes(b'weights')
            export=Path(tmp)/'model.onnx';export.write_bytes(b'export')
            registry=ModelRegistry()
            registry.register(ModelMetadata('test','Test','Test','Identity',1,'test',weights_path=str(weights)),onnx_path=export)
            with patch('model_registry.OnnxBackend.load'):
                model=registry.load('test',backend='onnx')
            self.assertEqual(model.artifact['sha256'],hashlib.sha256(b'export').hexdigest())
            self.assertEqual(model.artifact['path'],str(export))

    def test_identity_alias_and_scale_validation(self):
        registry=ModelRegistry();metadata=ModelMetadata('external','Original name','External','Architecture',4,'external-baseline')
        registry.register(metadata,torch_factory=lambda:torch.nn.Identity(),aliases=['alias'])
        self.assertEqual(registry.get('alias').native_scale,4)
        with self.assertRaises(ValueError):registry.register(metadata)
        with self.assertRaises(ValueError):ModelMetadata('bad','Bad','External','Unknown',0,'experimental')
        with self.assertRaises(ValueError):registry.load('alias',backend='ncnn')
    def test_backend_lifecycle(self):
        with tempfile.TemporaryDirectory() as folder:
            weights=Path(folder)/'weights.bin';weights.write_bytes(b'test weights')
            registry=ModelRegistry();registry.register(ModelMetadata('test','Test','Test','Identity',1,'test',weights_path=str(weights)),lambda:torch.nn.Identity())
            model=registry.load('test');x=torch.rand(1,3,8,8)
            self.assertTrue(torch.equal(model.warmup(x),x));registry.unload('test')
            with self.assertRaises(RuntimeError):model.infer(x)
    def test_onnx_sync_does_not_invoke_cuda(self):
        OnnxBackend(provider='DmlExecutionProvider').synchronize()
        with self.assertRaises(RuntimeError):OnnxBackend().infer(None)

    def test_frozen_baseline_rejects_modified_weights(self):
        from model_registry import build_registry
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'models/legacy/adore-lite-v3/frozen-20260930';folder.mkdir(parents=True)
            (folder/'adore-lite-v3-experimental.npz').write_bytes(b'changed')
            (folder/'baseline-lock.json').write_text(json.dumps({'files':[{'snapshot':'adore-lite-v3-experimental.npz','sha256':'0'*64}]}))
            registry=build_registry(root)
            self.assertTrue(registry.get('adore-lite-v3').legacy)
            with self.assertRaisesRegex(ValueError,'Frozen V3 changed'):registry.load('adore-lite-v3')

if __name__=='__main__':unittest.main()
