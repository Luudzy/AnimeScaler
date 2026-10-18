import unittest
from pathlib import Path
from unittest.mock import patch

from server import Renderer,registry_service_catalog,anime4kcpp_worker_availability
from model_registry import ModelMetadata,ModelRegistry


class FakeModel:
    def __init__(self): self.closed=0
    def close(self): self.closed+=1


class FakeRegistry:
    def __init__(self): self.unloaded=[]
    def unload(self,name): self.unloaded.append(name)


class ModelCacheTests(unittest.TestCase):
    def test_anime4kcpp_catalog_requires_worker_and_runtime_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertIn('binary',anime4kcpp_worker_availability(root)['reason'].lower())
            worker=root/'third_party/Anime4KCPP-master/worker/build/bin/Release/anime4kcpp_worker.exe'
            runtime=root/'third_party/Anime4KCPP-master/build-opencl-md/bin/Release'
            worker.parent.mkdir(parents=True);worker.write_bytes(b'worker')
            self.assertIn('runtime',anime4kcpp_worker_availability(root)['reason'].lower())
            runtime.mkdir(parents=True)
            self.assertTrue(anime4kcpp_worker_availability(root)['available'])

    def test_service_catalog_advertises_only_loadable_models_with_reasons(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weights=root/'good.bin';weights.write_bytes(b'weights')
            registry=ModelRegistry()
            registry.register(ModelMetadata('good','Good','Test','Identity',2,'test',weights_path=str(weights)),
                torch_factory=lambda:FakeModel())
            registry.register(ModelMetadata('missing','Missing','Test','Identity',2,'test',weights_path=str(root/'missing.bin')),
                torch_factory=lambda:FakeModel())
            catalog=registry_service_catalog(registry,'torch',root)
            self.assertEqual(catalog['models'],['good'])
            self.assertFalse(catalog['modelAvailability']['missing']['available'])
            self.assertIn('missing',catalog['modelAvailability']['missing']['reason'].lower())

    def test_model_switch_releases_all_inactive_models_before_loading_one(self):
        renderer=Renderer.__new__(Renderer)
        old_worker=FakeModel();renderer.models={'adore-lite-v3':FakeModel(),'anime4kcpp-acnet-f8b4':old_worker}
        renderer.registry=FakeRegistry();renderer.info={'models':['anime4kcpp-artcnn-c4f16']}
        renderer.current_model='anime4kcpp-acnet-f8b4';renderer.model=old_worker
        renderer.graph=object();renderer.shape=(40,40,80,80);renderer.external_model=True
        new_worker=FakeModel()
        with patch('server.Anime4KCPPWorker',return_value=new_worker):
            renderer.select_model('anime4kcpp-artcnn-c4f16')
        self.assertEqual(set(renderer.models),{'anime4kcpp-artcnn-c4f16'})
        self.assertEqual(renderer.registry.unloaded,['adore-lite-v3'])
        self.assertEqual(old_worker.closed,1)
        self.assertIs(renderer.model,new_worker)

    def test_selecting_current_model_still_evicts_stale_cache_entries(self):
        renderer=Renderer.__new__(Renderer)
        active=FakeModel();renderer.models={'anime4kcpp-artcnn-c4f16':active,'adore-lite-v3':FakeModel()}
        renderer.registry=FakeRegistry();renderer.info={'models':['anime4kcpp-artcnn-c4f16']}
        renderer.current_model='anime4kcpp-artcnn-c4f16';renderer.model=active
        renderer.select_model('anime4kcpp-artcnn-c4f16')
        self.assertEqual(set(renderer.models),{'anime4kcpp-artcnn-c4f16'})
        self.assertEqual(renderer.registry.unloaded,['adore-lite-v3'])
        self.assertIs(renderer.model,active)


if __name__=='__main__': unittest.main()
