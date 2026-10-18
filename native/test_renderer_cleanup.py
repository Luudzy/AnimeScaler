import unittest

from server import Renderer


class FakeModel:
    def __init__(self): self.close_calls=0
    def close(self): self.close_calls+=1


class FakeRegistry:
    def __init__(self): self.unloaded=[]
    def unload(self,name): self.unloaded.append(name)


class RendererCleanupTests(unittest.TestCase):
    def test_close_releases_cached_models_once_and_is_idempotent(self):
        worker=FakeModel();registry=FakeRegistry()
        renderer=Renderer.__new__(Renderer)
        renderer.models={'adore-lite-v3':FakeModel(),'anime4kcpp-acnet-f8b4':worker}
        renderer.registry=registry;renderer.graph=object();renderer.shape=(40,40,80,80)
        renderer.model=worker;renderer.current_model='anime4kcpp-acnet-f8b4'
        renderer.close();renderer.close()
        self.assertEqual(registry.unloaded,['adore-lite-v3'])
        self.assertEqual(worker.close_calls,1)
        self.assertEqual(renderer.models,{})
        self.assertIsNone(renderer.model)


if __name__=='__main__': unittest.main()
