import builtins
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server


class FakeWorker:
    def __init__(self,*args):self.closed=0
    def close(self):self.closed+=1


class OpenCLRendererTests(unittest.TestCase):
    def prepare_root(self,root):
        worker=root/'third_party/Anime4KCPP-master/worker/build/bin/Release/anime4kcpp_worker.exe'
        runtime=root/'third_party/Anime4KCPP-master/build-opencl-md/bin/Release'
        worker.parent.mkdir(parents=True);worker.write_bytes(b'fake worker')
        runtime.mkdir(parents=True)

    def test_opencl_renderer_skips_torch_rocm_and_miopen_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.prepare_root(root)
            original=builtins.__import__
            def importer(name,*args,**kwargs):
                if name=='torch':raise AssertionError('OpenCL-only startup imported Torch')
                return original(name,*args,**kwargs)
            with patch.object(server,'ROOT',root),patch.dict(os.environ,{'ANIMESCALER_BACKEND':'opencl'}),patch('builtins.__import__',side_effect=importer):
                renderer=server.Renderer()
            self.assertEqual(renderer.backend,'opencl')
            self.assertIsNone(renderer.torch)
            self.assertIsNone(renderer.registry)
            self.assertIsNone(renderer.current_model)
            self.assertTrue(renderer.info['models'])
            self.assertTrue(all(name.startswith('anime4kcpp-') for name in renderer.info['models']))
            self.assertEqual(renderer.info['precision'],'native')
            self.assertFalse((root/'diagnostics/native-cache').exists())
            renderer.close()

    def test_opencl_selection_uses_worker_only_and_cleans_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.prepare_root(root);worker=FakeWorker()
            with patch.object(server,'ROOT',root),patch.dict(os.environ,{'ANIMESCALER_BACKEND':'opencl'}):
                renderer=server.Renderer()
                with patch.object(server,'Anime4KCPPWorker',return_value=worker):
                    renderer.select_model('anime4kcpp-acnet-f8b4')
                self.assertTrue(renderer.external_model)
                with self.assertRaisesRegex(ValueError,'indisponível'):
                    renderer.select_model('adore')
                renderer.close()
            self.assertEqual(worker.closed,1)


if __name__=='__main__':unittest.main()
