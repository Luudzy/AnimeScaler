import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from tensorrt_profiles import DEFAULT_INPUT_SHAPES, parse_custom_shape
from tensorrt_model import TensorRTModel


class TensorRTProfileTests(unittest.TestCase):
    def test_custom_portrait_profile_shape_is_validated_without_tensorRT(self):
        self.assertEqual(parse_custom_shape('640x1136'),(640,1136))
        for value in ('641x1136','40x40x40','1924x1080','32x40'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):parse_custom_shape(value)

    def test_cache_lookup_matches_prepare_identity_and_exact_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);artifact=root/'model.onnx';artifact.write_bytes(b'onnx')
            model=object.__new__(TensorRTModel)
            model.path=artifact;model.cache=root/'cache'
            model.trt=SimpleNamespace(__version__='10.2.0')
            model.torch=SimpleNamespace(version=SimpleNamespace(cuda='12.4'),cuda=SimpleNamespace(
                get_device_name=lambda:'fake gpu',get_device_capability=lambda:(8,6)))
            shape=(1,3,540,960)
            key=hashlib.sha256(json.dumps([
                hashlib.sha256(b'onnx').hexdigest(),'10.2.0','12.4','fake gpu',(8,6),shape
            ]).encode()).hexdigest()
            model.cache.mkdir();(model.cache/(key+'.engine')).write_bytes(b'engine')

            self.assertTrue(model.has_cached_profile(shape))
            self.assertFalse(model.has_cached_profile((1,3,536,960)))

    def test_missing_cached_profiles_batches_exact_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);artifact=root/'model.onnx';artifact.write_bytes(b'onnx')
            model=object.__new__(TensorRTModel)
            model.path=artifact;model.cache=root/'cache'
            model.trt=SimpleNamespace(__version__='10.2.0')
            model.torch=SimpleNamespace(version=SimpleNamespace(cuda='12.4'),cuda=SimpleNamespace(
                get_device_name=lambda:'fake gpu',get_device_capability=lambda:(8,6)))
            available=(1,3,1136,640);missing=(1,3,852,480)
            model.cache.mkdir();(model.profile_cache_path(available)).write_bytes(b'engine')
            self.assertEqual(model.missing_cached_profiles([available,missing]),[missing])

    def test_default_profiles_cover_all_16_by_9_preset_adaptive_shapes(self):
        root=Path(__file__).resolve().parents[1]
        script=("import {PRESET_WIDTHS,resolveInputSize} from './src/core/settings.js';"
                "const shapes=new Set();"
                "for(const preset of Object.keys(PRESET_WIDTHS)) for(let level=0;level<8;level++){"
                "const size=resolveInputSize(3840,2160,preset,level);"
                "shapes.add(`${size.width}x${size.height}`);}"
                "console.log(JSON.stringify([...shapes]));")
        result=subprocess.run(['node','--input-type=module','-e',script],cwd=root,
                              check=True,capture_output=True,text=True,timeout=15)
        expected={f'{width}x{height}' for width,height in DEFAULT_INPUT_SHAPES}
        self.assertEqual(set(json.loads(result.stdout)),expected)


if __name__=='__main__':unittest.main()
