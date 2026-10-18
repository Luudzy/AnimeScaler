"""Independent shader-equation oracle, not a test of the conversion helpers."""
import re, unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]

def shader_oracle(text, value):
    # Interpret each matrix term directly. GLSL matrices are column-major;
    # texOff clamps at the source edge. Neither conv2d nor production parser used.
    features={'MAIN':np.concatenate([value,np.ones((*value.shape[:2],1),np.float32)],axis=2)}
    for section in text.split('//!DESC')[1:-1]:
        save=re.search(r'//!SAVE (\w+)',section)[1]
        sources={}
        for name,expr in re.findall(r'#define (go_\d+\(x_off, y_off\)|g_\d+) ([^\n]+)',section):
            key=name.split('(')[0]
            texture=re.search(r'(\w+)_tex',expr)[1]
            sources[key]=(texture,'max(' in expr,'max(-' in expr)
        result=np.zeros((*value.shape[:2],4),np.float32)
        for numbers,key,coords in re.findall(r'mat4\(([^)]+)\)\s*\*\s*(go_\d+|g_\d+)(?:\(([^)]+)\))?',section):
            matrix=np.array([float(v) for v in numbers.split(',')],np.float32).reshape(4,4).T
            texture,split,negative=sources[key]
            source=features[texture]
            if negative:source=-source
            if split:source=np.maximum(source,0)
            dx,dy=(map(lambda s:int(float(s)),coords.split(',')) if coords else (0,0))
            iy=np.clip(np.arange(source.shape[0])+dy,0,source.shape[0]-1)
            ix=np.clip(np.arange(source.shape[1])+dx,0,source.shape[1]-1)
            result+=np.einsum('oi,hwi->hwo',matrix,source[iy[:,None],ix[None,:]])
        bias=re.search(r'result \+= vec4\(([^)]+)\)',section)[1]
        result+=np.array([float(v) for v in bias.split(',')],np.float32)
        features[save]=result
    phases=features['conv2d_last_tf']
    residual=phases.reshape(value.shape[0],value.shape[1],2,2).transpose(0,2,1,3).reshape(value.shape[0]*2,value.shape[1]*2)
    base=F.interpolate(torch.from_numpy(value).permute(2,0,1)[None],scale_factor=2,mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
    return base+residual[:,:,None]


class Anime4KTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    @unittest.skipUnless((ROOT/'models/anime4k/Anime4K_Upscale_CNN_x2_S.glsl').is_file(), 'download Anime4K S shader first')
    def test_actual_s_and_m_match_shader_equations(self):
        from anime4k_model import Anime4K
        for variant in ['S','M']:
            path=ROOT/f'models/anime4k/Anime4K_Upscale_CNN_x2_{variant}.glsl'
            model=Anime4K.from_file(path).eval()
            for h,w in [(1,1),(5,7),(6,8)]:
                value=np.random.default_rng(929).random((h,w,3),dtype=np.float32)
                expected=shader_oracle(path.read_text(),value)
                with torch.inference_mode():actual=model(torch.from_numpy(value).permute(2,0,1)[None])[0].permute(1,2,0).numpy()
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=3e-6)

    @unittest.skipUnless((ROOT/'models/anime4k/Anime4K_Upscale_CNN_x2_S.glsl').is_file(), 'download Anime4K S shader first')
    def test_shader_parser_rejects_unknown_code_and_nonfinite_weights(self):
        from anime4k_model import Anime4K
        path=ROOT/'models/anime4k/Anime4K_Upscale_CNN_x2_S.glsl'
        text=path.read_text()
        for malformed in [text.replace('return result;','result += sin(result); return result;',1),
                          text.replace('-0.0057322932','nan',1),text.replace('MAIN_texOff','EVIL_texOff',1),text.replace('go_0(-1.0, -1.0)','go_0(-0.5, -1.0)',1),text.replace('(MAIN_texOff(vec2(x_off, y_off)))','(sin(MAIN_texOff(vec2(x_off, y_off))))',1),text.replace('float c1 = c0;', 'float c1 = -c0;'),text.replace('//!WIDTH MAIN.w','//!WIDTH MAIN.w 2 *',1)]:
            with self.assertRaises(ValueError):Anime4K.from_text(malformed)

    @unittest.skipUnless((ROOT/'models/anime4k/Anime4K_Upscale_CNN_x2_S.glsl').is_file(), 'download Anime4K S shader first')
    def test_repeat_upscale_obeys_target_and_avoids_unnecessary_pass(self):
        from anime4k_model import Anime4K
        model=Anime4K.from_file(ROOT/'models/anime4k/Anime4K_Upscale_CNN_x2_S.glsl')
        x=torch.rand(2,3,7,9)
        with torch.inference_mode():
            for size in [(14,18),(21,27),(28,36),(7,9),(5,6)]:
                y=model.forward_to(x,size)
                self.assertEqual(tuple(y.shape),(2,3,*size))
                self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.equal(model.forward_to(x,(7,9)),x))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    @unittest.skipUnless((ROOT/'models/anime4k/Anime4K_Upscale_CNN_x2_S.glsl').is_file(), 'download Anime4K S shader first')
    def test_hybrid_cap_is_explicit_preserves_small_input(self):
        from anime4k_model import Anime4K, RealESRAnime4K
        from test_realesr_optimized import CompactReference
        from realesr_optimized import RealESRAnimeVideo
        sr=RealESRAnimeVideo(CompactReference().eval())
        ak=Anime4K.from_file(ROOT/'models/anime4k/Anime4K_Upscale_CNN_x2_S.glsl')
        model=RealESRAnime4K(sr,ak,max_width=16,max_height=12)
        for h,w in [(8,12),(20,32),(24,12)]:
            x=torch.rand(1,3,h,w)
            with torch.inference_mode():
                y=model.forward_to(x,(h*2,w*2))
                self.assertEqual(y.shape,(1,3,h*2,w*2))
                self.assertLessEqual(model.last_sr_input[0],12)
                self.assertLessEqual(model.last_sr_input[1],16)
                if (h,w)==(8,12):torch.testing.assert_close(y,sr.forward_to(x,(h*2,w*2)),atol=0,rtol=0)

class Anime4KRegistryTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'models/realesr-animevideov3.pth').is_file(), 'download RealESR checkpoint first')
    def test_registered_sources_are_bound_and_tensorrt_rejected(self):
        from model_registry import build_registry
        registry=build_registry(ROOT)
        for name in ['anime4k-s','anime4k-m','realesr-anime4k-s-360']:
            meta=registry.get(name)
            self.assertEqual(meta.status,'experimental')
            self.assertFalse(meta.capture_graph)
            with self.assertRaisesRegex(ValueError,'Unsupported backend'):
                registry.load(name,backend='tensorrt',root=ROOT)
        import json,hashlib
        manifest=json.loads(Path(registry.get('realesr-anime4k-s-360').weights_path).read_text())
        self.assertEqual(len(manifest['artifacts']),2)
        for row in manifest['artifacts']:
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'])

    @unittest.skipUnless((ROOT/'models/realesr-animevideov3.pth').is_file(), 'download RealESR checkpoint first')
    def test_changed_component_rejected_before_loading(self):
        import tempfile,json,hashlib,shutil
        from model_registry import ModelRegistry
        from anime4k_registry import register_anime4k
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'models/anime4k';folder.mkdir(parents=True)
            manifest=json.loads((ROOT/'models/anime4k/realesr-anime4k-s-360.json').read_text())
            for row in manifest['artifacts']:
                target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/row['path'],target)
            shutil.copyfile(ROOT/'models/anime4k/realesr-anime4k-s-360.json',folder/'realesr-anime4k-s-360.json')
            (root/manifest['artifacts'][0]['path']).write_bytes(b'changed')
            registry=ModelRegistry();register_anime4k(registry,root)
            with patch('model_registry.TorchBackend.load') as loader:
                with self.assertRaisesRegex(ValueError,'Anime4K artifact changed'):
                    registry.load('realesr-anime4k-s-360')
                loader.assert_not_called()

if __name__=='__main__':unittest.main()
