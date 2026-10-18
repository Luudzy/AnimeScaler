from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
import unittest

from server import Renderer, parse_profile_check, tensorrt_profile_result


class TensorRTProfileProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_profile_check_accepts_valid_variable_aspect_shape(self):
        payload=json.dumps({'type':'profile-check','model':'adore','shapes':[[640,1136],[480,852]]})
        self.assertEqual(parse_profile_check(payload,'adore',['adore']),[(640,1136),(480,852)])

    def test_profile_check_rejects_wrong_model_and_invalid_shape(self):
        for data,selected in [
            ({'type':'profile-check','model':'other','shapes':[[640,1136]]},'adore'),
            ({'type':'profile-check','model':'adore','shapes':[[641,1136]]},'adore'),
            ({'type':'other','model':'adore','shapes':[[640,1136]]},'adore'),
            ({'type':'profile-check','model':'adore','shapes':[[640,1136]]*9},'adore'),
        ]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    parse_profile_check(json.dumps(data),selected,['adore'])

    async def test_profile_check_uses_renderer_cache_lookup_without_rendering(self):
        calls=[]
        def check(model,shapes):
            calls.append((model,shapes))
            return [(480,852)]
        renderer=SimpleNamespace(backend='tensorrt',info={'models':['adore']},missing_profiles=check)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result=await tensorrt_profile_result(renderer,executor,
                json.dumps({'type':'profile-check','model':'adore','shapes':[[640,1136],[480,852]]}),'adore')
        self.assertEqual(result,{'profileCheck':True,'supported':True,'available':False,'model':'adore',
                                 'shapes':[[640,1136],[480,852]],'missingShapes':[[480,852]]})
        self.assertEqual(calls,[('adore',[(640,1136),(480,852)])])

    def test_renderer_checks_the_exact_nchw_profile_for_the_requested_model(self):
        calls=[]
        renderer=object.__new__(Renderer)
        renderer.backend='tensorrt'
        renderer.select_model=lambda name:setattr(renderer,'selected',name)
        renderer.model=SimpleNamespace(backend=SimpleNamespace(model=SimpleNamespace(
            missing_cached_profiles=lambda shapes:calls.append(shapes) or [])))
        self.assertEqual(renderer.missing_profiles('adore',[(640,1136),(480,852)]),[])
        self.assertEqual(renderer.selected,'adore')
        self.assertEqual(calls,[[(1,3,1136,640),(1,3,852,480)]])


if __name__=='__main__': unittest.main()
