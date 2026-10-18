import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from zenkai_validation_manifest import write_validation_manifest
from prepare_zenkai_codec import prepare_codec_manifest, render_codec_sample
from zenkai_evaluation import ZenkaiFixedEvaluator


class FixedCodecTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'native/.tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe').is_file(), 'install the local ffmpeg tool to run codec integration test')
    def test_codec_cache_is_fixed_validation_only_and_detects_pixel_change(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            folder=Path(temporary);source=folder/'anime.png';Image.new('RGB',(64,64),'gray').save(source)
            dataset=folder/'dataset';dataset.mkdir()
            row={'file':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'split':'validation','domain':'anime','cluster':'c1'}
            hidden={**row,'split':'final-hidden','file':str(folder/'must-not-be-opened.png')}
            (dataset/'manifest.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(hidden)+'\n')
            fixed=folder/'fixed.json';write_validation_manifest(dataset,fixed,allowed_roots=[folder],patch_lr=16)
            output=folder/'codec'
            report=prepare_codec_manifest(fixed,output,count=1,codecs=('h264',),allowed_roots=[folder])
            self.assertEqual(len(report['samples']),1)
            sample=report['samples'][0]
            low,high=render_codec_sample(sample,output,allowed_roots=[folder])
            self.assertEqual(low.size,(16,16));self.assertEqual(high.size,(32,32))
            self.assertEqual(report['realVideoValidation'],'REAL VIDEO VALIDATION NOT EXECUTED')
            evaluator=ZenkaiFixedEvaluator(fixed,per_case=1,allowed_roots=[folder])
            evaluator.attach_codec(output/'complete.json')
            import torch
            result=evaluator.evaluate({'model':torch.nn.Upsample(scale_factor=2,mode='nearest')},device='cpu')
            self.assertEqual(result['sampleCounts']['codec-intra-h264'],1)
            self.assertEqual(result['codecManifestSha256'],hashlib.sha256((output/'complete.json').read_bytes()).hexdigest())
            bad=folder/'bad.json';bad.write_text(json.dumps({**report,'fixedManifestSha256':'changed'}))
            with self.assertRaises(ValueError):evaluator.attach_codec(bad)
            with self.assertRaises(FileExistsError):prepare_codec_manifest(fixed,output,allowed_roots=[folder])
            Image.new('RGB',(16,16),'red').save(output/sample['lrFile'])
            with self.assertRaises(ValueError):render_codec_sample(sample,output,allowed_roots=[folder])
            with self.assertRaises(ValueError):render_codec_sample({**sample,'lrFile':'../anime.png'},output,allowed_roots=[folder])


if __name__=='__main__':unittest.main()
