import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from lite_evaluation import load_patches, validate_codec_cache


class DatasetTests(unittest.TestCase):
    def test_online_dataset_can_reuse_an_explicit_external_codec_cache(self):
        import torch,random
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from lite_online_dataset import OnlineAnimeDataset
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'shared-codec';cache.mkdir()
            source=root/'anime.png';Image.new('RGB',(256,256),(120,80,40)).save(source)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            (root/'manifest.jsonl').write_text(json.dumps({'file':str(source),'sha256':digest,'split':'train'})+'\n')
            Image.new('RGB',(128,128),(90,60,30)).save(cache/f'{digest}.png')
            (cache/f'{digest}.json').write_text(json.dumps({'sourceSha256':digest,'scale':2,'lrSize':[128,128]}))
            dataset=OnlineAnimeDataset(root,patch_lr=64,seed=1,hard_fraction=0,codec_fraction=1,codec_cache=cache)
            low,high=dataset[0]
            self.assertEqual(tuple(low.shape),(3,64,64));self.assertEqual(tuple(high.shape),(3,128,128))
            self.assertTrue(torch.allclose(low[0].mean(),torch.tensor(90/255),atol=.002))
            self.assertEqual(dataset.codec_samples,1)

    def test_explicit_dataset_excludes_training_and_hidden_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for split, value in [('train', 25), ('validation', 100), ('final-hidden', 240)]:
                path = root / f'{split}.png'
                Image.new('RGB', (256, 256), (value, value, value)).save(path)
                rows.append({'file': str(path), 'split': split, 'domain': 'anime'})
            (root / 'manifest.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            patches = load_patches(dataset=root)
            self.assertEqual(len(patches), 1)
            self.assertEqual(patches[0].shape, (192, 192, 3))
            self.assertTrue((patches[0] == 100).all())

    def test_codec_cache_rejects_same_size_different_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            patch = np.zeros((192, 192, 3), dtype=np.uint8)
            record = {'index': 0, 'targetSha256': hashlib.sha256(patch.tobytes()).hexdigest()}
            (root / 'complete.json').write_text(json.dumps({'frames': [record]}))
            validate_codec_cache(root, [patch])
            with self.assertRaisesRegex(ValueError, 'target mismatch'):
                validate_codec_cache(root, [patch + 1])
            with self.assertRaisesRegex(ValueError, 'counts differ'):
                validate_codec_cache(root, [])


if __name__ == '__main__':
    unittest.main()
