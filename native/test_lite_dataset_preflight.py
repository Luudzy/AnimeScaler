import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from lite_online_dataset import OnlineAnimeDataset


class OnlineDatasetPreflightTests(unittest.TestCase):
    def make_dataset(self, root, size=(64,64), digest_override=None, paired_size=None):
        image_path=root/'hr.png'
        Image.fromarray(np.zeros((size[1],size[0],3),dtype=np.uint8)).save(image_path)
        digest=hashlib.sha256(image_path.read_bytes()).hexdigest()
        row={'file':str(image_path),'split':'train','sha256':digest_override or digest}
        if paired_size:
            pair_path=root/'lr.png'
            Image.fromarray(np.zeros((paired_size[1],paired_size[0],3),dtype=np.uint8)).save(pair_path)
            row['pairedLr']=str(pair_path)
        (root/'manifest.jsonl').write_text(json.dumps(row))

    def test_invalid_degradation_fractions_fail_at_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.make_dataset(root)
            with self.assertRaisesRegex(ValueError,'fraction'):
                OnlineAnimeDataset(root,hard_fraction=1.1)
            with self.assertRaisesRegex(ValueError,'fraction'):
                OnlineAnimeDataset(root,codec_fraction=-0.1)

    def test_source_hash_mismatch_fails_before_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.make_dataset(root,digest_override='0'*64)
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                OnlineAnimeDataset(root,patch_lr=16)

    def test_patch_larger_than_source_fails_at_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.make_dataset(root)
            with self.assertRaisesRegex(ValueError,'too small'):
                OnlineAnimeDataset(root,patch_lr=33)

    def test_fixed_pair_geometry_mismatch_fails_at_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.make_dataset(root,paired_size=(31,32))
            with self.assertRaisesRegex(ValueError,'Paired LR geometry'):
                OnlineAnimeDataset(root,patch_lr=16)


    def make_split_dataset(self, root, holdout_group='holdout-clip', holdout_split='holdout', holdout_hash=None):
        train_image = root / 'train.png'
        Image.new('RGB', (64, 64), (30, 60, 90)).save(train_image)
        train_hash = hashlib.sha256(train_image.read_bytes()).hexdigest()
        train = {'file': str(train_image), 'sha256': train_hash, 'split': 'train', 'splitGroup': 'train-clip'}
        holdout = {'file': str(root / 'missing-holdout.png'), 'sha256': holdout_hash or 'a' * 64,
                   'split': holdout_split, 'splitGroup': holdout_group}
        train_bytes = (json.dumps(train) + '\n').encode('utf-8')
        holdout_bytes = (json.dumps(holdout) + '\n').encode('utf-8')
        (root / 'manifest.jsonl').write_bytes(train_bytes)
        (root / 'holdout.jsonl').write_bytes(holdout_bytes)
        metadata = {'state': 'complete', 'trainPairs': 1, 'holdoutPairs': 1,
                    'sourceSha256': hashlib.sha256(train_bytes).hexdigest(),
                    'holdoutSha256': hashlib.sha256(holdout_bytes).hexdigest(),
                    'splitStrategy': 'whole source clips; frames from one clip never cross splits',
                    'sourceGroups': 2}
        (root / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        return train, holdout

    def test_source_group_holdout_contract_passes_without_opening_holdout_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_split_dataset(root)
            dataset = OnlineAnimeDataset(root, patch_lr=16, codec_fraction=0)
            self.assertEqual(len(dataset), 1)
            self.assertIsNotNone(dataset.holdout_manifest_sha256)
            self.assertFalse((root / 'missing-holdout.png').exists())

    def test_source_group_leakage_fails_before_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_split_dataset(root, holdout_group='train-clip')
            with self.assertRaisesRegex(ValueError, 'splitGroup is shared'):
                OnlineAnimeDataset(root, patch_lr=16, codec_fraction=0)

    def test_wrong_holdout_label_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_split_dataset(root, holdout_split='train')
            with self.assertRaisesRegex(ValueError, 'wrong split label'):
                OnlineAnimeDataset(root, patch_lr=16, codec_fraction=0)

    def test_duplicate_image_hash_across_splits_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_split_dataset(root)
            train = json.loads((root / 'manifest.jsonl').read_text().splitlines()[0])
            holdout = json.loads((root / 'holdout.jsonl').read_text().splitlines()[0])
            holdout['sha256'] = train['sha256']
            holdout_bytes = (json.dumps(holdout) + '\n').encode('utf-8')
            (root / 'holdout.jsonl').write_bytes(holdout_bytes)
            metadata = json.loads((root / 'metadata.json').read_text())
            metadata['holdoutSha256'] = hashlib.sha256(holdout_bytes).hexdigest()
            (root / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'file SHA-256 is shared'):
                OnlineAnimeDataset(root, patch_lr=16, codec_fraction=0)

    def test_changed_manifest_is_rejected_before_image_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_split_dataset(root)
            (root / 'manifest.jsonl').write_text((root / 'manifest.jsonl').read_text() + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Training manifest hash differs'):
                OnlineAnimeDataset(root, patch_lr=16, codec_fraction=0)


if __name__=='__main__': unittest.main()
