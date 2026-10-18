import tempfile
import unittest
from pathlib import Path
from PIL import Image
from generate_sr_comparison import write_comparison


class ComparisonTests(unittest.TestCase):
    def test_blind_files_use_labels_and_keep_a_separate_key(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            images={'secret-model':Image.new('RGB',(64,48),'gray'),'reference':Image.new('RGB',(64,48),'black')}
            key=write_comparison(images,folder,blind=True,seed=42)
            self.assertEqual(set(key.values()),set(images))
            self.assertFalse(any('secret-model' in p.name for p in folder.iterdir()))
            with Image.open(folder/'A-full.png') as image:self.assertEqual(image.size,(64,48))
            self.assertTrue((folder/'2x-crop.png').is_file())
            self.assertTrue((folder/'4x-crop.png').is_file())

    def test_comparison_rejects_mismatched_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'resolution'):
                write_comparison({'reference':Image.new('RGB',(32,32)),'bad':Image.new('RGB',(16,16))},Path(temp))


if __name__=='__main__':unittest.main()
