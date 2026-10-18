import hashlib,json,tempfile,unittest
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
if importlib.util.find_spec('pytorch_msssim') is None:
    raise unittest.SkipTest('pytorch-msssim is required for Lite V3.1 loss tests')
from lite_v31 import AdoreLiteV31,receptive_field,VARIANTS
from lite_online_dataset import OnlineAnimeDataset
from lite_v31_losses import reconstruction_loss

class V31Tests(unittest.TestCase):
    def test_codec_counter_counts_only_samples_that_use_paired_lr(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);high=np.random.default_rng(17).integers(0,256,(64,64,3),dtype=np.uint8)
            high_path=root/'high.png';Image.fromarray(high).save(high_path)
            low=np.full((32,32,3),(11,37,83),dtype=np.uint8);low_path=root/'low.png';Image.fromarray(low).save(low_path)
            digest=hashlib.sha256(high_path.read_bytes()).hexdigest()
            (root/'manifest.jsonl').write_text(json.dumps({'file':str(high_path),'split':'train','sha256':digest,'pairedLr':str(low_path)}))
            paired=OnlineAnimeDataset(root,patch_lr=16,seed=5,hard_fraction=0,codec_fraction=0,augment_paired=False)
            paired_x,_=paired[0]
            expected=np.broadcast_to(np.asarray((11,37,83),dtype=np.uint8),(16,16,3)).copy()
            self.assertTrue(torch.equal(paired_x,torch.tensor(expected.transpose(2,0,1),dtype=torch.float32)/255))
            self.assertEqual(paired.codec_samples,1)
            augmented=OnlineAnimeDataset(root,patch_lr=16,seed=5,hard_fraction=0,codec_fraction=0,augment_paired=True)
            augmented_x,_=augmented[0]
            self.assertFalse(torch.equal(augmented_x,paired_x))
            self.assertEqual(augmented.codec_samples,0)
            self.assertEqual(augmented.samples,1)

    def test_unique_crop_counter_has_fixed_memory_and_labeled_estimate(self):
        from lite_online_dataset import BoundedUniqueCropCounter
        counter=BoundedUniqueCropCounter(precision=10)
        for index in range(10000):
            counter.add(('image',index,0,192))
        for index in range(1000):
            counter.add(('image',index,0,192))
        self.assertEqual(counter.memory_bytes,1024)
        self.assertEqual(counter.method,'hyperloglog-p10')
        self.assertAlmostEqual(len(counter),10000,delta=1000)
        self.assertEqual(counter.report_fields('crops')['cropsCountType'],'estimate')
        self.assertEqual(counter.report_fields('crops')['cropsCounterMemoryBytes'],1024)

    def test_architecture_and_receptive_field(self):
        self.assertEqual(receptive_field([1]*4),23)
        for variant,rf,count in [('a',31,70980),('b',31,124588),('c',39,124588)]:
            model=AdoreLiteV31(variant)
            self.assertEqual(receptive_field(VARIANTS[variant][1]),rf)
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            x=torch.rand(1,3,16,20)
            self.assertTrue(torch.equal(model(x),F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
    def test_online_crops_are_diverse_and_codec_aligned(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'codec-full').mkdir()
            low=np.random.default_rng(3).integers(0,256,(160,160,3),dtype=np.uint8)
            high=low.repeat(2,0).repeat(2,1);path=root/'high.png';Image.fromarray(high).save(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'manifest.jsonl').write_text(json.dumps({'file':str(path),'split':'train','sha256':digest}))
            stem=root/'codec-full'/digest;Image.fromarray(low).save(stem.with_suffix('.png'))
            stem.with_suffix('.json').write_text(json.dumps({'sourceSha256':digest,'scale':2,'lrSize':[160,160]}))
            data=OnlineAnimeDataset(root,patch_lr=96,codec_fraction=1)
            for _ in range(8):
                x,y=data[0];self.assertTrue(torch.equal(x.repeat_interleave(2,-2).repeat_interleave(2,-1),y))
            self.assertGreater(len(data.unique_crops),1)
    def test_detail_loss_has_finite_gradients(self):
        pred=torch.rand(1,3,192,192,requires_grad=True);target=torch.rand_like(pred)
        loss,terms=reconstruction_loss(pred,target,edge_weight=.1)
        loss.backward();self.assertTrue(torch.isfinite(pred.grad).all());self.assertIn('ms_ssim',terms)

if __name__=='__main__':unittest.main()
