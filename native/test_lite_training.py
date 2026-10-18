import random
import unittest
import numpy as np
from PIL import Image
from train_lite_v2 import pair, training_lr

class LiteTrainingTests(unittest.TestCase):
    def test_schedule(self):
        self.assertAlmostEqual(training_lr(0,10,1e-4,1e-5),1e-4)
        self.assertAlmostEqual(training_lr(9,10,1e-4,1e-5),1e-5)
        self.assertGreater(training_lr(3,10,1e-4,1e-5),training_lr(6,10,1e-4,1e-5))

    def test_robust_pairs_reproducible_and_bounded(self):
        image=Image.fromarray(np.random.default_rng(3).integers(0,256,(128,128,3),dtype=np.uint8))
        a=pair(image,random.Random(7),robust=True)
        b=pair(image,random.Random(7),robust=True)
        self.assertEqual(tuple(a[0].shape),(3,48,48))
        self.assertEqual(tuple(a[1].shape),(3,96,96))
        for x,y in zip(a,b):
            self.assertTrue(x.equal(y));self.assertGreaterEqual(x.min(),0);self.assertLessEqual(x.max(),1)

    def test_validation_never_receives_training_degradation(self):
        image=Image.new('RGB',(128,128),(100,100,100))
        a=pair(image,random.Random(7),augment=False,robust=True)
        b=pair(image,random.Random(7),augment=False,robust=False)
        for x,y in zip(a,b):self.assertTrue(x.equal(y))

if __name__=='__main__':unittest.main()
