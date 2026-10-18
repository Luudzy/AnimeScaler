import unittest

from training_amp import amp_optimizer_step


class FakeOptimizer:
    def __init__(self): self.steps=0
    def step(self): self.steps+=1


class FakeScaler:
    def __init__(self,optimizer,scale_after): self.optimizer=optimizer;self.scale=8.0;self.scale_after=scale_after
    def get_scale(self): return self.scale
    def step(self,optimizer):
        if self.scale_after>=self.scale: optimizer.step()
    def update(self): self.scale=self.scale_after


class AmpOptimizerTests(unittest.TestCase):
    def test_skipped_scaler_step_is_not_counted_as_optimizer_update(self):
        optimizer=FakeOptimizer();scaler=FakeScaler(optimizer,4.0)
        self.assertFalse(amp_optimizer_step(scaler,optimizer))
        self.assertEqual(optimizer.steps,0)

    def test_successful_scaler_step_is_counted_as_optimizer_update(self):
        optimizer=FakeOptimizer();scaler=FakeScaler(optimizer,8.0)
        self.assertTrue(amp_optimizer_step(scaler,optimizer))
        self.assertEqual(optimizer.steps,1)


if __name__=='__main__': unittest.main()
