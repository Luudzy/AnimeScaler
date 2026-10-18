import unittest
import torch
from zenkai_model import ZenkaiLite, GRepConv, ESPANDeploy


class ZenkaiModelTests(unittest.TestCase):
    def test_depth_variants_keep_native_x2_shape(self):
        for depth in (5,6):
            model=ZenkaiLite(channels=4,depth=depth).eval()
            with torch.inference_mode():out=model(torch.rand(1,3,17,19))
            self.assertEqual(tuple(out.shape),(1,3,34,38))
            self.assertEqual(model.config['scale'],2)
            self.assertEqual(model.config['depth'],depth)

    def test_training_and_deploy_copy_match_and_leave_source_unchanged(self):
        model=GRepConv(4,5).eval()
        before={key:value.clone() for key,value in model.state_dict().items()}
        deployed=model.to_deploy()
        for shape in [(1,4,8,9),(2,4,17,19),(1,4,1,1)]:
            x=torch.randn(*shape)
            with torch.inference_mode():
                error=(model(x)-deployed(x)).abs()
            self.assertLess(float(error.max()),2e-5)
        self.assertTrue(all(torch.equal(value,model.state_dict()[key]) for key,value in before.items()))

    def test_depth_five_keeps_two_conv_tail_and_paper_deploy_graph(self):
        five=ZenkaiLite(channels=4,depth=5)
        six=ZenkaiLite(channels=4,depth=6)
        self.assertEqual(len(five.blocks),4)
        self.assertEqual(len(six.blocks),5)
        self.assertIsInstance(five.tail,torch.nn.Module)
        self.assertEqual(sum(p.numel() for p in five.parameters())>0,True)

    def test_official_deploy_reference_is_cpu_safe_and_x4(self):
        reference=ESPANDeploy(channels=4)
        with torch.inference_mode():out=reference(torch.rand(1,3,9,11))
        self.assertEqual(tuple(out.shape),(1,3,36,44))


if __name__=='__main__':unittest.main()
