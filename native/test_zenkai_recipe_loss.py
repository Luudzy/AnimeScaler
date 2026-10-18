import copy
import subprocess
import sys
from pathlib import Path
import unittest
import torch
import zenkai_experiments as experiments
from lite_losses import LOSS_WEIGHTS,loss_terms


class RecipeLossTests(unittest.TestCase):
    def test_skip_recipe_entrypoint_bootstraps_isolated_python(self):
        script=Path(__file__).with_name('zenkai_skip_recipe.py')
        code="import sys,runpy; script=sys.argv[1]; sys.path=[p for p in sys.path if p and p!=sys.argv[2]]; sys.argv=[script,'--help']; runpy.run_path(script,run_name='__main__')"
        result=subprocess.run([sys.executable,'-c',code,str(script.resolve()),str(script.parent.resolve())],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_reserved_loss_pair_only_accepts_exact_run_and_budget(self):
        from zenkai_recipe_loss import paired_recipe_reference
        run='zenkai-ablation-spab-identity-skip-legacy-loss-1k'
        self.assertEqual(paired_recipe_reference('zenkai-a',run,1000,1000,'zenkai_legacy_v3'),
            'zenkai-ablation-spab-identity-skip-1k')
        for args in [('zenkai-b',run,1000,1000),('zenkai-a',run,3000,1000),
                     ('zenkai-a',run,1000,500),('zenkai-a','zenkai-ablation-arbitrary',1000,1000)]:
            with self.assertRaises(ValueError):paired_recipe_reference(*args,'zenkai_legacy_v3')

    def test_historical_recipe_matches_values_and_gradients(self):
        self.assertTrue(hasattr(experiments,'make_experiment_loss'))
        torch.manual_seed(31)
        x=torch.rand(1,3,17,19,requires_grad=True);target=torch.rand_like(x)
        terms=loss_terms(x,target);expected=sum(LOSS_WEIGHTS[k]*v for k,v in terms.items())
        grad=torch.autograd.grad(expected,x)[0]
        criterion=experiments.make_experiment_loss('zenkai_legacy_v3')
        actual,values=criterion(x,target)
        self.assertEqual(criterion.weights,LOSS_WEIGHTS)
        self.assertEqual(set(values),set(terms))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual,x)[0],grad,rtol=0,atol=0)

    def test_matched_start_rejects_rng_sampler_and_weight_changes(self):
        self.assertTrue(hasattr(experiments,'validate_paired_initial_state'))
        fields=('model','optimizer','scaler','rng','datasetRng','samplerIndices','nextSampleOffset','scheduler','step')
        base={key:torch.tensor([1,2]) for key in fields}
        base['step']=0;base['config']={}
        experiments.validate_paired_initial_state(base,copy.deepcopy(base))
        for key in ('model','rng','samplerIndices','datasetRng','optimizer'):
            changed=copy.deepcopy(base);changed[key][0]=3
            with self.assertRaisesRegex(ValueError,key):experiments.validate_paired_initial_state(base,changed)


if __name__=='__main__':unittest.main()
