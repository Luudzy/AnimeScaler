import copy,unittest
import torch
from zenkai_factor_screens import validate_factor_initial,FACTOR_SPECS
from zenkai_probe_evaluation import load_probe
from zenkai_ablation_skip import IdentitySkipProbe
from pathlib import Path
import tempfile
from unittest import mock

class FactorScreenTests(unittest.TestCase):
    def state(self):
        keys=('model','optimizer','scaler','rng','datasetRng','samplerIndices','nextSampleOffset','scheduler')
        s={k:torch.tensor([1,2]) for k in keys};s['step']=0
        s['config']={'family':'ZenkaiIdentitySkipProbe','patchLR':128,'patchHR':256,'codecFraction':.25,
            'learningRate':[.0002,.00001],'loss':'zenkai_frequency','spabIdentitySkipGain':1.,'scheduleTotalSteps':5000}
        return s
    def test_only_intended_initial_difference_is_allowed(self):
        for factor in FACTOR_SPECS:
            ref=self.state();actual=copy.deepcopy(ref)
            if factor=='dataset-sampling':actual['samplerIndices']=torch.tensor([7,8])
            if factor=='crop96':actual['config'].update(patchLR=96,patchHR=192)
            if factor=='codec50':actual['config']['codecFraction']=.5
            if factor=='scheduler-constant':
                actual['scheduler']={'kind':'same warmup300 then constant2e-4'}
                actual['config']['learningRate']=[.0002,.0002]
            validate_factor_initial(actual,ref,factor)
            bad=copy.deepcopy(actual);bad['model'][0]=9
            with self.assertRaises(ValueError):validate_factor_initial(bad,ref,factor)
            bad=copy.deepcopy(actual);bad['rng'][0]=9
            with self.assertRaises(ValueError):validate_factor_initial(bad,ref,factor)
    def test_factor_loader_requires_reserved_config(self):
        model=IdentitySkipProbe()
        reference=self.state()['config']
        reference.update(channels=32,depth=5,nativeScale=2)
        # Loader now checks the full frozen protocol, so a partial fixture no longer represents a valid run.
        with tempfile.TemporaryDirectory() as folder, mock.patch(
                'zenkai_probe_evaluation.load_factor_reference_config',return_value=reference):
            path=Path(folder)/'probe.pt'
            for factor,spec in FACTOR_SPECS.items():
                cfg={**reference,'family':'ZenkaiIdentitySkipProbe','channels':32,'depth':5,'nativeScale':2,
                     'spabIdentitySkipGain':1.,'run':'zenkai-ablation-skip-factor-'+factor+'-1k',
                     'factorName':factor,'loss':'zenkai_frequency',
                     'comparisonRun':'zenkai-ablation-spab-identity-skip-1k',**spec}
                if factor=='dataset-sampling':cfg['samplingPolicy']='uniform heuristic group then uniform image'
                if factor=='scheduler-constant':cfg['schedulerExperiment']='same warmup300 then constant2e-4'
                torch.save({'model':model.state_dict(),'config':cfg,'step':1000},path)
                self.assertEqual(load_probe(path)[1]['step'],1000)
                cfg['loss']='zenkai_baseline'
                torch.save({'model':model.state_dict(),'config':cfg,'step':1000},path)
                with self.assertRaises(ValueError):load_probe(path)

    def test_config_drift_outside_factor_rejected(self):
        ref=self.state();actual=copy.deepcopy(ref);actual['config']['loss']='zenkai_baseline'
        with self.assertRaises(ValueError):validate_factor_initial(actual,ref,'dataset-sampling')
        actual=copy.deepcopy(ref);actual['samplerIndices'][0]=9
        with self.assertRaises(ValueError):validate_factor_initial(actual,ref,'crop96')

    def test_interval_guard_parses_equals_and_repeated_options(self):
        from zenkai_factor_screens import parse_factor_args
        for options in (['--interval=500'], ['--interval','500','--interval','1000'],
                        ['--interval=1000','--interval=500']):
            with self.assertRaises(ValueError):
                parse_factor_args(['--factor','crop96',*options])
        for options in ([], ['--interval=1000'], ['--interval','1000']):
            factor, remaining = parse_factor_args(['--factor','crop96','--steps','1000',*options])
            self.assertEqual(factor,'crop96')
            self.assertEqual(remaining,['--steps','1000','--interval','1000'])

    def test_loader_rejects_second_factor_even_when_declared_factor_matches(self):
        reference = self.state()['config']
        reference.update(channels=32,depth=5,nativeScale=2,seed=929,batch=4,
                         accumulation=2,optimizer='AdamW',lossWeights={'charbonnier':1.,'hf':1.,'dct':.001})
        cfg=copy.deepcopy(reference)
        cfg.update(run='zenkai-ablation-skip-factor-crop96-1k',factorName='crop96',patchLR=96,patchHR=192,
                   comparisonRun='zenkai-ablation-spab-identity-skip-1k')
        with tempfile.TemporaryDirectory() as folder, mock.patch(
                'zenkai_probe_evaluation.load_factor_reference_config',return_value=reference,create=True):
            path=Path(folder)/'probe.pt'
            model=IdentitySkipProbe()
            for field,value in (('codecFraction',.5),('seed',930),('batch',8),('lossWeights',{'charbonnier':1.}),
                                ('samplingPolicy','uniform heuristic group then uniform image'),
                                ('schedulerExperiment','same warmup300 then constant2e-4')):
                bad=copy.deepcopy(cfg);bad[field]=value
                torch.save({'model':model.state_dict(),'config':bad,'step':1000},path)
                with self.assertRaises(ValueError,msg=field):load_probe(path)

if __name__=='__main__':unittest.main()
