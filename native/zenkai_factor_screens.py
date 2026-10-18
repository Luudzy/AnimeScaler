"""Reserved single-factor1000-step identity-skip experiments; no product changes."""
from pathlib import Path
import sys,copy,json,collections
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from artifact_io import atomic_json_write
import zenkai_experiments as trainer
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_recipe_loss import validate_paired_initial_state,_equal

REFERENCE='zenkai-ablation-spab-identity-skip-1k'
FACTOR_SPECS={'dataset-sampling':{},'crop96':{'patchLR':96,'patchHR':192},
              'codec50':{'codecFraction':.5},'scheduler-constant':{'learningRate':[.0002,.0002]}}
SCHEDULE_KIND='same warmup300 then constant2e-4'

def parse_factor_args(arguments=None):
    import argparse
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--factor',choices=tuple(FACTOR_SPECS),required=True)
    parser.add_argument('--interval',type=int,action='append')
    options,remaining=parser.parse_known_args(arguments)
    if any(value!=1000 for value in options.interval or []):
        raise ValueError('Factor validation interval must1000')
    return options.factor,[*remaining,'--interval','1000']

def validate_factor_config(config,reference):
    factor=config.get('factorName')
    if factor not in FACTOR_SPECS:raise ValueError('Unreviewed factor')
    adjusted=copy.deepcopy(config)
    for key,value in FACTOR_SPECS[factor].items():
        _equal(config.get(key),value,'intended '+key)
        adjusted[key]=reference[key]
    ignored={'run','name','architecture','comparisonRun','singleChangedVariable','wrapperSha256','exportStatus',
             'sourceCodeSha256','sourceFilesSha256','gitWorkingTree'}
    for key,value in reference.items():
        if key not in ignored:_equal(adjusted.get(key),value,'config.'+key)
    policy='uniform heuristic group then uniform image' if factor=='dataset-sampling' else None
    schedule=SCHEDULE_KIND if factor=='scheduler-constant' else None
    if config.get('samplingPolicy')!=policy:raise ValueError('Undeclared sampling policy')
    if config.get('schedulerExperiment')!=schedule:raise ValueError('Undeclared scheduler change')

def validate_factor_initial(actual,reference,factor):
    if factor not in FACTOR_SPECS:raise ValueError('Unreviewed factor')
    for config in (actual['config'],reference['config']):
        if config.get('loss')!='zenkai_frequency' or config.get('family')!='ZenkaiIdentitySkipProbe':
            raise ValueError('Factor pairs require frequency-loss identity-skip')
    adjusted=copy.deepcopy(actual)
    for key,value in FACTOR_SPECS[factor].items():
        _equal(actual['config'].get(key),value,'intended '+key)
        adjusted['config'][key]=reference['config'][key]
    ignored={'run','name','architecture','comparisonRun','singleChangedVariable','wrapperSha256','exportStatus',
             'sourceCodeSha256','sourceFilesSha256','gitWorkingTree'}
    for key,value in reference['config'].items():
        if key not in ignored:_equal(adjusted['config'].get(key),value,'config.'+key)
    exemptions=[]
    if factor=='dataset-sampling':
        adjusted['samplerIndices']=reference['samplerIndices'];exemptions.append('samplerIndices')
    if factor=='scheduler-constant':
        if not isinstance(actual['scheduler'],dict) or actual['scheduler'].get('kind')!=SCHEDULE_KIND:
            raise ValueError('Wrong constant scheduler descriptor')
        if isinstance(reference['scheduler'],dict):
            for key,value in reference['scheduler'].items():
                if key!='kind':_equal(actual['scheduler'].get(key),value,'scheduler.'+key)
        adjusted['scheduler']=reference['scheduler'];exemptions.append('scheduler.kind')
    matched=validate_paired_initial_state(adjusted,reference)
    return {**matched,'factor':factor,'configExemptions':list(FACTOR_SPECS[factor]),
            'stateExemptions':exemptions,'scope':'Only declared factor exempt; all other initialized state matches'}

def main():
    # Remove only our own argument before forwarding the unchanged trainer CLI.
    factor,remaining=parse_factor_args()
    sys.argv=[sys.argv[0],*remaining]
    original_protocol=trainer.protocol_config;original_snapshot=trainer.state_snapshot;original_lr=trainer.learning_rate
    run='zenkai-ablation-skip-factor-'+factor+'-1k'
    def build(kind):
        if kind!='zenkai-a':raise ValueError('Factor screens use same A skip architecture')
        return IdentitySkipProbe()
    def protocol(source,kind,name,steps,loss):
        if (kind,name,steps,loss)!=('zenkai-a',run,1000,'zenkai_frequency'):
            raise ValueError('Reserved1000-step single-factor run only')
        result=original_protocol(source,kind,name,steps,loss)
        result.update(family='ZenkaiIdentitySkipProbe',spabIdentitySkipGain=1.,comparisonRun=REFERENCE,
            factorName=factor,singleChangedVariable=factor,wrapperSha256=trainer._sha(Path(__file__)),
            exportStatus='Separate probe family, no ordinary export')
        result.update(FACTOR_SPECS[factor])
        if factor=='dataset-sampling':result['samplingPolicy']='uniform heuristic group then uniform image'
        if factor=='scheduler-constant':result['schedulerExperiment']=SCHEDULE_KIND
        return result
    def snapshot(model,optimizer,scaler,step,config,dataset_rng,sampler_indices):
        config['sourceFilesSha256'].update({name:trainer._sha(Path(__file__).with_name(name)) for name in ('zenkai_factor_screens.py','zenkai_ablation_skip.py')})
        actual=original_snapshot(model,optimizer,scaler,step,config,dataset_rng,sampler_indices)
        if factor=='scheduler-constant':actual['scheduler']['kind']=SCHEDULE_KIND
        if step==0:
            output=trainer.ROOT/'models/zenkai/experiments'/run
            atomic_json_write(config,output/'config.json')
            paired=trainer.ROOT/'models/zenkai/experiments'/REFERENCE/'initial-training-state.pt'
            pins=json.loads((paired.parent/'metadata.json').read_text())['checkpointsSha256']
            digest=pins[paired.name]
            if trainer._sha(paired)!=digest:raise ValueError('Reference initialization hash changed')
            reference=torch.load(paired,map_location='cpu',weights_only=True)
            if trainer._sha(Path(__file__).with_name('zenkai_ablation_skip.py'))!=reference['config']['wrapperSha256']:
                raise ValueError('Identity-skip forward wrapper changed')
            for name,expected in reference['config']['sourceFilesSha256'].items():
                if name not in ('zenkai_experiments.py','zenkai_recipe_loss.py') and trainer._sha(Path(__file__).with_name(name))!=expected:
                    raise ValueError('Shared dependency changed:'+name)
            pairing=validate_factor_initial(actual,reference,factor)
            pairing.update(reference=str(paired),referenceSha256=digest)
            atomic_json_write(pairing,output/'initial-pairing.json')
            source=output/'source';source.mkdir()
            for name,digest in config['sourceFilesSha256'].items():
                content=Path(__file__).with_name(name).read_bytes()
                if trainer._sha(Path(__file__).with_name(name))!=digest:raise ValueError('Source changed during snapshot')
                (source/name).write_bytes(content)
            print(json.dumps({'initialPairing':'matched except declared factor','factor':factor,'referenceSha256':pairing['referenceSha256']}),flush=True)
        return actual
    if factor=='dataset-sampling':
        def sampler(data,replacement,num_samples,generator):
            counts=collections.Counter(row['cluster'] for row in data.rows)
            weights=[1./counts[row['cluster']] for row in data.rows]
            return torch.utils.data.WeightedRandomSampler(weights,num_samples,replacement=True,generator=generator)
        trainer.RandomSampler=sampler
    if factor=='scheduler-constant':trainer.learning_rate=lambda step,total:original_lr(step,total) if step<=300 else 2e-4
    trainer.build_experiment_model=build;trainer.protocol_config=protocol;trainer.state_snapshot=snapshot
    trainer.main()

if __name__=='__main__':main()
