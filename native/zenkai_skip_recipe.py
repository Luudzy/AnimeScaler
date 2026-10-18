"""Matched legacy-loss experiment inside the unchanged identity-skip model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import zenkai_experiments as trainer
from zenkai_ablation_skip import IdentitySkipProbe

RUN='zenkai-ablation-spab-identity-skip-legacy-loss-1k'
REFERENCE='zenkai-ablation-spab-identity-skip-1k'

def main():
    original_protocol=trainer.protocol_config
    def build(kind):
        if kind!='zenkai-a':raise ValueError('Skip loss pair uses A only')
        return IdentitySkipProbe()
    def protocol(source,kind,run,steps,loss):
        if (kind,run,steps,loss)!=('zenkai-a',RUN,1000,'zenkai_legacy_v3'):
            raise ValueError('Reserved1000-step skip legacy loss pair only')
        result=original_protocol(source,kind,run,steps,loss)
        result.update(family='ZenkaiIdentitySkipProbe',spabIdentitySkipGain=1.,comparisonRun=REFERENCE,
            singleChangedVariable='Complete loss profile within identical identity-skip architecture',
            wrapperSha256=trainer._sha(Path(__file__)),exportStatus='Ordinary exporter intentionally rejects probe family')
        return result
    trainer.build_experiment_model=build
    trainer.protocol_config=protocol
    trainer.main()

if __name__=='__main__':main()
