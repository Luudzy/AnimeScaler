"""Separate one-variable SPAB identity-skip smoke; original gates unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from torch import nn
from zenkai_model import ZenkaiLite
import zenkai_experiments as trainer


class SkipBlock(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core
    def forward(self,value):return self.core(value)+value


class IdentitySkipProbe(ZenkaiLite):
    def __init__(self):
        super().__init__(32,5)
        self.blocks=nn.ModuleList(SkipBlock(block) for block in self.blocks)
        self.tail=SkipBlock(self.tail)
        self.config.update(family='ZenkaiIdentitySkipProbe',spabIdentitySkipGain=1.)


def main():
    original_protocol=trainer.protocol_config
    def build(kind):
        if kind!='zenkai-a':raise ValueError('Identity skip probe uses A only')
        return IdentitySkipProbe()
    def protocol(source,kind,run,steps,loss):
        if run!='zenkai-ablation-spab-identity-skip-1k' or steps!=1000 or loss!='zenkai_frequency':
            raise ValueError('Fixed1k frequency-loss identity skip probe only')
        result=original_protocol(source,kind,run,steps,loss)
        result.update(family='ZenkaiIdentitySkipProbe',spabIdentitySkipGain=1.,
            comparisonRun='zenkai-ablation-frequency-1k',
            singleChangedVariable='SPAB identity bypass gain0 versus1, existing centered gate unchanged',
            wrapperSha256=trainer._sha(Path(__file__)),exportStatus='Ordinary exporter intentionally rejects probe family')
        return result
    trainer.build_experiment_model=build
    trainer.protocol_config=protocol
    trainer.main()


if __name__=='__main__':main()
