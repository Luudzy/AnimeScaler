"""Single-variable output-residual gain probe, .2 versus original gain1.0.

This does not change SPAB's internal residual or gate. It is a separate
family, intentionally rejected by the ordinary Zenkai export loader.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch.nn.functional as F
from zenkai_model import ZenkaiLite
import zenkai_experiments as trainer


class ResidualScaleProbe(ZenkaiLite):
    def __init__(self):
        super().__init__(32,5)
        self.output_residual_scale=.2
        self.config.update(family='ZenkaiResidualScaleProbe',outputResidualScale=.2)

    def forward(self,value):
        prediction=super().forward(value)
        base=F.interpolate(value,scale_factor=2,mode='bilinear',align_corners=False)
        return base+self.output_residual_scale*(prediction-base)


def main():
    original_protocol=trainer.protocol_config
    def build(kind):
        if kind!='zenkai-a':raise ValueError('Output residual probe uses A architecture only')
        return ResidualScaleProbe()
    def protocol(source,kind,run,steps,loss):
        if run!='zenkai-ablation-output-residual-02-1k' or steps!=1000 or loss!='zenkai_frequency':
            raise ValueError('Fixed1k frequency-loss output residual probe only')
        result=original_protocol(source,kind,run,steps,loss)
        result.update(family='ZenkaiResidualScaleProbe',outputResidualScale=.2,
            comparisonRun='zenkai-ablation-frequency-1k',
            singleChangedVariable='Output learned residual multiplier .2 versus1.0; SPAB unchanged',
            wrapperSha256=trainer._sha(Path(__file__)),exportStatus='Ordinary exporter intentionally rejects probe family')
        return result
    trainer.build_experiment_model=build
    trainer.protocol_config=protocol
    trainer.main()


if __name__=='__main__':main()
