"""Exact legacy-loss adapter and initialized-state guard for one-variable probes."""
import torch
from torch import nn
from lite_losses import LOSS_WEIGHTS,loss_terms
from zenkai_losses import ZenkaiLoss,PRESETS

RECIPE_PRESETS=(*PRESETS,'zenkai_legacy_v3')

class LegacyV3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=dict(LOSS_WEIGHTS)

    def forward(self,pred,target):
        if pred.shape!=target.shape:raise ValueError('Loss input shape mismatch')
        with torch.autocast(device_type=pred.device.type,enabled=False):
            terms=loss_terms(pred.float(),target.float())
            return sum(self.weights[k]*v for k,v in terms.items()),terms

def make_experiment_loss(preset):
    return LegacyV3Loss() if preset=='zenkai_legacy_v3' else ZenkaiLoss(preset)

def _equal(a,b,path):
    if isinstance(a,torch.Tensor):
        same=isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        same=isinstance(b,dict) and a.keys()==b.keys()
        if same:
            for key in a:_equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        same=type(a)==type(b) and len(a)==len(b)
        if same:
            for i,(x,y) in enumerate(zip(a,b)):_equal(x,y,path+'.'+str(i))
    else:same=type(a)==type(b) and a==b
    if not same:raise ValueError('Paired initial state mismatch:'+path)

def validate_paired_initial_state(actual,reference):
    if actual['step']!=0 or reference['step']!=0:raise ValueError('Expected initialization, not resume')
    fields=('model','optimizer','scaler','rng','datasetRng','samplerIndices','nextSampleOffset','scheduler','step')
    for field in fields:_equal(actual[field],reference[field],field)
    for key in ('family','channels','depth','nativeScale','seed','optimizer','learningRate','betas','weightDecay',
        'batch','accumulation','effectiveBatch','patchLR','patchHR','codecFraction','codecCacheCompleteSha256',
        'datasetManifestSha256','fixedValidationSha256','legacyV3Sha256','torch','hip','scheduleTotalSteps'):
        if key in reference['config']:_equal(actual['config'].get(key),reference['config'][key],'config.'+key)
    return {'state':'matched','fields':list(fields),'scope':'Before first optimizer update; only loss recipe differs'}
