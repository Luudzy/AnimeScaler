"""Retrospective same-timeline budget evidence from preserved skip1k?3k."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
from frozen_video_protocol import validate_protocol
R1='zenkai-ablation-spab-identity-skip-1k';R3='zenkai-ablation-spab-identity-skip-3k-continuation'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def table(h,rows):return ['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]
def main():
 f1=ROOT/'models/zenkai/experiments'/R1;f3=ROOT/'models/zenkai/experiments'/R3
 c1=read(f1/'config.json');c3=read(f3/'config.json');r3=read(f3/'report.json')
 assert c3['sourceRun']==R1 and sha(Path(c3['continuedFrom']))==c3['continuedFromSha256']
 for k in ('family','channels','depth','spabIdentitySkipGain','seed','optimizer','betas','weightDecay','learningRate','loss','lossWeights','patchLR','patchHR','batch','accumulation','codecFraction','scheduleTotalSteps','datasetManifestSha256','fixedValidationSha256','codecCacheCompleteSha256'):
  assert c1[k]==c3[k],k
 assert r3['stepsExecuted']==r3['optimizerUpdates']==3000
 q1=read('diagnostics/zenkai-ablations/skip-loss-1k-quality/quality.json');q3=read('diagnostics/zenkai-ablations/probes-3k-quality/quality.json')
 assert q1['indices']==q3['indices'] and q1['validationManifestSha256']==q3['validationManifestSha256']
 v1=read('diagnostics/zenkai-real-video-skip-frequency-1k/report.json');v3=read('diagnostics/zenkai-real-video-skip-3k-complete/report.json')
 validate_protocol(v1);validate_protocol(v3)
 lines=['# Zenkai: preserved SkipFrequency1k?3k budget evidence','',
 'No training repeated. Existing1000?3000 continuation restored optimizer/scaler/RNG/dataRNG/sampler with the same5000-step cosine timeline. This measures progression along one frozen training trajectory; it does not isolate budget from the LR evolution inherent in that timeline. It is not a5k versus historical25k budget-only comparison and not a license for15k.', '',
 'Same initialized architecture/seed/data/crop/codec/loss, same160 fixed inputs and144 video frames. The1k model was reevaluated during the loss pair; the3k evaluation is pinned from the previous run. Numeric gates passed then; small repeat/backend numerical variance is not a multi-seed uncertainty estimate.', '',
 '## Individual fixed metrics','','Errors/LPIPS lower;PSNR/SSIM/MS-SSIM higher.','']
 rows=[];deltas={}
 for domain,m in q3['cases'].items():
  deltas[domain]={}
  for metric,a in q1['cases'][domain]['skip-frequency-1k'].items():
   b=m['zenkai-identity-skip-3k'][metric];h=m['adore-lite-v3'][metric];d=(b/a-1)*100
   deltas[domain][metric]=d
   rows.append([domain,metric,f'{a:.9f}',f'{b:.9f}',f'{h:.9f}',f'{d:+.4f}',f'{(b/h-1)*100:+.4f}'])
 lines+=table(['Domain','Metric','Skip1k','Skip3k','HistoricalV3','3k/1k %','3k/V3 %'],rows)
 lines+=['','## Frozen real video','','CPU FP32; pixel temporal differences are not optical-flow or perceptual flicker tests.','']
 rows=[]
 for a,b in zip(v1['sequences'],v3['sequences']):
  for k in ('manifestSha256','frameCount','suspectedCutTransitions'):assert a[k]==b[k]
  for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
   x=a['models'][R1][metric];y=b['models'][R3][metric];h=a['models']['adore-lite-v3'][metric]
   assert abs(h-b['models']['adore-lite-v3'][metric])<1e-9
   rows.append([Path(a['manifest']).parent.name,metric,f'{x:.9f}',f'{y:.9f}',f'{h:.9f}',f'{(y/x-1)*100:+.4f}',f'{(y/h-1)*100:+.4f}'])
 lines+=table(['Clip','Metric','Skip1k','Skip3k','V3','3k/1k %','3k/V3 %'],rows)
 lines+=['','## Finding','',
 'Fixed MAE improved6.9234/2.5715/1.4986/1.8364/2.8421% clean/heavy/H264/H265/AV1 from1k to3k. This shows useful learning along the preserved timeline. At3k all five MAEs remain worse than HistoricalV3, and two video clips still regress against it. Reported LPIPS/SSIM/edge/temporal values remain visible above; no composite-only winner.', '',
 'Learning curves support further controlled investigation of training protocol, not assuming that any15k run wins. The300-step warmup is30% of1k and10% of3k; the separate scheduler screen is needed before attributing progress to optimizer update count alone. No C5k started by this analysis.', '',
 'Latency of Skip1k:NOT MEASURED. Skip3k previously measured19.105/70.980ms medians for360?720/720?1080 including resize, not playerFPS. Training budget does not reduce inference architecture cost. Source parity/performance in docs/zenkai-ablations.md.']
 (ROOT/'docs/zenkai-budget-analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 result={'reference':R1,'continuation':R3,'continuedFromSha256':c3['continuedFromSha256'],'configComparedKeys':list(k for k in c1 if k in c3 and c1[k]==c3[k]),'fixedSamples':160,'videoFramesEach':144,'fixedMetricDeltasPct':deltas,'scope':'Single preserved trajectory with frozen5k LR timeline; not independent-seed budget effect','releaseApproved':False}
 (ROOT/'diagnostics/zenkai-budget-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('Existing budget progression verified/reported; no new GPU work')
if __name__=='__main__':main()
