"""CPU metadata-only audit of separate protocol factors, never hidden pixels."""
import json,hashlib,math,collections,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def lr(step,total=5000):return (1e-5+(.0002-1e-5)*.5*(1+math.cos(math.pi*(step-1)/(total-1))))*min(1,step/300)

def main():
 rows=[json.loads(s) for s in (ROOT/'native/training-data/zenkai/manifest.jsonl').read_text().splitlines() if s]
 train=[r for r in rows if r['split']=='train']
 assert len(train)==1792 and all(r['domain']=='anime' for r in train)
 codec=[];missing=[]
 for row in train:
  p=ROOT/'native/training-data/v31/codec-full'/(row['sha256']+'.json')
  if not p.is_file():missing.append(row['sha256']);continue
  m=read(p)
  assert m['sourceSha256']==row['sha256'] and m['scale']==2
  codec.append(m)
 historical=read('models/legacy/adore-lite-v3/frozen-20260930/report.json')
 runs={}
 for run in ['zenkai-ablation-frequency-1k','zenkai-ablation-spab-identity-skip-1k','zenkai-ablation-frequency-3k-continuation','zenkai-ablation-spab-identity-skip-3k-continuation']:
  r=read('models/zenkai/experiments/'+run+'/report.json')
  runs[run]={k:r.get(k) for k in ('stepsExecuted','optimizerUpdates','samples','codecSamples','uniqueSpatialCrops','patchLR','batch','accumulation','scheduleTotalSteps')}
 groups=collections.Counter(r['cluster'] for r in train)
 sizes=collections.Counter(tuple(m['lrSize']) for m in codec)
 result={'scope':'CPU metadata/source audit only; no causal training ablations and no image pixels read',
  'updatedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'manifestSha256':sha('native/training-data/zenkai/manifest.jsonl'),
  'dataset':{'train':len(train),'domains':dict(collections.Counter(r['domain'] for r in train)),
   'heuristicGroups':len(groups),'largestGroups':groups.most_common(5),'sampling':'uniform image replacement, not uniform cluster',
   'historicalTrain':historical['dataset']['domainSplits'],'independence':'Heuristic clusters do not prove scene/episode independence; development was historical TRAIN'},
  'crop':{'lr':128,'hr':256,'hardFraction':.3,'hardCandidateCount':4,'coordinateAlignmentHR':2,
   'augmentation':'independent horizontal/vertical flips .5; hard crop chosen by grayscale finite-difference detail score',
   'historicalLR':historical['patchLR'],'historicalHR':historical['patchHR'],'historicalPatchesPerImage':historical['patchesPerImage'],
   'hrAreaRatio128vs96':(256/192)**2,'cacheLrSizes':dict((str(k),v) for k,v in sizes.items()),
   'nativeHR256EligibilityFromCodecMetadata':sum(min(m['lrSize'])*2>=256 for m in codec)},
  'codecSampling':{'configuredFraction':.25,'trainMetadataCovered':len(codec),'missing':missing,
   'codecs':dict(collections.Counter(m['codec'] for m in codec)), 'crfs':dict(collections.Counter(m['crf'] for m in codec)),
   'scopes':dict(collections.Counter(m['scope'] for m in codec)),
   'onlineDistribution':'One cached codec pair per source; codec branch probability.25, not necessarily equal per codec',
   'evaluation':'32 each H264/H265/AV1 intra; real video360p H264CRF24GOP48'},
  'trainingBudget':{'historicalSelectedStep':historical['selectedStep'],'historicalCumulativeSteps':historical['cumulativeExecutedSteps'],
   'screeningStepBudgets':[1000,3000,5000],'effectiveBatch':8,'runs':runs},
  'scheduler':{'function':'native/train_zenkai.learning_rate','totalSteps':5000,'warmup':300,
   'lrAtStep':{str(k):lr(k) for k in [1,100,300,1000,3000,5000]},
   'screeningWarmupFraction':{'1k':.3,'3k':.1,'5k':.06},
   'resumePolicy':'Stored RNG/optimizer and original schedule preserved; no rewriting completed schedule'}
 }
 (ROOT/'diagnostics/zenkai-protocol-factors.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 lines=['# Zenkai: separate protocol-factor investigation','',
 'CPU metadata/source audit completed. No image pixels or final-hidden opened. This records differences and hypotheses; it does not establish which difference causes the HistoricalV3 gap. No five-factor combined change and no C5k launched.', '',
 '## Dataset', '',
 f'Train:{len(train)} anime images in{len(groups)} heuristic groups; image-uniform replacement overweights large groups relative to a group-uniform sampler. HistoricalV3 trained on1984 anime+604 illustration; Zenkai uses1792 anime. Historical development exposure and heuristic grouping limit independence. Dataset causality:NOT MEASURED.', '',
 'Controlled next dataset test:freeze identical TRAIN identities and compare image-uniform with group-uniform sampling as one sampling policy factor, while preserving validation and all model/loss/crop/codec/budget/schedule choices. Do not restore hidden/development items or disallowed illustration paths just to mimic the historical counts.', '',
 '## Crop', '',
 'Current LR128/HR256,30% detail-biased selection among4 candidates,2-pixel HR alignment and independent flips. Historical metadata has LR96/HR192 and2 pre-generated patches/image. Area differs1.7778x; static/online policy also differs. These are separate differences, not evidence that larger crops are better.', '',
 'Controlled crop test:LR96 versus128 with same online/hard selection policy, effective batch,loss,dataset,codec fraction and1000-step budget. Record changed pixels per update and latency; do not silently compensate by changing batch/steps. Validation keeps the original LR128 inputs. A separate hard-fraction experiment would be a different test.', '',
 '## Codec sampling', '',
 f"TRAIN metadata coverage:{len(codec)}/{len(train)}. Codecs:{result['codecSampling']['codecs']}; CRFs:{result['codecSampling']['crfs']}; scopes:{result['codecSampling']['scopes']}.", '',
 'A Bernoulli.25 branch selects one cached pair per source; image sampling defines codec mix. Intra cached stills differ from real H264GOP48 inter-frame input. This mismatch is a hypothesis, not a proven cause of temporal errors. Controlled fraction test:.25 versus.50 on the same cache and image sampler. Do not also regenerate inter-frame caches in that comparison; cache strategy is a separate later experiment requiring train-only sequences.', '',
 '## Training budget', '',
 'HistoricalV3 selected25k after a cumulative30k run; screens are1k/3k/5k and fresh initialization. A3k skip learning curve cannot isolate this entire historical gap. Full training states exist for probes; originalA/B final checkpoints are weights-only.', '',
 'Controlled budget test:only extend a promising selected smoke from1k to3k TOTAL with the same initialized experiment,optimizer,RNG,data sampler and fixed5k LR timeline. Retain1k/2k/3k measurements. This is research budget evidence, not automatic authorization for C5k or Stage2. Do not compare a new schedule with an old short run and call it budget-only.', '',
 '## Scheduler', '',
 f"Current explicit5k cosine with300-step warmup; LR samples:{result['scheduler']['lrAtStep']}. Warmup occupies30% of1k screening,10% of3k and6% of5k.", '',
 'Controlled scheduler test:two fresh matched1k runs, same warmup300 and peak2e-4, cosine5k prefix versus constant post-warmup. The categorical changed factor is the LR timeline; all other factors and1000-step budget fixed. Do not alter scheduler state halfway through an existing experiment. No scheduler ablation has run in this audit.', '',
 '## Decision order', '',
 'Finish Skip+Frequency vs Skip+Legacy first. Freeze the provisional recipe for each separate investigation; report regressions individually. C5k only after the chosen dataset/crop/codec/budget/scheduler protocol is explicit. Stage2 total15k only when C starts consistently surpassing HistoricalV3 on fixed quality,real video and acceptable latency. No release/preset changes.', '',
 'Sources:diagnostics/zenkai-protocol-factors.json; native/lite_online_dataset.py; native/lite_degradations.py; native/train_zenkai.py; models/legacy/adore-lite-v3/frozen-20260930/report.json.']
 (ROOT/'docs/zenkai-protocol-factors.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('Separate five-factor metadata audit saved; causal training tests NOT EXECUTED')

if __name__=='__main__':main()
