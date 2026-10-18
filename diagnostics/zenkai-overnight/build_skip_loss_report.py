"""Consolidate the frozen3k gate and the completed within-skip loss pair."""
import json,hashlib,datetime,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
from frozen_video_protocol import validate_protocol
RUN='zenkai-ablation-spab-identity-skip-legacy-loss-1k'
REF='zenkai-ablation-spab-identity-skip-1k'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]
def pct(a,b):return (a/b-1)*100

def main():
 prefix='skip-loss-1k-recovery1'
 assert read('diagnostics/zenkai-training/'+prefix+'.status.json')['state']=='complete'
 exits={}
 for stage in ('train','fixed-quality','gradients','video-gate','video-skip-frequency','video-skip-legacy-loss'):
  p='diagnostics/zenkai-training/'+prefix+'-'+stage+'.exit.json';v=read(p)
  assert v['exitCode']==0
  exits[stage]={'path':p,'sha256':sha(p),'exitCode':0}
 folder=ROOT/'models/zenkai/experiments'/RUN
 config=read(folder/'config.json');training=read(folder/'report.json');pair=read(folder/'initial-pairing.json')
 assert training['stepsExecuted']==1000 and training['optimizerUpdates']==1000 and pair['state']=='matched'
 assert config['family']=='ZenkaiIdentitySkipProbe' and config['comparisonRun']==REF
 assert config['lossWeights']==read('models/legacy/adore-lite-v3/frozen-20260930/report.json')['loss']
 for name,digest in config['sourceFilesSha256'].items():assert sha(folder/'source'/name)==digest
 q=read('diagnostics/zenkai-ablations/skip-loss-1k-quality/quality.json')
 assert q['state']=='complete' and list(q['sampleCounts'].values())==[32]*5
 parity=read('diagnostics/zenkai-ablations/skip-loss-1k-quality/parity.json')
 assert parity['state']=='complete' and all(r['parityApproved'] for r in parity['models'].values())
 videos=[read('diagnostics/zenkai-real-video-'+label+'-1k/report.json') for label in ('skip-frequency','skip-legacy-loss')]
 for v in videos:validate_protocol(v)
 gradient=read('diagnostics/zenkai-skip-loss-gradient-analysis.json')
 lines=['# Zenkai: Skip+Frequency versus Skip+Legacy','',f"Measured UTC:{datetime.datetime.now(datetime.timezone.utc).isoformat()}",'',
 'The user-approved3k gate was already complete. Skip3k improves MAE in five domains and visual MAE in three clips versus Frequency3k, with one temporal regression and7?8% extra latency. It recovers block gradients but still loses to HistoricalV3 in all five fixed MAEs and two video clips. This justifies a research loss pair, not C5k/Stage2 or promotion.', '',
 '## Frozen3k gate','',
 'All original values and individual metrics are in docs/zenkai-ablations.md. No3k training repeated. Key differences (lower is better):','']
 q3=read('diagnostics/zenkai-ablations/probes-3k-quality/quality.json');rows=[]
 for domain,m in q3['cases'].items():
  a=m['zenkai-frequency-3k']['mae'];b=m['zenkai-identity-skip-3k']['mae'];v=m['adore-lite-v3']['mae']
  rows.append([domain,f'{v:.9f}',f'{a:.9f}',f'{b:.9f}',f'{pct(b,a):+.4f}',f'{pct(b,v):+.4f}'])
 lines+=table(['Domain MAE','HistoricalV3','Frequency3k','Skip3k','Skip/ref %','Skip/V3 %'],rows)
 lines+=['','Latency medians17.832/65.847ms Frequency3k;19.105/70.980ms Skip3k;7.696/27.000ms HistoricalV3 (360?720 /720?1080).50 warmups/500 distinct measured frames, ROCm FP16 inference+resize. Not playerFPS. New1k-checkpoint latency:NOT MEASURED.', '',
 'Block1 init?3k accumulated weight-change/init ratio4.006e-6 frequency versus0.326277 skip; scaled-AMP zero gradients99.984% versus0% on recorded clean sample. These are checkpoint-local probes, not every training step.', '',
 '## Within-skip protocol','',
 f"Both32ch/depth5 x2 identity skip, seed929, LR128 online crops,batch4?2,codec.25,AdamW,same first1000 steps of5k LR timeline. Changed factor:complete loss profile. Initialized state matched reference SHA {pair['referenceSha256']}. This is not the earlier vanilla-A legacy-loss experiment.", '',
 f"Training exit0;1000 steps/updates;duration{training['trainingSeconds']:.3f}s. Loss values across different objectives are not quality comparisons. FP32 master/full resume state preserved; no checkpoint/default replaced.", '',
 '## ROCm numeric equivalence','',
 'Worst case max/mean error, not pooled mean. Synthetic numerical gates do not prove quality:','']
 rows=[]
 for model,r in parity['models'].items():
  for key in ('fusionFp32','deployFp16'):
   rows.append([model,key,f"{max(c[key]['max_abs_error'] for c in r['cases']):.9g}",f"{max(c[key]['mean_abs_error'] for c in r['cases']):.9g}"])
 lines+=table(['Model','Comparison','max_abs_error','mean_abs_error'],rows)
 lines+=['','ONNX FP32/FP16 probe exports and ONNX GPU:NOT MEASURED. Separate experimental family retained, no ordinary-product registry integration.','', '## Fixed160 samples','',
 'MAE/MSE/LPIPS/Edge/Laplacian/HF/DCT lower;PSNR/SSIM/MS-SSIM higher. Percentage PSNR change is a table difference on a dB-valued metric, not a linear quality percentage. Individual metrics take priority.','']
 rows=[]
 for domain,m in q['cases'].items():
  for metric,a in m['skip-frequency-1k'].items():
   b=m['skip-legacy-loss-1k'][metric];v=m['adore-lite-v3'][metric]
   rows.append([domain,metric,f'{v:.9f}',f'{a:.9f}',f'{b:.9f}',f'{pct(b,a):+.4f}',f'{pct(b,v):+.4f}'])
 lines+=table(['Domain','Metric','HistoricalV3','SkipFrequency1k','SkipLegacy1k','Legacy/ref %','Legacy/V3 %'],rows)
 vanilla=read('diagnostics/zenkai-ablations/legacy-v3-loss-1k-quality/quality.json')
 assert vanilla['indices']==q['indices'] and vanilla['validationManifestSha256']==q['validationManifestSha256']
 lines+=['','## Descriptive loss effect in both architectures','',
  'Same160 fixed samples. Relative MAE changes only; this single-seed2x2 description is not a statistical interaction test. Positive means legacy loss worsens MAE.','']
 rows=[]
 for domain,m in q['cases'].items():
  v=vanilla['cases'][domain]
  rows.append([domain,f"{pct(v['zenkai-legacy-v3-loss-1k']['mae'],v['zenkai-frequency-1k']['mae']):+.4f}",
   f"{pct(m['skip-legacy-loss-1k']['mae'],m['skip-frequency-1k']['mae']):+.4f}"])
 lines+=table(['Domain','Legacy/frequency vanilla %','Legacy/frequency identity-skip %'],rows)
 lines+=['','## Frozen144-frame real video','',
 'Three identical48-frame clips,CPU FP32,360?720. Reference-compensated temporal pixel errors do not certify absence of perceptual flicker/shimmering. All-transition metric includes the retained clip600 suspected cut; standard temporal metric excludes it.','']
 rows=[]
 for a,b in zip(videos[0]['sequences'],videos[1]['sequences']):
  for field in ('manifestSha256','frameCount','suspectedCutTransitions'):assert a[field]==b[field]
  for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
   x=a['models'][REF][metric];y=b['models'][RUN][metric];v=a['models']['adore-lite-v3'][metric]
   assert abs(v-b['models']['adore-lite-v3'][metric])<1e-9
   rows.append([Path(a['manifest']).parent.name,metric,f'{v:.9f}',f'{x:.9f}',f'{y:.9f}',f'{pct(y,x):+.4f}',f'{pct(y,v):+.4f}'])
 lines+=table(['Clip','Metric','V3','SkipFrequency1k','SkipLegacy1k','Legacy/ref %','Legacy/V3 %'],rows)
 lines+=['','## Checkpoint-local gradients','',
 'Same fixed clean/heavy sample IDs,FP32 and scaledAMP using each actual saved scaler. Different loss objectives change gradient magnitudes. No optimizer update performed in probes. Accumulated update norms are init?1k deltas, not one-step optimizer norms.','']
 rows=[]
 for run,r in gradient['runs'].items():
  assert r['checkpointSha256']==sha(ROOT/'models/zenkai/experiments'/run/'step-1000-master-fp32.pt')
  for case in r['cases']:
   for group,g in case['amp']['groups'].items():
    w=r['accumulatedWeightChanges'][group];current=r['currentWeightL2'][group]
    rows.append([run,case['case'],group,f"{g['gradientL2']:.9g}",f"{g['zeroElementsPct']:.4f}",f'{current:.9g}',f"{w['deltaL2']:.9g}",f"{w['deltaL2']/max(current,1e-30):.9g}"])
 lines+=table(['Run','Case','Block','Grad L2 AMP','Zero %','Weight L2 current','Delta L2 init?1k','Delta/current ratio'],rows)
 lines+=['','## Separate protocol investigations','',
 'Metadata-only findings and independently specified tests for dataset,crop,codec sampling,budget,scheduler are in docs/zenkai-protocol-factors.md. Those causal training tests have NOT EXECUTED in this round. Do not call differences causes or mix all five into C.', '',
 '## Infrastructure and scope','',
 'Initial launch failed during import before any run directory/update because embedded ROCm excludes the script directory from sys.path. Original failed logs/status/exit retained. Explicit script-parent bootstrap tested RED?GREEN; ROCm --help exit0; unique recovery1 runner completed the actual run. No training checkpoint reset or restart.', '',
 'Human blind preference study:NOT EXECUTED. Blind panel generation is not a study. Manual still review and measured decision are appended after inspection. Stage2 and C5k not launched; final-hidden unopened.']
 lines+=['','## Measured decision and visual review','',
  'Within-skip legacy-loss superiority:INCONCLUSIVE. Advancement against HistoricalV3:FAIL. No3k extension of the legacy smoke, C5k or Stage2 started.', '',
  'Legacy versus skip-frequency fixed MAE:clean+0.3840%,heavy?0.0292%,H264?0.0489%,H265+0.0075%,AV1+0.0244%. Clean metrics all regress; individual other metrics are mixed. All five domain MAEs remain worse than HistoricalV3. Clean LPIPS alone beats HistoricalV3, which does not outweigh the other regressions.', '',
  'On video, legacy visual MAE improves0.0343/0.2120/0.1857% versus skip-frequency. Temporal improves0.0198/0.0841% on clips60/1200 and worsens0.0042% on clip600. Absolute values are retained. Against HistoricalV3,both skip1k candidates still lose visual and temporal errors in clips60/600 and win clip1200. These are tiny mixed differences and do not establish practical/statistical loss superiority.', '',
  'Block1 has0% zero AMP gradients in both clean probes; heavy has0% Frequency and0.001946% Legacy (3 of154176 elements), with init?1k delta/initial ratios0.226563 frequency and0.227592 legacy. Thus recovered learning persists with either profile; the legacy profile does not demonstrate a new quality breakthrough.', '',
  'Nine panels generated at the same frames0/24/47 and ROI(448,100,832,484). Manual unblinded inspection covered clip60/frame0,clip600/frame24,clip1200/frame47. Both skip losses show subtle neural differences and improve some contours versus bilinear, but retain lost fine eye/hair/finger/background details relative to reference. No clear consistent visible preference between skip losses was established. Provenance/source hashes in diagnostics/zenkai-real-video-comparison-crops-skip-loss-1k/provenance.json. Still inspection is not moving-video or blind human validation.', '',
  'Provisional experimental reference:keep Skip+Frequency for separate protocol-factor tests because the legacy switch has no coherent demonstrated gain. This is a conservative experiment choice, not a declared quality winner or extension/default release. The independent factor audit is completed; causal dataset/crop/codec/budget/scheduler training tests remain pending. User order preserved:investigate those before C5k; require C gains over HistoricalV3 before total15k.', '',
  'Limit:one seed and1000-step screening cannot establish equivalence or exclude a late-budget/loss interaction. Final-hidden remains untouched. HistoricalV3 and all prior runs remain preserved.']
 (ROOT/'docs/zenkai-skip-loss.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 for name,value in {'quality-report.json':q,'real-video-report.json':videos[1],'gradient-report.json':gradient['runs'][RUN],'training-history.json':training['history']}.items():
  (folder/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
 m={'run':RUN,'config':config,'stepFinal':1000,'trainingExitCode':0,'trainingSeconds':training['trainingSeconds'],
  'screeningDecision':{'lossSuperiority':'INCONCLUSIVE','advancementAgainstHistoricalV3':'FAIL','provisionalReference':'Skip+Frequency'},'best':training['best'],'initialPairing':pair,'stageExitRecords':exits,'validationHistory':training['history'],
  'checkpointsSha256':{p.name:sha(p) for p in folder.glob('*.pt')},
  'evaluationArtifactsSha256':{p.name:sha(p) for p in folder.glob('*report.json')},
  'sourceSnapshotSha256':config['sourceFilesSha256'],'releaseApproved':False,'performance':'NOT MEASURED','onnx':'NOT MEASURED'}
 (folder/'metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
 print('Measured within-skip loss report and metadata saved')

if __name__=='__main__':main()
