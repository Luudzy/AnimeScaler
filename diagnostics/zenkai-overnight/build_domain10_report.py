"""Selected complete domain10 measurements; frozen reference identities and all metrics."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
from artifact_provenance import validate_artifact_pins,validate_checkpoint_file
from frozen_video_protocol import validate_protocol
from build_codec3k_report import selected,table,pct
from zenkai_domain_sampler import RUN,SOURCE,load_candidate,validate_initial_pair,INITIAL_SHA
BASE=ROOT/'diagnostics/zenkai-domain10-1k'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_pinned_comparison(path,metadata):
 validate_checkpoint_file(path,metadata.get('evaluationReportSha256'))
 report=read(path)
 if report.get('state')!='COMPLETE':raise ValueError('Historical comparison not complete')
 return report
def main():
 p=read(BASE/'pipeline.json');t=read(ROOT/'diagnostics/zenkai-training/domain10-1k-training/operation.json')
 if p['state']!='COMPLETE' or t['state']!='PASS':raise ValueError('Operations not finished')
 results={k:selected(v) for k,v in p['results'].items()}
 f=ROOT/'models/zenkai/experiments'/RUN;cfg=read(f/'config.json');train=read(f/'report.json');index=read(f/'checkpoint-index.json')
 if train['stepsExecuted']!=1000 or train['optimizerUpdates']!=1000 or train['samples']!=8000 or index['lastCompleteStep']!=1000:raise ValueError('Wrong actualbudget')
 for n,pin in cfg['sourceFilesSha256'].items():
  if sha(f/'source'/n)!=pin:raise ValueError('Archived source changed')
 for n,pin in index['checkpoints'].items():
  if sha(f/n)!=pin:raise ValueError('Indexed state changed')
 validate_checkpoint_file(f/'initial-training-state.pt',index['initialSha256'])
 import torch
 source_initial=ROOT/'models/zenkai/experiments'/SOURCE/'initial-training-state.pt'
 validate_checkpoint_file(source_initial,INITIAL_SHA)
 validate_initial_pair(torch.load(f/'initial-training-state.pt',map_location='cpu',weights_only=True),
     torch.load(source_initial,map_location='cpu',weights_only=True))
 load_candidate(f/'step-1000-master-fp32.pt')
 pins={v.name:sha(v) for v in f.glob('*.pt')}
 if pins['step-1000-master-fp32.pt']!=t['result']['sha256'] or pins['step-1000-training-state.pt']!=t['result']['stateSha256']:raise ValueError('Training provenance changed')
 pair=read(f/'initial-pairing.json')
 if pair['state']!='matched' or not pair['baselineSamplerReproducedInActualStack']:raise ValueError('Unpaired sampler')
 source_metadata=read(ROOT/'models/zenkai/experiments'/SOURCE/'metadata.json')
 old=load_pinned_comparison(ROOT/'diagnostics/zenkai-v2-dataset-1k/comparison.json',source_metadata);baseline_sha=old['checkpointsSha256']['step-1000-master-fp32.pt']
 if old['state']!='COMPLETE' or sha(ROOT/'models/zenkai/experiments'/SOURCE/'step-1000-master-fp32.pt')!=baseline_sha:raise ValueError('Uniform baseline changed')
 domains=train['sampledTrainDomainCounts'];expect=read(ROOT/'diagnostics/zenkai-training/domain10-preflight.json')['plannedFirst8000DomainDraws']
 if domains!=expect:raise ValueError('Actual sampler counts differ from planned verified stream')
 exposure=100*(domains['anime']/old['training']['sampledTrainDomainCounts']['anime']-1)
 report={'state':'COMPLETE','run':RUN,'trainingOperation':t,'config':cfg,'training':train,'initialPairing':pair,'checkpointsSha256':pins,'operations':p['results'],
  'baselineRun':SOURCE,'baselineComparisonSha256':sha(ROOT/'diagnostics/zenkai-v2-dataset-1k/comparison.json'),'samplingAnalysis':{'actualDomainDraws':domains,'referenceDomainDraws':old['training']['sampledTrainDomainCounts'],'animeExposureChangePercent':exposure,'targetIllustrationProbability':.1,'actualIllustrationFraction':domains['illustration']/8000},
  'quality':None,'parity':None,'video':None,'performance':None,'gradients':None,'Stage2Started':False,'releaseApproved':False,'promoted':False,'finalHiddenOpened':False}
 lines=['# Sampler V2:10% ilustração versus união uniforme','',
  'Mesmo2394TRAIN;somente distribuição de amostragem mudou. Pesos/optimizer/scaler/Python/Torch/GPU/NumPy/datasetRNG iniciais pareados;baselineuniforme reproduzido noROCm real. WeightedRandomSampler com replacement,seed929,40000 índices e posição completa salva.','',
  f"1000 steps/1000updates,exit0,tentativa{t['selectedAttempt']},loop{train['trainingSeconds']:.3f}s.8000 draws:anime{domains['anime']},illustration{domains['illustration']}({domains['illustration']/8000:.4%});alvo probabilístico10%. Uniforme anterior:{old['training']['sampledTrainDomainCounts']};exposiçãoanime{exposure:+.5f}%. Codec draws{train['codecSamples']};identidades distintas{train['distinctTrainImagesSampled']}.",'',
  '32ch/depth5/x2/skip1,Frequency loss,codec.25,patch128,batch4×2,AdamW(.9,.99),wd1e-5,warmup300/originalcosine5000,seed929. Validação fixaanime-only;nenhum heldoutV2/hidden aberto. Loss entre distribuições não escolhe vencedor.','']
 lines+=table(['Operation','State','Selected attempt'],[[k,v['state'],v['selectedAttempt']] for k,v in p['results'].items()])
 if results['quality']:
  q=read(results['quality']['quality']);parity=read(results['quality']['parity']);report.update(quality=q,parity=parity)
  for k in ('indices','validationManifestSha256','sourceManifestSha256','codecManifestSha256'):
   if q[k]!=old['quality'][k]:raise ValueError('Fixed160 protocol changed')
  validate_artifact_pins({'new':q['models'].get('v2-domain10-1k',{}),'uniform':q['models'].get('v2-dataset-1k',{}),'v3':q['models'].get('adore-lite-v3',{})},
   {'new':pins['step-1000-master-fp32.pt'],'uniform':baseline_sha,'v3':cfg['legacyV3Sha256']})
  rows=[]
  for dom,m in q['cases'].items():
   for metric,v3 in m['adore-lite-v3'].items():
    a=m['v2-dataset-1k'][metric];b=m['v2-domain10-1k'][metric]
    delta=lambda ref:f'{b-ref:+.5f}dB' if metric=='psnr' else pct(b,ref)
    rows.append([dom,metric,f'{v3:.9f}',f'{a:.9f}',f'{b:.9f}',delta(a),delta(v3)])
  lines+=['','## Mesmos160 samples','', 'MAE/MSE/Edge/Laplacian/HF/DCT/LPIPS:menor melhor;PSNR/SSIM/MS-SSIM:maior melhor. PSNR delta emdB;demais deltas em%. Regressões individuais preservadas.','']+table(['Domain','Metric','HistoricalV3','Uniform1k','Domain10 1k','Delta vsUniform','Delta vsV3'],rows)
  rows=[]
  for name,v in parity['models'].items():
   for comp in ('fusionFp32','deployFp16'):
    rows.append([name,comp,max(c[comp]['max_abs_error'] for c in v['cases']),max(c[comp]['mean_abs_error'] for c in v['cases'])])
  lines+=['','## Paridade ROCm','']+table(['Model','Comparison','Worst max_abs_error','Worst mean_abs_error'],rows)
 if results['video']:
  v=read(results['video']['path']);ref=old['video'];validate_protocol(v);validate_protocol(ref);report['video']=v
  validate_artifact_pins({'new':v['artifacts'].get(RUN,{}),'old':ref['artifacts'].get(SOURCE,{}),'oldV3':ref['artifacts'].get('adore-lite-v3',{}),'v3':v['artifacts'].get('adore-lite-v3',{})},
   {'new':pins['step-1000-master-fp32.pt'],'old':baseline_sha,'oldV3':cfg['legacyV3Sha256'],'v3':cfg['legacyV3Sha256']})
  if v['cutPolicy']!=ref['cutPolicy'] or v['precision']!=ref['precision'] or v['backend']!=ref['backend']:raise ValueError('Video protocol changed')
  rows=[]
  for a,b in zip(ref['sequences'],v['sequences']):
   if a['manifestSha256']!=b['manifestSha256'] or a['suspectedCutTransitions']!=b['suspectedCutTransitions'] or a['frameCount']!=48 or b['frameCount']!=48:raise ValueError('Different real frames/cuts')
   for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
    v3=a['models']['adore-lite-v3'][metric];u=a['models'][SOURCE][metric];n=b['models'][RUN][metric]
    if abs(v3-b['models']['adore-lite-v3'][metric])>1e-9:raise ValueError('V3 CPU baseline changed')
    rows.append([Path(a['manifest']).parent.name,metric,v3,u,n,pct(n,u),pct(n,v3)])
  lines+=['','## Mesmos144 frames','', '3×48,360→720,CPUFP32,LRH264,referênciaWebRip comprimida. Mesmos cortes;all-transitions preservados. Não é avaliação cega/perceptual/player nem prova de ausência de flicker.','']+table(['Clip','Metric(lower better)','V3','Uniform1k','Domain10','10/Uniform %','10/V3 %'],rows)
 if results['benchmark']:
  perf=read(results['benchmark']['path']);report['performance']=perf
  if not perf['complete'] or perf['precision']!='fp16' or perf['provider']!='ROCm' or len(perf['cases'])!=4:raise ValueError('Incomplete performance')
  rows=[]
  for c in perf['cases']:
   if c['warmup']!=50 or c['iterations']!=500 or sha(c['artifactPath'])!=c['artifactSha256']:raise ValueError('Changed performance artifact/protocol')
   for comp,s in c['components'].items():
    if 'medianMs' in s:rows.append([c['model'],f"{c['width']}×{c['height']}→{c['outputWidth']}×{c['outputHeight']}",comp,*[s[k] for k in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps')],c['peakAllocatedMiB']])
  lines+=['','## Performance','', '50warmups/500inputs distintos,FP16ROCm. Uma única execução completa por protocolo;sem partialmerge/reuse. FPS equivalente não é FPS do player. Geometria não muda com sampler;variações não provam aceleração.','']+table(['Model','Shape','Component','Mean','Median','P50','P95','P99','Min','Max','FPS equivalent','Peak MiB'],rows)
 if results['gradients']:
  g=read(results['gradients']['path']);report['gradients']=g;rows=[]
  if g['runs'][RUN]['initialSha256']!=index['initialSha256'] or g['runs'][SOURCE]['initialSha256']!=INITIAL_SHA:
   raise ValueError('Gradient initialization provenance changed')
  for name,master_pin,full_pin in ((RUN,pins['step-1000-master-fp32.pt'],t['result']['stateSha256']),
       (SOURCE,baseline_sha,old['trainingOperation']['result']['stateSha256'])):
   if g['runs'][name]['checkpointSha256']!=master_pin or g['runs'][name]['trainingStateSha256']!=full_pin:
    raise ValueError('Gradient master/full provenance changed')
  if g['validationManifestSha256']!=cfg['fixedValidationSha256']:raise ValueError('Gradient fixed protocol changed')
  for name,v in g['runs'].items():
   for case in v['cases']:
    for group,fp in case['fp32']['groups'].items():
     amp=case['amp']['groups'][group];c=v['accumulatedWeightChanges'][group]
     rows.append([name,case['case'],group,fp['gradientL2'],amp['gradientL2'],c['initialL2'],c['deltaL2'],c['deltaToInitialRatio'],amp['zeroElementsPct']])
  lines+=['','## Gradientes','', 'BackwardFP32/scaledAMP local1k,scaler real,sem optimizerupdate;normaAMP apósunscale. Delta acumulado init→1k não é atualização isolada.','']+table(['Run','Case','Group','FP32 gradL2','AMP gradL2','Initial weightL2','DeltaL2','Delta/initial','AMP zero %'],rows)
 for k,result in results.items():
  if result is None:lines+=['',f'{k}:NOT MEASURED — see selected operation error; other valid results preserved.']
 lines+=['','ONNX deste smoke:NOT EXECUTED. Stage2/release/default/hidden:NÃO. Decisão separada segue o gate predeclarado;não escolher por composite ou loss.']
 (BASE/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(ROOT/'docs/zenkai-domain10-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (f/'metadata.json').write_text(json.dumps({**report,'trainingExitCode':0,'validationHistory':train['history'],'evaluationReport':str(BASE/'comparison.json'),'evaluationReportSha256':sha(BASE/'comparison.json')},indent=2),encoding='utf-8')
 print('Domain10 completed selected operation reports and metadata saved')
if __name__=='__main__':main()
