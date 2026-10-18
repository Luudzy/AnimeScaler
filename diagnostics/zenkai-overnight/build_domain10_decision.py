"""Separate sampler/product gates; no automatic training or product promotion."""
import argparse,hashlib,json
from pathlib import Path
from build_codec3k_report import table
ROOT=Path(__file__).resolve().parents[2]
RUN='zenkai-ablation-skip-v2-domain10-1k';SOURCE='zenkai-ablation-skip-v2-dataset-1k'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(a,b):return 100*(a/b-1)

def quality_deltas(cases):
 percent={};db={}
 for dom,m in cases.items():
  percent[dom]={k:{'vsUniform1k':rel(m['v2-domain10-1k'][k],m['v2-dataset-1k'][k]),'vsHistoricalV3':rel(m['v2-domain10-1k'][k],m['adore-lite-v3'][k])} for k in m['v2-domain10-1k'] if k!='psnr'}
  db[dom]={label:m['v2-domain10-1k']['psnr']-m[model]['psnr'] for label,model in (('vsUniform1k','v2-dataset-1k'),('vsHistoricalV3','adore-lite-v3'))}
 return percent,db
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--sampler-gate',choices=('PROMISING','FAIL','INCONCLUSIVE'),required=True);parser.add_argument('--reason',required=True);a=parser.parse_args()
 path=ROOT/'diagnostics/zenkai-domain10-1k/comparison.json';r=read(path);old=read(ROOT/'diagnostics/zenkai-v2-dataset-1k/comparison.json')
 if r['state']!='COMPLETE' or any(v['state']!='PASS' for v in r['operations'].values()):raise ValueError('A complete numeric decision requires all operations')
 q,psnr_db=quality_deltas(r['quality']['cases'])
 v={}
 for b,n in zip(old['video']['sequences'],r['video']['sequences']):
  v[Path(n['manifest']).parent.name]={k:{'vsUniform1k':rel(n['models'][RUN][k],b['models'][SOURCE][k]),'vsHistoricalV3':rel(n['models'][RUN][k],n['models']['adore-lite-v3'][k])} for k in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae')}
 product='FAIL' if all(m['mae']['vsHistoricalV3']>0 for m in q.values()) and sum(all(m[k]['vsHistoricalV3']>0 for k in ('mae','referenceCompensatedTemporalMae')) for m in v.values())>=2 else 'INCONCLUSIVE'
 d={'run':RUN,'comparisonSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samplerHypothesis':a.sampler_gate,'samplerReason':a.reason,'advancementAgainstHistoricalV3':product,'operationState':'PASS','meaning':'Selection criterion only;valid training/evaluation/backend operations preserved','qualityRelativePercent':q,'videoRelativePercent':v,'samplingAnalysis':r['samplingAnalysis'],'longerRunStarted':False,'Stage2Started':False,'releaseApproved':False,'promoted':False,'finalHiddenOpened':False,'limitations':['1k smoke/single seed','Same uniform-union dataset;sampler distribution and associated crop exposure changed intentionally','Frozen validation anime-only','Three correlated excerpts of one compressed episode','No blind moving/player certification','HistoricalV3 training budget differs;product gate is not pure architectural attribution']}
 d['qualityPsnrDeltaDb']=psnr_db
 (ROOT/'docs/zenkai-domain10-decision.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
 lines=['# Decisão do sampler10%','',f'**Sampler:{a.sampler_gate}. Avanço contra HistoricalV3:{product}.**','',a.reason,'',
  'Operações técnicas PASS. Os gates descrevem seleção desta triagem,não falhaHIP/checkpoint/modelo. Não treinar3k/5k/15k só por terminar1k.','']
 for baseline in ('vsUniform1k','vsHistoricalV3'):
  lines+=['## '+baseline,'','Positivo é pior para as métricas abaixo;negativo melhor. Todas as demais métricas individuais constam no relatório completo.','']
  lines+=table(['Domain','MAE %','Edge %','Laplacian %','LPIPS %'],[[dom,*[m[k][baseline] for k in ('mae','edgeMae','laplacianMae','lpipsSqueeze')]] for dom,m in q.items()])+['']
  lines+=table(['Clip','Visual MAE %','Temporal compensated %','All transitions %'],[[dom,*[m[k][baseline] for k in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae')]] for dom,m in v.items()])+['']
 s=r['samplingAnalysis'];lines+=[f"8000 draws:{s['actualDomainDraws']};baselineuniforme:{s['referenceDomainDraws']};exposiçãoanime{s['animeExposureChangePercent']:+.5f}%. Targetillustrationprobability10%,fração realizada{s['actualIllustrationFraction']:.4%}. Algoritmo weightedreplacement versus uniformereplacement é a mudança explícita deste fator.",'',
 'Norma maior de gradiente/PSNR/LPIPS isolado não comprova lineart nem estabilidade visual. Os deltas acumulados de pesos não são updates de um step. Mais treino não reduz o custo fixo de inferência. Final-hidden permanece reservado;nenhuma alteração de produto ou export/release neste smoke.']
 (ROOT/'docs/zenkai-domain10-decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'sampler':a.sampler_gate,'product':product,'video':v},indent=2))
if __name__=='__main__':main()
