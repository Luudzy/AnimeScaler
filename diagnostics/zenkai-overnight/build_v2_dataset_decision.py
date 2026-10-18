"""Evidence table for the completed dataset-only screen; no training or promotion."""
import hashlib,json
from pathlib import Path
from build_codec3k_report import table
ROOT=Path(__file__).resolve().parents[2]
RUN='zenkai-ablation-skip-v2-dataset-1k'
REF='zenkai-ablation-spab-identity-skip-1k'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(a,b):return 100*(a/b-1)
def main():
 p=ROOT/'diagnostics/zenkai-v2-dataset-1k/comparison.json';r=read(p)
 if r['state']!='COMPLETE' or any(o['state']!='PASS' for o in r['operations'].values()):raise ValueError('This full decision requires complete operations')
 q={d:{k:{'vsAnime1k':rel(m['v2-dataset-1k'][k],m['skip-frequency-1k'][k]),'vsHistoricalV3':rel(m['v2-dataset-1k'][k],m['adore-lite-v3'][k])} for k in m['v2-dataset-1k']} for d,m in r['quality']['cases'].items()}
 old=read(ROOT/'diagnostics/zenkai-real-video-skip-frequency-1k/report.json');v={}
 for a,b in zip(old['sequences'],r['video']['sequences']):
  v[Path(b['manifest']).parent.name]={k:{'vsAnime1k':rel(b['models'][RUN][k],a['models'][REF][k]),'vsHistoricalV3':rel(b['models'][RUN][k],b['models']['adore-lite-v3'][k])} for k in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae')}
 # Predocumented gate: mixed/small gains require visual/temporal support, not composite selection.
 gate='INCONCLUSIVE'
 if sum(m['mae']['vsAnime1k']>0 for m in q.values())>=3 and sum(all(m[k]['vsAnime1k']>0 for k in ('mae','referenceCompensatedTemporalMae')) for m in v.values())>=2:gate='FAIL'
 product='FAIL' if all(m['mae']['vsHistoricalV3']>0 for m in q.values()) and sum(all(m[k]['vsHistoricalV3']>0 for k in ('mae','referenceCompensatedTemporalMae')) for m in v.values())>=2 else 'INCONCLUSIVE'
 d={'run':RUN,'comparisonSha256':hashlib.sha256(p.read_bytes()).hexdigest(),'datasetHypothesis':gate,'advancementAgainstHistoricalV3':product,'operationState':'PASS','meaning':'Selection criteria only; all technical measurements preserved','qualityRelativePercent':q,'videoRelativePercent':v,'samplingAnalysis':r['samplingAnalysis'],'coherentGainEstablished':False,'longerRunStarted':False,'Stage2Started':False,'releaseApproved':False,'promoted':False,'finalHiddenOpened':False,'limitations':['Single seed and1k smoke','Uniform union reduces anime exposure25.0125%; intrinsic illustration quality is not isolated','Frozen validation anime-only','Three correlated excerpts of one compressed WebRip','No blind moving/player evaluation; no temporal H265/AV1 protocol'],'nextInformativeExperiment':'If this allocation is revisited, independently test domain sampling at fixed total budget with the same frozen union and math; do not combine crop, codec, loss or scheduler switches.'}
 (ROOT/'docs/zenkai-v2-dataset-decision.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
 lines=['# Decisão da triagem V2 de1k','',f'**Dataset: {gate}. Avanço contra Historical V3: {product}.**','', 'Todos os processos concluídos são PASS. A classificação de seleção não representa falha do modelo/checkpoint/HIP. Não iniciar3k/5k/15k por conclusão automática desta triagem.','', 'MAE/Edge/Laplacian/LPIPS e erro temporal:menor é melhor. Positivo nas tabelas abaixo significa pior. Todas as demais métricas permanecem no relatório completo.','']
 for baseline in ('vsAnime1k','vsHistoricalV3'):
  lines+=['## '+baseline,'']+table(['Domain','MAE %','Edge MAE %','Laplacian MAE %','LPIPS %'],[[dom,*[f'{m[k][baseline]:+.5f}' for k in ('mae','edgeMae','laplacianMae','lpipsSqueeze')]] for dom,m in q.items()])
  lines+=['']+table(['Clip','Visual MAE %','Temporal compensated %','All transitions %'],[[dom,*[f'{m[k][baseline]:+.5f}' for k in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae')]] for dom,m in v.items()])+['']
 lines+=['A união uniforme substituiu2001 dos8000 draws de anime por ilustrações. O teste mede essa distribuição de treinamento com orçamento fixo; não condena o material V2 nem prova que treiná-lo por mais steps resolveria as regressões. Ganhos pequenos em algumas métricas, inclusive LPIPS, não isolam recuperação de detalhes.','', 'Não existe finalista coerente para Stage2 ou release. C5k continua preservado e não supera Historical V3 no gate já documentado. Outro budget/seed/material independente, temporal H265/AV1 e player só devem sustentar uma conclusão mais ampla com candidato promissor. Final-hidden continua reservado.','', 'Próxima hipótese independente, se necessário:amostragem por domínio sobre o mesmo union, orçamento total fixo e inicialização pareada. Não misturar novo codec/crop/loss/scheduler.']
 (ROOT/'docs/zenkai-v2-dataset-decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'dataset':gate,'product':product,'video':v},indent=2))
if __name__=='__main__':main()
