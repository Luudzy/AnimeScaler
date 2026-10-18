"""Document this completed screening criterion, without promotion or a new training run."""
import hashlib
import json
from pathlib import Path
from build_codec3k_report import table

ROOT = Path(__file__).resolve().parents[2]
RUN = 'zenkai-lite-v1-c-codec50-5k'


def relative(value, reference):
    return 100*(value/reference-1)


def main():
    path = ROOT/'diagnostics/zenkai-c5k/comparison.json'
    report = json.loads(path.read_text())
    if report['state'] != 'COMPLETE' or any(report['operations'][k]['state'] != 'PASS'
            for k in ('quality','gradients','benchmark','video','export','runtime')):
        raise ValueError('Complete valid operations required for this full decision')
    quality = {}
    for domain, models in report['quality']['cases'].items():
        c, v3 = models['zenkai-c-5k'], models['adore-lite-v3']
        quality[domain] = {k:relative(c[k],v3[k]) for k in ('mae','edgeMae','laplacianMae','lpipsSqueeze')}
    video = {}
    for s in report['video']['sequences']:
        c, v3 = s['models'][RUN], s['models']['adore-lite-v3']
        video[Path(s['manifest']).parent.name] = {k:relative(c[k],v3[k])
            for k in ('mae','referenceCompensatedTemporalMae')}
    both_video_regress = sum(all(v>0 for v in metrics.values()) for metrics in video.values())
    all_mae_regress = all(m['mae']>0 for m in quality.values())
    all_edge_regress = all(m['edgeMae']>0 for m in quality.values())
    gate = 'FAIL' if all_mae_regress and all_edge_regress and both_video_regress>=2 else 'INCONCLUSIVE'
    decision = {'run':RUN, 'comparisonSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'advancementAgainstHistoricalV3':gate,
        'meaning':'Screening quality/temporal advancement criterion only; training/export/backend operations passed',
        'qualityRelativePercentLowerIsBetter':quality, 'videoRelativePercentLowerIsBetter':video,
        'allFiveDomainsMaeRegress':all_mae_regress, 'allFiveDomainsEdgeRegress':all_edge_regress,
        'clipsWithBothVisualAndTemporalRegression':both_video_regress,
        'coherentGainEstablished':False, 'Stage2Started':False, 'Stage2AuthorizedByEvidence':False,
        'releaseApproved':False, 'promoted':False, 'finalHiddenOpened':False,
        'nextControlledHypothesis':'Prepared V2 union as a dataset-only smoke versus original Skip+Frequency1k. Do not combine Codec50/group/crop/legacy/scheduler switches.',
        'definitiveGeneralization':'NOT ESTABLISHED:single seed and three excerpts of one compressed source'}
    (ROOT/'docs/zenkai-c5k-decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    lines = ['# Decisão do screening C5k','',
        f'**Avanço contra Historical V3: {gate}. Stage2: NÃO iniciado. Sem promoção.**','',
        'O treino e todas as operações técnicas concluíram com PASS. FAIL aqui descreve o critério de ganho de qualidade/estabilidade deste screening; não representa checkpoint corrompido ou falha HIP. Nenhuma regressão foi escondida em composite.','',
        'O C melhora MAE nos cinco domínios frente ao próprio Codec50 3k, mas a comparação abaixo conserva as perdas frente ao baseline de produto. Ganho de budget e avanço do cosine ocorreram juntos; não é uma ablation causal pura do budget.','']
    lines += table(['Domínio','MAE %','Edge MAE %','Laplacian MAE %','LPIPS %'],
        [[d,*[f'{m[k]:+.5f}' for k in ('mae','edgeMae','laplacianMae','lpipsSqueeze')]] for d,m in quality.items()])
    lines += ['', 'Percentuais C/V3; positivo é pior, negativo é melhor para estas métricas.','']
    lines += table(['Trecho','MAE visual %','MAE temporal compensado %'],
        [[d,f'{m["mae"]:+.5f}',f'{m["referenceCompensatedTemporalMae"]:+.5f}'] for d,m in video.items()])
    lines += ['',
        'O ganho em um trecho não compensa automaticamente a regressão visual/temporal nos outros dois. A redução de erro temporal de pixels não equivale a ausência de flicker/shimmering. O clean tem LPIPS melhor, mas os demais resultados e os detalhes visuais precisam ser considerados juntos.','',
        'A latência ROCm FP16 continua muito acima do V3;720→1080 não atende ao orçamento de33,3ms de inferência nesta GPU.360→720 cabe nesse orçamento isolado, sem medição do player. A geometria convolucional tem cerca de3,08× os MACs do V3; mais treino não reduz esse custo.','',
        'Não iniciar15k/30k nem abrir final-hidden. Os exports numericamente válidos podem ser embalados como candidatos experimentais para revisão privada; isso não é release de produto. Preservar5000 final/best,4000 e todos os estados de recuperação.','',
        'Próximo teste informativo:dataset-only com as602 ilustrações V2 verificadas e o novo manifest separado. Usar uma inicialização pareada e todos os demais fatores iguais ao Skip+Frequency1k; não adicionar as imagens ao C concluído. Outra seed/material independente e validação no player vêm após existir um candidato promissor.','']
    (ROOT/'docs/zenkai-c5k-decision.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'screening':gate,'Stage2Started':False,'releaseApproved':False}))


if __name__ == '__main__':
    main()
