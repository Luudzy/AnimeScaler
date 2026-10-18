"""Write latest complete round ahead of preserved historical overnight entries."""
import json
from pathlib import Path
from build_codec3k_report import table

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    report = read(ROOT/'diagnostics/zenkai-c5k/comparison.json')
    decision = read(ROOT/'docs/zenkai-c5k-decision.json')
    package = read(ROOT/'models/zenkai/review-packages/c5k/packages.json')
    motion = read(ROOT/'diagnostics/zenkai-c5k/motion-review-v2/provenance.json')
    audit = read(ROOT/'diagnostics/zenkai-c5k/integrity.json')
    raw_tests = (ROOT/'diagnostics/zenkai-training/c5k-final-tests.log').read_bytes()
    tests = raw_tests.decode('utf-16' if raw_tests.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    if not all(report['operations'][k]['state']=='PASS' for k in ('quality','gradients','benchmark','video','export','runtime')):
        raise ValueError('Screening operations not all complete')
    if audit['state']!='PASS' or 'Ran 165 tests' not in tests or not tests.rstrip().endswith('OK'):
        raise ValueError('Integrity/tests incomplete')
    train = report['training']
    prior = read(ROOT/'models/zenkai/experiments/zenkai-ablation-skip-factor-codec50-3k-continuation/report.json')
    lines = ['# ZENKAI OVERNIGHT REPORT — C5k concluído, 2026-10-04', '',
        '**C5k completo; operações técnicas PASS; avanço contra Historical V3 FAIL neste screening. Stage2 NÃO iniciado. Sem promoção.**', '',
        '## 1. Treinamento', '',
        f'Final:5000 steps/5000 updates, exit0, tentativa1. Continuação fiel do estado completo3000→5000, em pasta nova. Best:5000, composite{train["best"]["composite"]:.9f}; loss final{train["history"][-1]["trainingLoss"]:.9f}.', '',
        f'Tempo acumulado registrado pelo loop (inclui suas validações):{train["trainingSeconds"]:.3f}s; intervalo3k→5k:{train["trainingSeconds"]-prior["trainingSeconds"]:.3f}s. Não é FPS do player nem tempo corrido incluindo pausas entre runs.', '',
        f'Master SHA256:`{audit["masterSha256"]}`. Estado completo:`{audit["fullStateSha256"]}`.20 checkpoints de recuperação verificados porSHA, FP32 master/optimizer/scaler/RNG/datasetRNG/sampler/scheduler íntegros, best/final iguais em tensores. Fonte3k, dataset e fontes arquivadas preservados. Config/histórico/gitCommit/hashes em metadata do run.', '',
        '## 2. Exports e paridade', '',
        'FP32 deploy, FP16 NPZ, ONNX FP32/FP16:PASS. Fusão/precisão/ONNX CPU e deploy PyTorch recarregado no ROCm passaram as tolerâncias existentes; max_abs_error/mean_abs_error por caso estão em [resultados](zenkai-c5k-results.md). ONNX GPU/TensorRT/WebGPU:NOT MEASURED. Nenhum registro/default foi alterado.', '',
        '## 3. Performance', '',
        'ROCm FP16,50 warmups/500 entradas distintas, execução completa selecionada. Inferência, post-resize e total estão separados no relatório; tabela abaixo:total medido. FPS equivalente não é FPS do player.', '']
    rows = []
    for c in report['performance']['cases']:
        t = c['components']['totalMeasured']
        rows.append([c['model'],f'{c["height"]}→{c["outputHeight"]}',f'{t["medianMs"]:.5f}',f'{t["p95Ms"]:.5f}',c['parameters']])
    lines += table(['Modelo','Altura entrada→saída','Mediana ms','P95 ms','Params deploy'],rows)
    lines += ['', '720→1080 do C não cabe em33,3ms nesta GPU;360→720 cabe apenas no orçamento isolado. Player end-to-end:NOT MEASURED.','',
        '## 4. Validação fixa','',
        'Mesmo160 samples/32 por domínio. Todos os domínios melhoram MAE frente ao Codec50 3k, mas continuam acima de V3; Edge MAE também piora nos cinco. Clean LPIPS ganha, heavy e codecs perdem. As50 métricas individuais e deltas frente ao3k/V3 estão no relatório; nenhum composite determina sozinho a decisão.','']
    lines += table(['Domínio','MAE C/V3 % (menor melhor)','LPIPS C/V3 % (menor melhor)'],
        [[d,f'{m["mae"]:+.5f}',f'{m["lpipsSqueeze"]:+.5f}'] for d,m in decision['qualityRelativePercentLowerIsBetter'].items()])
    lines += ['', '## 5. Vídeo real','',
        'Mesmos144 frames/3×48,CPU FP32,suspected-cut policy original. Referência:WebRip comprimido,não comprovado master4K.','']
    lines += table(['Trecho','MAE visual C/V3 %','MAE temporal C/V3 %'],
        [[d,f'{m["mae"]:+.5f}',f'{m["referenceCompensatedTemporalMae"]:+.5f}'] for d,m in decision['videoRelativePercentLowerIsBetter'].items()])
    lines += ['',
        'O C melhora60s/600s frente ao3k, mas piora1200s frente ao3k. Contra V3 perde visual/temporal em dois trechos e ganha em um. Inspecionados três stills fixos:[observações](zenkai-c5k-visual-review.md). Três painéis em movimento FFV1 foram preparados e todos144 frames RGB empilhados conferidos porMD5 após decode; isso prepara revisão humana, não certifica preferência/flicker/player.','',
        'Defeito encontrado:identificação por profundidade rotulava o checkpoint C comozenkai-a. SHA correto confirmado, teste RED→GREEN, nova passagem CPU completa em pasta separada. Pipeline/report/logs originais preservados. O relatório final usa apenas o replay completo com identidade corrigida. Uma terceira passagem independente salvou todos os PNGs para motion-review e reproduziu as métricas selecionadas.','',
        '## 6. Gradientes','',
        'Quatro blocos internos ativos nas duas amostras fixas FP32/scaled AMP; sem evidência de dead block nestas probes. Mudanças de peso3k→5k reais preservadas. Delta acumulado não é update-to-weight de um único step; normas por-step históricas:NOT MEASURED. [Análise completa](zenkai-c5k-gradient-analysis.md).','',
        '## 7. Controles e screening','',
        'Adore V3 Retrained Control5k e A/B anteriores já concluídos e preservados; não repetidos nesta rodada. Scheduler-v2novo1k passou todas as operações, porém LR constante NOT SELECTED por falta de ganho coerente. C manteve cosine/Codec50 e nenhum fator foi combinado.','',
        f'C5k advancement:{decision["advancementAgainstHistoricalV3"]}. Training/quality/video/gradient/benchmark/export/runtime operations:PASS. Stage2Started:NO;15k/30k/hidden/default/presets:NO. Artefato experimental local de revisão privada:`models/zenkai/review-packages/c5k/zenkai-lite-v1-c-codec50-5k.zip`.', '',
        f'ZIP SHA256:`{package["sha256"]}`;28 membros, CRC/memberSHA aprovados. Export técnico correto não equivale a release de qualidade.','',
        '## 8. V2 e próxima hipótese','',
        '602 ilustrações TRAIN adicionais auditadas porSHA e full pixel decode;602 pares codec também decodificados/geometria conferida.166 heldouts históricos excluídos; nenhum hidden pixel aberto. Novo union manifest2394 TRAIN preparado, cache existente completo. Treino misto NÃO iniciado. Próxima hipótese:dataset-only1k contra Skip+Frequency1k, todos os demais fatores pareados. Não misturar Codec50/crop/grupo/loss/scheduler.','',
        '## 9. Problemas e verificações','',
        '165 testes nativos passaram. Nenhuma falha HIP nesta sequência C; retries controlados continuam disponíveis. Defeito de label e salvamento limitado a3 previews corrigidos sem mexer em pesos/protocolo. Tentativa inicial de motion builder encontrou apenas3 frames salvos; pasta vazia preservada, substituída pela geração explícita de material completo em outra pasta.','',
        '## 10. Cinco pontos para revisão humana','',
        '1. C ainda perde V3 visual/temporal em dois trechos, apesar de se aproximar nas métricas fixas.',
        '2. O ganho no1200s não certifica recuperação de detalhes; assistir os painéis e comparar luminosidade, dedos, olhos e cabelo.',
        '3. C custa cerca de3,08× os MACs do V3;720→1080 permanece fora do orçamento de30fps de inferência.',
        '4. Dataset-only V2 é a próxima hipótese controlada preparada; não começar15k ou combinar fatores sem evidência.',
        '5. Antes de conclusão de release:outra seed/material independente, percepção em movimento, player end-to-end e final-hidden somente após protocolo congelado.', '',
        'Relatórios anteriores abaixo mantêm suas medições históricas; estados antigos de running/failure não substituem esta conclusão atual.','']
    overnight = ROOT/'docs/zenkai-overnight-report.md'
    old = overnight.read_text(encoding='utf-8')
    marker = '# Latest completed round: Codec25/50 3k'
    overnight.write_text('\n'.join(lines)+old[old.index(marker):],encoding='utf-8')
    progress = ROOT/'docs/zenkai-progress.md'
    old = progress.read_text(encoding='utf-8')
    prefix = '# Current round COMPLETE: C5k, 2026-10-04\n\nC5000/5000 completed exit0. Fixed160, same144 video, gradients,50/500 latency, FP32/FP16/ONNX exports and reloadedROCm parity PASS. Advancement against HistoricalV3 FAIL:MAE/Edge regress in all five domains and visual/temporal regress in two of three clips. No Stage2/promotion/default/hidden access.165 native tests PASS;20 recovery hashes and full state/master/best verified.\n\n[Results](zenkai-c5k-results.md), [decision](zenkai-c5k-decision.md), [gradients](zenkai-c5k-gradient-analysis.md), [still review](zenkai-c5k-visual-review.md), [remaining validation](zenkai-remaining-validation.md). Experimental ZIP under `models/zenkai/review-packages/c5k`; lossless motion-review material under `diagnostics/zenkai-c5k/motion-review-v2`, prepared but not human/blind-reviewed.\n\nScheduler constant NOT SELECTED. V2 union2394 TRAIN prepared,602 new TRAIN sources/pairs decoded and verified; no mixed-source run started, C dataset preserved. Older entries below preserve historical states.\n\n'
    progress.write_text(prefix+old[old.index(marker):],encoding='utf-8')
    for filename, text in [('zenkai-screening-results.md','C5k completed all operations; advancement againstHistoricalV3 FAIL. Stage2NO; no promotion. See [C5k results](zenkai-c5k-results.md) and [decision](zenkai-c5k-decision.md).'),
                           ('zenkai-real-video.md','C5k same144-frame evaluation complete; visual/temporal regression in2/3 clips againstV3. Identity bug corrected with complete new CPU run; all originals preserved. Still observations and prepared lossless motion clips are distinct from human perceptual review. See [results](zenkai-c5k-results.md) and [visual review](zenkai-c5k-visual-review.md).'),
                           ('zenkai-gradient-analysis.md','C5k fixed clean/heavy backward complete; all four internal blocks active in FP32/scaled AMP. Actual3k→5k accumulated changes reported separately from per-step updates. See [C5k analysis](zenkai-c5k-gradient-analysis.md).')]:
        path = ROOT/'docs'/filename
        path.write_text('# Latest C5k round, 2026-10-04\n\n'+text+'\n\n'+path.read_text(encoding='utf-8'),encoding='utf-8')
    print('Final C5k overnight/progress/links saved; previous reports preserved')


if __name__ == '__main__':
    main()
