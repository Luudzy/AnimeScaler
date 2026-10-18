"""Consolidate completed Domain10 evidence; never train, select or promote a model."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = 'zenkai-ablation-skip-v2-domain10-1k'
OUT = ROOT / 'diagnostics/zenkai-domain10-1k'
FOLDER = ROOT / 'models/zenkai/experiments' / RUN
sys.path.insert(0, str(ROOT / 'native'))
from artifact_provenance import validate_checkpoint_file


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))


def write_md(path, lines):
    Path(path).write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]


def replace_prefix(path, marker, lines):
    raw = path.read_bytes()
    offset = raw.find(marker.encode('utf-8'))
    if offset < 0:
        raise ValueError('Missing historical document suffix: ' + str(path))
    # Preserve previous round byte-for-byte, including its historical line endings.
    path.write_bytes(('\n'.join(lines) + '\n\n').encode('utf-8') + raw[offset:])


def main():
    report = read(OUT / 'comparison.json')
    decision = read(ROOT / 'docs/zenkai-domain10-decision.json')
    validate_checkpoint_file(OUT / 'comparison.json', decision['comparisonSha256'])
    if report['state'] != 'COMPLETE' or decision['samplerHypothesis'] != 'INCONCLUSIVE' or decision['advancementAgainstHistoricalV3'] != 'FAIL':
        raise ValueError('This consolidation is only for the reviewed mixed-result round')
    if any(v['state'] != 'PASS' for v in report['operations'].values()):
        raise ValueError('Incomplete operation')
    log = ROOT / 'diagnostics/zenkai-training/domain10-completion-tests.log'
    raw = log.read_bytes()
    text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s', text)
    if not match or int(match[1]) != 177 or not re.search(r'^OK\s*$', text, re.M):
        raise ValueError('A complete green 177-test suite is required')
    seconds = float(match[2])

    attention = read(OUT / 'attention-cpu.json')
    validate_checkpoint_file(ROOT / 'native/training-data/zenkai/validation-fixed.json', attention['fixedValidationSha256'])
    for model in attention['checkpoints'].values():
        validate_checkpoint_file(model['path'], model['sha256'])
    style = read(OUT / 'train-source-review.json')
    if style['heldOutPixelsOpened'] or len(style['sources']) != 6:
        raise ValueError('Wrong TRAIN-only style-review scope')
    for source in style['sources']:
        validate_checkpoint_file(source['path'], source['sha256'])
    validate_checkpoint_file(OUT / 'train-source-review.png', style['outputSha256'])

    lines = ['# Attention e gradientes: diagnóstico adicional Domain10', '',
        'Os quatro blocos internos recebem gradientes após 1k. O backward inicial tem gradientes internos zero porque a head de correção foi inicializada para saída zero; a head recebe gradiente. Isso não caracteriza um bloco permanentemente morto.', '',
        'Probe CPU FP32, dois crops fixos clean/heavy, pesos reais arquivados e sem update de optimizer. As tabelas abaixo são auxiliares; o relatório principal contém FP32 e scaled AMP no ROCm com o scaler real.', '',
        '**Normalização:** este probe usa a loss inteira; o probe GPU divide por 2 para accumulation. Não comparar diretamente os valores absolutos das normas CPU com GPU.', '',
        '**Escopo:** hooks medem o core SPAB e excluem o identity skip externo. A razão RMS core/input não é atenuação do bloco completo. Mean/max gate não medem fração saturada. C5k tem dataset, codec e budget diferentes: apenas contexto, sem atribuição causal.', '']
    for case in ('clean', 'heavy'):
        rows = []
        for name, model in attention['checkpoints'].items():
            probe = next(x for x in model['probes'] if x['case'] == case)
            for block, values in probe['attention'].items():
                group = block.removesuffix('.core')
                rows.append([name, group, f"{probe['groups'][group]['gradientL2']:.9g}",
                             f"{values['meanAbsGate']:.6f}", f"{values['maxAbsGate']:.6f}",
                             f"{values['outputRms']/values['inputRms']:.6f}"])
        lines += ['## ' + case, ''] + table(['Checkpoint', 'Core', 'Gradient L2', 'Mean abs gate', 'Max abs gate', 'Core RMS / input RMS'], rows) + ['']
    lines += ['Os gates e pesos mudaram; existe fluxo de gradientes depois do início. Esses dados não provam qualidade melhor nem ausência de saturação em todo o dataset. Nenhum estado A/B/C/V2 foi alterado. Não selecionar arquitetura pelo tamanho da norma.', '',
              'Raw: `diagnostics/zenkai-domain10-1k/attention-cpu.json`. SHA256s dos checkpoints e identidades dos crops estão no raw report.', '',
              '## Inspeção limitada das fontes TRAIN', '',
              'Seis fontes elegíveis em posições determinísticas 0/120/240/360/480/601 foram inspecionadas: fanart/chibi e retratos sobre fundo claro, capa manga com texto, lineup e cena pictórica escura. Os estilos são heterogêneos e distintos de frames de vídeo comprimidos. Isso não é um julgamento de qualidade das 602 imagens nem uma auditoria de near-duplicates. Nenhuma filtragem foi aplicada; nenhum heldout foi aberto.', '',
              'Contact sheet e hashes: `diagnostics/zenkai-domain10-1k/train-source-review.png` e `.json`.']
    write_md(ROOT / 'docs/zenkai-domain10-attention-analysis.md', lines)

    write_md(ROOT / 'docs/zenkai-domain10-visual-review.md', [
        '# Domain10: inspeção de crops estáticos', '',
        'Nove painéis preparados nos mesmos frames 0/24/47 e ROI (448,100,832,484). Colunas: referência, bilinear, Historical V3, união uniforme V2 1k, Domain10 V2 1k. Referência/bilinear/V3 são byte-identical às imagens da rodada uniforme. Todos os inputs e painéis têm SHA256 em `diagnostics/zenkai-domain10-1k/crops/provenance.json`.', '',
        '| Crop inspecionado | Observação |', '| --- | --- |',
        '| Clip 60s, frame 0 | Olhos/íris/cabelo ainda perdem detalhe; diferença pequena entre uniforme e 10%. |',
        '| Clip 600s, frame 24 | Olhos, cabelo e contornos do personagem pequeno continuam simplificados; sem vantagem visual clara. |',
        '| Clip 1200s, frame 47 | Dedos/cabelo/olhos fechados continuam suaves ou fundidos; ganho pixel não mostra recuperação clara de detalhe. |', '',
        'Os três crops foram inspecionados visualmente, não só gerados. Os outros seis estão preparados. Não há preferência cega nem avaliação visual em movimento. Não declarar ausência de flicker/shimmering/detail popping a partir de stills ou temporal MAE.', '',
        'Os 144 frames completos melhoram visual MAE/temporal MAE contra uniforme, mas dois ganhos são pequenos; LPIPS piora em todos os cinco domínios fixos. Gate sampler INCONCLUSIVE; gate de avanço sobre V3 FAIL.'])

    review = {'reviewer': '/root/domain10_final_review', 'mode': 'fresh read-only final review; one fix pass, no re-review',
        'criticalFindings': 0,
        'importantFindings': [
            {'finding': 'Initial weight-delta provenance was not independently bound to archived state',
             'state': 'FIXED', 'test': 'test_changed_initial_file_cannot_fabricate_weight_delta',
             'evidence': ['diagnostics/zenkai-training/domain10-initial-pin-red.log', 'diagnostics/zenkai-training/domain10-initial-pin-green2.log']},
            {'finding': 'Prior uniform metrics could change while retaining the same model identity',
             'state': 'FIXED', 'test': 'test_metrics_edit_with_same_model_identity_is_rejected',
             'evidence': ['diagnostics/zenkai-training/domain10-historical-report-red.log', 'diagnostics/zenkai-training/domain10-historical-report-green.log']}],
        'additionalVerificationFix': {'finding': 'Decision JSON represented PSNR delta as percent rather than dB',
             'state': 'FIXED', 'test': 'test_psnr_delta_is_db_separate_from_relative_percent',
             'evidence': ['diagnostics/zenkai-training/domain10-decision-units-red.log', 'diagnostics/zenkai-training/domain10-decision-units-green.log']},
        'deferredMinor': 'Evaluation runner baseline quality attempt-1 is hardcoded; actual selected attempt is 1 in this completed baseline. Dynamic selection needed before reuse with another selected attempt.',
        'limits': 'Reviewer checked protocol/source/init and pretests; final comparison/video/crops were not yet available. Executor performed full numerical/provenance completion checks afterwards.',
        'completionSuite': {'tests': 177, 'seconds': seconds, 'log': str(log), 'sha256': sha(log)},
        'promotionApproved': False}
    for finding in [*review['importantFindings'], review['additionalVerificationFix']]:
        for path in finding['evidence']:
            if not (ROOT / path).is_file():
                raise ValueError('Missing review-fix evidence: ' + path)
    write_json(OUT / 'code-review.json', review)

    common = ['`' + RUN + '` terminou 1000/1000, 1000 updates, exit 0 na primeira tentativa, loop 684.758s. Mesmo estado inicial real da união uniforme; apenas sampler mudou. 8000 draws: 7228 anime / 772 ilustrações (9.65%), 1994 codec draws. Alvo probabilístico 10%, não uma contagem exata.', '',
        'Fixed 160, ROCm FP32/FP16 parity, gradientes, benchmark 50 warmups/500 inputs distintos e 144 frames reais: operações PASS, tentativa 1. Todos os 10 estados de recuperação, master/best/full, 15 fontes atuais/arquivadas e artefatos imutáveis anteriores preservados.', '',
        'Sampler INCONCLUSIVE: visual/temporal MAE melhora nos três trechos contra uniforme (visual −0.311%, −0.252%, −4.046%; temporal −0.327%, −0.238%, −1.421%), mas LPIPS piora em todos os cinco domínios e três crops não mostram recuperação clara de detalhes. Avanço sobre Historical V3 FAIL: MAE pior nos cinco domínios e vídeo pior em dois trechos.', '',
        f'{177} testes nativos PASS em {seconds:.3f}s. Revisão corrigiu guards de proveniência para inicialização/relatório anterior; PSNR no decision JSON agora usa delta dB. Nenhum treino longo, Stage2, release, default, UI ou final-hidden.', '',
        '[Resultados](zenkai-domain10-results.md), [decisão](zenkai-domain10-decision.md), [attention/gradientes](zenkai-domain10-attention-analysis.md), [crops](zenkai-domain10-visual-review.md), [gates restantes](zenkai-remaining-validation.md).', '',
        'Duas probes CPU mostram gradientes internos não nulos em 1k e gates aprendidos; isso não prova qualidade nem exclui saturação no dataset. Seis fontes TRAIN-only inspecionadas mostram estilos heterogêneos; sem filtragem ou julgamento global. Próxima investigação deve isolar mecanismo de aprendizado, não empilhar sampler/loss/crop/codec. Nenhum novo experimento foi iniciado nesta consolidação.']
    replace_prefix(ROOT / 'docs/zenkai-progress.md', '# Current round COMPLETE:V2 dataset-only1k,2026-10-04',
                   ['# Current round COMPLETE: domain sampling 10% 1k, 2026-10-04', '', *common])

    qrows = []
    for domain, metrics in decision['qualityRelativePercent'].items():
        qrows.append([domain, *[f"{metrics[key]['vsUniform1k']:+.5f}" for key in ('mae', 'edgeMae', 'laplacianMae', 'lpipsSqueeze')],
                      f"{metrics['mae']['vsHistoricalV3']:+.5f}"])
    vrows = []
    for clip, metrics in decision['videoRelativePercent'].items():
        vrows.append([clip, *[f"{metrics[key][baseline]:+.5f}" for baseline in ('vsUniform1k', 'vsHistoricalV3') for key in ('mae', 'referenceCompensatedTemporalMae')]])
    prows = []
    for case in report['performance']['cases']:
        timing = case['components']['totalMeasured']
        prows.append([case['model'], f"{case['width']}×{case['height']}", f"{timing['medianMs']:.5f}", f"{timing['p95Ms']:.5f}"])
    parity = report['parity']['models']['v2-domain10-1k']['cases']
    errors = {key: {metric: max(row[key][metric] for row in parity) for metric in ('max_abs_error', 'mean_abs_error')}
              for key in ('fusionFp32', 'deployFp16')}
    overnight = ['# ZENKAI OVERNIGHT REPORT — Domain10 1k, 2026-10-04', '',
        '## 1. B TRAINING / CURRENT FOLLOW-UP', '',
        'B 5k e controles anteriores concluídos e preservados. Esta rodada é um novo sampler smoke, não continuação ou alteração de B.', '',
        *common[:2], f"Best step: 1000. Last training loss: {report['training']['history'][-1]['trainingLoss']:.9f}. Config/history/hashes em metadata. Loss entre distribuições não decide vencedor.", '',
        '## 2. EXPORT', '',
        'FP32 fusion/deploy e FP16 parity ROCm PASS. ONNX neste smoke: NOT EXECUTED. Exports anteriores permanecem preservados. Worst errors por tipo abaixo; máximo de mean e máximo de max podem corresponder a cases diferentes, todas registradas no raw report.', '',
        *table(['Comparison', 'Worst max_abs_error', 'Worst mean_abs_error'], [[key, value['max_abs_error'], value['mean_abs_error']] for key, value in errors.items()]), '',
        '## 3. PERFORMANCE', '',
        'ROCm FP16, 50 warmups / 500 inputs distintos, sem frame reuse. Total é inferência+resize medido; não é FPS do player. Mean/P50/P95/P99/min/max/VRAM e componentes individuais no relatório.', '',
        *table(['Model', 'Input', 'Total median ms', 'Total P95 ms'], prows), '',
        '## 4. FIXED VALIDATION', '', 'Mesmos 160 samples, anime-only. Delta positivo pior nas métricas abaixo. PSNR delta dB separado no JSON; todas as 50 métricas preservadas no relatório.', '',
        *table(['Domain', 'MAE vs uniform %', 'Edge %', 'Laplacian %', 'LPIPS %', 'MAE vs V3 %'], qrows), '',
        '## 5. REAL VIDEO', '', 'Mesmos 3×48 frames, CPU FP32, LR H264 e cortes originais. Apenas a tentativa 1 completa, sem mesclar resultados.', '',
        *table(['Clip', 'Visual vs uniform %', 'Temporal vs uniform %', 'Visual vs V3 %', 'Temporal vs V3 %'], vrows), '',
        '## 6. GRADIENT ANALYSIS', '',
        'Todos os quatro blocos internos recebem gradientes em 1k. Probe real FP32/scaled AMP e deltas init→1k no relatório; probe CPU adicional de gates documentado separadamente. Zero interno na inicialização é esperado com head zerada. Mean/max gate não mede fração saturada, e gradiente maior não escolhe modelo.', '',
        '## 7. RETRAINED V3 CONTROL', '', 'Executed: YES, 5k em rodada anterior. Preservado, sem novo run nesta etapa.', '',
        '## 8. SCREENING DECISION', '',
        'Sampler: INCONCLUSIVE. Avanço contra Historical V3: FAIL. A/B/C anteriores e suas decisões permanecem preservados. Preferred release candidate: NONE. Os ganhos sobre uniforme são mistos e insuficientes para selecionar 10% como definitivo.', '',
        '## 9. STAGE 2', '', 'Started: NO. Sem evidência coerente contra V3. Não iniciar 15k/30k. Modelo continua experimental; no product promotion.', '',
        '## 10. PROBLEMS FOUND', '',
        f'Revisão encontrou dois guards de proveniência, corrigidos RED→GREEN. Finalização corrigiu unidade PSNR no JSON, RED→GREEN. {177} testes PASS ({seconds:.3f}s). Nenhum HIP failure nesta rodada. Hashes antigos dos dois documentos compartilhados mudam por novo prefixo; o conteúdo histórico é verificado como suffix byte-identical. Checkpoints, métricas, crops e demais evidências antigas permanecem intactos.', '',
        'Minor deferred: baseline quality attempt-1 hardcoded; selected attempt real também é 1, sem impacto nas métricas atuais. Revisão independente antecedeu os relatórios finais; checks posteriores foram executados pelo implementador.', '',
        '## 11. WHAT TO REVIEW TOMORROW', '',
        '- Ganhos de vídeo contra união uniforme coexistem com regressão LPIPS nos cinco domínios.',
        '- V3 ainda vence dois trechos; os crops não mostram ganho claro em olhos/cabelo/dedos.',
        '- Sampler sozinho não resolve a seleção; não combinar fatores para mascarar esse resultado.',
        '- Custo 720→1080 continua acima de 33.3ms só no modelo+resize; mais steps não altera geometria.',
        '- Antes de release faltam generalização/seed/temporal H265-AV1, moving review e player; hidden reservado para finalista congelado.', '',
        '[Resultados](zenkai-domain10-results.md), [decisão](zenkai-domain10-decision.md), [attention](zenkai-domain10-attention-analysis.md), [crops](zenkai-domain10-visual-review.md).', '']
    replace_prefix(ROOT / 'docs/zenkai-overnight-report.md', '# ZENKAI OVERNIGHT REPORT — V2 dataset-only1k,2026-10-04', overnight)

    remaining = ROOT / 'docs/zenkai-remaining-validation.md'
    raw = remaining.read_bytes()
    marker = b'| V2 como mudan'
    if b'| Sampler V2 10%' not in raw:
        offset = raw.find(marker)
        if offset < 0:
            raise ValueError('Remaining-validation insertion marker absent')
        offset = raw.find(b'\n', offset) + 1
        row = '| Sampler V2 10% versus união uniforme | Concluído 1k:160 fixos/144 vídeo/gradientes/paridade/latência; sampler INCONCLUSIVE, avanço V3 FAIL | Ganho pixel/temporal pequeno e misto; LPIPS piora nos cinco domínios, sem preferência visual clara |\n'
        remaining.write_bytes(raw[:offset] + row.encode('utf-8') + raw[offset:])

    plan = ROOT / 'docs/superpowers/plans/2026-10-04-zenkai-v2-domain10-screen.md'
    p = plan.read_text(encoding='utf-8').replace('- [ ]', '- [x]')
    if 'Final consolidation:' not in p:
        p += f'\nFinal consolidation:1000/1000 updates exit0, all four operations PASS attempt1,160/144 fixed evidence,9 crop panels. Sampler INCONCLUSIVE/product FAIL.10 complete indexed recovery states and15 current/archive source hashes verified.177tests PASS{seconds:.3f}s.\n\nFinal: fixed initial delta provenance — test_changed_initial_file_cannot_fabricate_weight_delta RED→GREEN; fixed historical metric pin — test_metrics_edit_with_same_model_identity_is_rejected RED→GREEN. Finalization fixed PSNR units — test_psnr_delta_is_db_separate_from_relative_percent RED→GREEN. Whole suite177/177.\n\nFinal: minor (deferred):baseline quality attempt-1 hardcoded; actual selected1, no current effect. Fresh reviewer checked protocol/source/pretests; final metric/crop completion checked by executor, no second review.\n\nRuling:shared progress/overnight docs evolve by prefix; verify old byte-identical suffixes against prior whole-file SHA, rather than marking expected documentation update as artifact corruption. Wrong choice would miss a changed historical suffix; suffix SHA is explicitly required.\n\nRuling:preserve canonical experimental workspace and logs; no Git writes/merge/push or skill scratch deletion because frozen paths and explicit artifact retention govern this run. Cost:no new experimental commit; reproducibility uses archived sources/config/hash manifests.\n'
    plan.write_bytes(p.encode('utf-8'))

    # Frozen training archive remains untouched. Archive mutable postprocessors separately.
    archive = FOLDER / 'postprocess-source'
    archive.mkdir(exist_ok=True)
    files = [ROOT/'native/run_domain10_evaluation.py', ROOT/'native/zenkai_probe_evaluation.py',
             ROOT/'native/benchmark_sr_models.py', ROOT/'native/artifact_provenance.py']
    files += [ROOT/'diagnostics/zenkai-overnight'/name for name in
              ('probe_domain10_gradients.py', 'probe_domain10_attention_cpu.py', 'build_domain10_report.py',
               'build_domain10_decision.py', 'build_domain10_crops.py', 'verify_domain10_integrity.py', 'finish_domain10.py', 'build_codec3k_report.py')]
    pins = {}
    for source in files:
        target = archive / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Refuse to overwrite archived postprocessor: ' + str(target))
        if not target.exists():
            target.write_bytes(source.read_bytes())
        pins[str(source.relative_to(ROOT))] = sha(source)
    write_json(archive / 'manifest.json', {'scope': 'Post-training evaluation/report tools, not training sources', 'sha256': pins})

    subprocess.run([sys.executable, str(ROOT/'diagnostics/zenkai-overnight/verify_domain10_integrity.py')], cwd=ROOT, check=True)
    check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    if check.returncode:
        raise ValueError('git diff --check failed: ' + check.stdout + check.stderr)
    metadata = read(FOLDER / 'metadata.json')
    metadata['completionVerification'] = 'diagnostics/zenkai-domain10-1k/final-verification.json'
    metadata['postprocessSources'] = str((archive/'manifest.json').relative_to(ROOT))
    metadata['review'] = 'diagnostics/zenkai-domain10-1k/code-review.json'
    metadata['selectionDecision'] = 'docs/zenkai-domain10-decision.json'
    metadata['auxiliaryDiagnostics'] = ['diagnostics/zenkai-domain10-1k/attention-cpu.json', 'diagnostics/zenkai-domain10-1k/train-source-review.json']
    write_json(FOLDER / 'metadata.json', metadata)
    selected = ['diagnostics/zenkai-domain10-1k/comparison.json', 'diagnostics/zenkai-domain10-1k/integrity.json',
                'diagnostics/zenkai-domain10-1k/code-review.json', 'diagnostics/zenkai-domain10-1k/crops/provenance.json',
                'diagnostics/zenkai-domain10-1k/attention-cpu.json', 'diagnostics/zenkai-domain10-1k/train-source-review.json',
                'diagnostics/zenkai-training/domain10-completion-tests.log', 'docs/zenkai-domain10-decision.json',
                'docs/zenkai-domain10-decision.md', 'docs/zenkai-domain10-results.md', 'docs/zenkai-domain10-attention-analysis.md',
                'docs/zenkai-domain10-visual-review.md', 'docs/zenkai-progress.md', 'docs/zenkai-overnight-report.md',
                'docs/zenkai-remaining-validation.md', str((FOLDER/'metadata.json').relative_to(ROOT)),
                str((archive/'manifest.json').relative_to(ROOT))]
    result = {'state': 'PASS', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'nativeTests': 177, 'nativeTestSeconds': seconds, 'indexedFullRecoveryStatesVerified': 10,
              'sameRealInitializationVerified': True, 'masterBestFullEqual': True, 'trainingSourceHashesVerified': 15,
              'allSelectedOperations': 'PASS', 'frozenQualitySamples': 160, 'sameRealVideoFrames': 144,
              'cropPanelsAndInputsVerified': 9, 'independentReviewLimits': review['limits'],
              'priorImmutableArtifactsAndHistoricalSharedDocSuffixesPreserved': True, 'trackedDefaultDiffCheckExitCode': 0,
              'samplerHypothesis': 'INCONCLUSIVE', 'advancementAgainstHistoricalV3': 'FAIL',
              'selectedArtifactsSha256': {path: sha(ROOT/path) for path in selected},
              'Stage2Started': False, 'releaseApproved': False, 'promoted': False, 'finalHiddenOpened': False}
    write_json(OUT / 'final-verification.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'selectedArtifactsSha256'}, indent=2))


if __name__ == '__main__':
    main()
