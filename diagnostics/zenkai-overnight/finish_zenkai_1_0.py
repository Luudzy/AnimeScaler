"""Save the consolidated candidate and its verified evidence, without promotion."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native'))
from zenkai_1_0 import MODEL_ID, PACKAGE_RELATIVE, SOURCE_MASTER_SHA, digest, validate_package


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))


def write_md(path, lines):
    Path(path).write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows]


def decoded_log(path):
    raw = Path(path).read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')


def main():
    folder = ROOT / PACKAGE_RELATIVE
    manifest = validate_package(folder)
    out = ROOT / 'diagnostics/zenkai-1.0'
    final = out / 'final-verification.json'
    archive = folder.parent / 'zenkai-1.0-experimental.zip'
    if final.exists() or archive.exists():
        raise FileExistsError('Preserve completed candidate and ZIP')
    pipeline = read(ROOT / 'diagnostics/zenkai-1.0-corrected-protocol/pipeline.json')
    if pipeline['state'] != 'COMPLETE' or any(row['state'] != 'PASS' for row in pipeline['operations'].values()):
        raise ValueError('Registered runtime/benchmark must complete before packaging')
    reports = {}
    for name, operation in pipeline['operations'].items():
        path = Path(operation['result']['path'])
        if digest(path) != operation['result']['sha256']:
            raise ValueError('Selected complete result changed: ' + name)
        reports[name] = read(path)
    cpu = read(out / 'cpu-runtime.json')
    onnx = read(out / 'cpu-onnx-runtime.json')
    if cpu['state'] != 'PASS' or onnx['state'] != 'PASS' or len(onnx['cases']) != 7:
        raise ValueError('Incomplete CPU/ONNX registry parity')
    log = ROOT / 'diagnostics/zenkai-training/zenkai-1.0-completion-native-suite.log'
    text = decoded_log(log)
    native = re.search(r'Ran (\d+) tests in ([\d.]+)s', text)
    if not native or int(native[1]) != 183 or not re.search(r'^OK\s*$', text, re.M):
        raise ValueError('Final native suite not green')
    frontend_log = ROOT / 'diagnostics/zenkai-training/zenkai-1.0-frontend-suite.log'
    frontend = decoded_log(frontend_log)
    if not re.search(r'tests 138\s*$', frontend, re.M) or not re.search(r'fail 0\s*$', frontend, re.M):
        raise ValueError('Final frontend suite not green')

    shared = {'docs/zenkai-progress.md': '# Current round COMPLETE: domain sampling 10% 1k, 2026-10-04',
              'docs/zenkai-overnight-report.md': '# ZENKAI OVERNIGHT REPORT — Domain10 1k, 2026-10-04',
              'docs/zenkai-remaining-validation.md': '# Remaining validation gates'}
    previous = read(ROOT / 'diagnostics/zenkai-domain10-1k/final-verification.json')
    for path, pin in previous['selectedArtifactsSha256'].items():
        if digest(ROOT / path) != pin:
            raise ValueError('Previous completed round changed before consolidation: ' + path)
    # All immutable previous report/model/ZIP evidence remains intact.
    old_immutable = []
    for name in ('zenkai-c5k', 'zenkai-v2-dataset-1k'):
        for path, pin in read(ROOT / 'diagnostics' / name / 'final-verification.json')['selectedArtifactsSha256'].items():
            normalized = path.replace('\\', '/')
            if normalized in ('docs/zenkai-progress.md', 'docs/zenkai-overnight-report.md'):
                continue  # These older suffixes were verified at Domain10 completion.
            if digest(ROOT / path) != pin:
                raise ValueError('Prior immutable evidence changed: ' + path)
            old_immutable.append({'path': normalized, 'sha256': pin})

    runtime = reports['runtime']
    benchmark = reports['benchmark']
    rows = []
    for case in benchmark['cases']:
        timing = case['components']['totalMeasured']
        rows.append([case['model'], f"{case['width']}×{case['height']}", f"{case['outputWidth']}×{case['outputHeight']}",
                     f"{timing['medianMs']:.4f}", f"{timing['p95Ms']:.4f}", f"{timing['p99Ms']:.4f}"])
    worst = {key: {stat: max(case[key][stat] for case in runtime['cases'])
                   for stat in ('max_abs_error', 'mean_abs_error')} for key in ('registeredFp32', 'registeredFp16')}
    comparison = read(folder / 'research-comparison.json')
    decision = read(folder / 'research-decision.json')
    qrows = [[domain, *[f"{values[key]:+.4f}%" for key in ('mae', 'edgeMae', 'laplacianMae', 'lpipsSqueeze')]]
             for domain, values in decision['qualityRelativePercentLowerIsBetter'].items()]
    vrows = [[clip, f"{values['mae']:+.4f}%", f"{values['referenceCompensatedTemporalMae']:+.4f}%"]
             for clip, values in decision['videoRelativePercentLowerIsBetter'].items()]
    lines = ['# Zenkai 1.0 — experimental', '',
        'A 1.0 consolida a receita C5k e integra seus artefatos na extensão. **Os pesos são os mesmos C5k já treinados em 5000 steps; não houve novo treino, mistura de checkpoints nem ganho de qualidade novo alegado.**', '',
        'Reúne identity skip gain1, 32 canais/depth5/x2, Frequency loss, codec50, crop128 e scheduler original. GRep foi fusionado para inferência:804648 parâmetros no training graph,154124 no deploy. PyTorch roda FP16 no ROCm; FP32 master e ONNX FP32/FP16 permanecem disponíveis. Isso descreve os recursos consolidados, não uma aceleração nova em relação ao C5k já fusionado.', '',
        'Crop96/group sampler/constant scheduler ficaram fora porque ganhos de clean não transferiram coerentemente. V2/domain10 também ficou fora por resultado misto. Frequency versus Legacy dentro do skip foi misto; Frequency é a escolha da receita, não um vencedor universal.', '',
        '## Como usar', '',
        '1. Recarregue a extensão em `chrome://extensions` ou `edge://extensions`.',
        '2. Reinicie o serviço local por `Iniciar-ROCm.cmd` para carregar o novo registro.',
        '3. No popup, selecione processamento **ROCm local** e **Zenkai 1.0 · 2× FP16 (experimental, local)**.',
        '4. Para começar a comparar, use o preset 720p (IA360→720). O preset1080p usa IA720→1440 seguida de redução para1080, e é mais caro.', '',
        'Nenhum modelo/preset padrão foi alterado. Selecionar este modelo com WebGPU gera a indicação de backend nativo obrigatório; não executa Adore silenciosamente. A escolha está integrada aos controles existentes de viewer/tela cheia, sem alterar seu comportamento.', '',
        'NVIDIA: o registro entrega o ONNX dinâmico ao backend TensorRT existente, que prepara FP16 no ambiente NVIDIA. **TensorRT: NOT MEASURED** neste PC AMD. ONNX/WebGPU não foi integrado nesta rodada. Alvos2K/4K existentes continuam disponíveis, sem nova certificação de qualidade/FPS.', '',
        '## Validação da nova integração', '',
        'CPU PyTorch FP32/FP16, ONNX CPU FP32 e nova carga ROCm FP32/FP16 passaram sete casos, incluindo formas ímpares, contra o training master original. Renderer ROCm passou três frames distintos nos limites reais do protocolo, com erro quantizado zero nos casos medidos, sem frame reuse e com resize1.5. Não é um teste de reprodução no browser/socket ou de vídeo em movimento.', '',
        *table(['Registered ROCm comparison', 'Worst max_abs_error', 'Worst mean_abs_error'], [[key, stats['max_abs_error'], stats['mean_abs_error']] for key, stats in worst.items()]), '',
        'Os exports CPU originais também incluem paridade ONNX FP32/FP16 e reload fusion, com tolerâncias registradas no export report. Nenhuma paridade TensorRT/ONNX GPU foi inventada.', '',
        f'{int(native[1])} testes nativos passaram em {float(native[2]):.3f}s;138 testes frontend passaram. A suíte Node usou `--test-isolation=none` porque o sandbox negou child spawn no runner padrão; o teste foi executado, não removido.', '',
        '## Performance da opção registrada', '',
        'ROCm FP16,50 warmups/500 inputs distintos; relatório completo selecionado, tentativa1. Model inference/post-resize/total separados; mean/P50/P95/P99/min/max/VRAM no raw JSON. Estes totais não são FPS do player.', '',
        *table(['Modelo', 'Entrada', 'Saída', 'Total median ms', 'P95 ms', 'P99 ms'], rows), '',
        'Ambos Zenkai e V3 ficaram mais lentos nesta sessão que nas medições antigas. Os resultados completos foram preservados, sem substituir números ou atribuir uma causa não medida. A1.0 usa a mesma geometria e pesos de C5k; não atribuir essa variação à consolidação.720→1080 permanece acima do orçamento33.3ms apenas no modelo+resize.', '',
        '## Qualidade herdada, sem nova avaliação de treino', '',
        'Mesmos160 casos fixos e144 frames reais do C5k. Pesos/expor​ts são idênticos, e paridade da nova carga foi medida. MAE/Edge/Laplacian/LPIPS:menor melhor; percentuais abaixo são contra HistoricalV3, não contra C5k. Todas as métricas individuais/copias das referências estão no research-comparison.json.', '',
        *table(['Domínio', 'MAE vsV3', 'Edge vsV3', 'Laplacian vsV3', 'LPIPS vsV3'], qrows), '',
        *table(['Trecho', 'Visual MAE vsV3', 'Temporal compensated vsV3'], vrows), '',
        '**Gate de qualidade contra V3: FAIL no screening medido.** A versão1.0 é utilizável como modelo experimental, não uma declaração de superioridade/release estável. Não substitui V3 nem abre final-hidden. Ainda faltam generalização independente, outra seed, temporalH265/AV1, moving review e latência no player.', '',
        '## Artefatos e reprodução', '',
        'Pacote: `models/zenkai/zenkai-1.0`. ZIP: `models/zenkai/zenkai-1.0-experimental.zip`. Manifesto contém pins fixos do master/export/ONNX/evidência. O loader usa o identity-skip exato e recusa arquivo alterado antes de inicializar PyTorch/ORT/TensorRT.', '',
        f'Master de origem SHA256: `{SOURCE_MASTER_SHA}`.', '',
        'Builder: `python native/build_zenkai_1_0.py`, a partir dos artefatos originais do projeto; recusa sobrescrever a1.0. ONNX é portátil; deploy PyTorch exige a arquitetura/loader deste projeto. O pacote de modelo não instala dependências nem o runtime completo.', '',
        '## Problemas e limite de revisão', '',
        'O primeiro smoke da ponte usou inputs pequenos/ímpares que violam seu mínimo40/múltiplo4. Esse erro foi do teste, não HIP/modelo. Teste de regressão RED→GREEN; nova operação em processo limpo, diretório distinto, passou integralmente. Logs antigos não foram apagados nem aproveitados para completar resultados.', '',
        'Revisão independente: NOT EXECUTED — agente revisor atingiu limite de uso. Nenhum achado de revisão independente foi fabricado. Checks de protocolo, hashes, padrões, família, paridade, runtime, benchmark e suítes foram executados pelo implementador; não equivalem a uma revisão externa.', '']
    # Remove accidental invisible typography from report copy.
    lines = [line.replace('\u200b', '') for line in lines]
    write_md(ROOT / 'docs/zenkai-1.0.md', lines)
    write_md(folder / 'README.md', lines)

    review = {'state': 'NOT_EXECUTED', 'reviewer': '/root/zenkai_1_0_review',
              'reason': 'Reviewer usage limit before review; no findings returned',
              'executorChecks': ['exact-source and copied artifacts SHA256', 'identity-skip forward on odd shapes',
                                 'invalid/changed ONNX rejected before backend load', 'frontend/default preserved',
                                 'CPU/ROCm/ONNX registry parity and actual Renderer', 'full suites and benchmark'],
              'fixes': ['HTML first option preserved as animejanai; regression RED→GREEN',
                        'Renderer smoke corrected to min40/multiple4 protocol; regression RED→GREEN'],
              'coverageLimits': ['No independent code review', 'TensorRT unavailable', 'No browser/player or moving-video certification'],
              'releaseApproved': False}
    write_json(out / 'review-status.json', review)
    status = ['# Zenkai 1.0 consolidation COMPLETE, 2026-10-04', '',
        'Pedido do usuário:consolidar os resultados úteis num Zenkai1.0. Pacote/ZIP experimental e opção nativa criados. Mesmo C5k5000:identity skip,Frequency,codec50,patch128,cosine original;fusion FP16. Nenhum novo treino ou peso superior alegado. Crop96/group/constantLR e V2domain10 inconclusivo excluídos.', '',
        'Nova carga CPU/ROCm/ONNX e Renderer passaram;50/500 benchmark completo. Medianas modelo+resize30.7935ms360→720 e100.2703ms720→1080,sem frame reuse;V3 também mais lento nesta sessão. Não é FPS do player. Gate de qualidade herdado FAIL;modelo experimental selecionável,default/presets preservados.', '',
        f'183 testes nativos PASS{float(native[2]):.3f}s;138 frontend PASS. Primeiro smokeRenderer usou shapes inválidos;corrigido RED→GREEN e nova operação completa,logs preservados. Revisão independente NOT EXECUTED por cota do revisor;limitação explícita.', '',
        '[Uso e resultados](zenkai-1.0.md). Sem Stage2/15k/30k/final-hidden/release automático. Registros anteriores preservados abaixo.']
    shared_notes = []
    for path, marker in shared.items():
        target = ROOT / path
        raw = target.read_bytes()
        old_pin = previous['selectedArtifactsSha256'][path]
        if not raw.startswith(marker.encode('utf-8')):
            raise ValueError('Unexpected document prefix: ' + path)
        prefix = status if path != 'docs/zenkai-remaining-validation.md' else [
            '# Zenkai 1.0: atualização dos gates', '',
            'Consolidação C5k, pacotes e integração nativa concluídos;183 testes Python/138 JS e nova paridade/benchmark PASS. Não houve novo treino ou vencedor de qualidade:gateV3 FAIL herdado. Revisão independente pendente por cota. Gates de generalização/player continuam abaixo.', '']
        target.write_bytes(('\n'.join(prefix) + '\n\n').encode('utf-8') + raw)
        content = target.read_bytes()
        suffix = content[content.find(marker.encode('utf-8')):]
        if hashlib.sha256(suffix).hexdigest() != old_pin:
            raise ValueError('Historical shared document bytes changed: ' + path)
        shared_notes.append({'path': path, 'previousWholeDocumentSha256': old_pin,
                             'historicalSuffixVerified': True, 'currentSha256': digest(target)})

    plan = ROOT / 'docs/superpowers/plans/2026-10-04-zenkai-1-0.md'
    p = plan.read_text(encoding='utf-8').replace('- [ ]', '- [x]')
    p += '\nCompletion ledger: package/source hashes unchanged; six new native guards/forward tests pass;183 native and138 frontend tests green. CPU/ONNX/ROCm exact registry parity and realRenderer passed. Full50/500 benchmark PASS. ZIP CRC/member SHA verified separately. No new train/default/preset/hidden/promotion.\n\nRuling: final review attempted once but unavailable due reviewer quota; retain NOT EXECUTED and ship only experimental, not claim independently reviewed. Cost: independent correctness review remains a release gate; verified tests/runtime do not substitute it.\n\nRuling: new registry timing differs from older runs for both models; retain measured values and uncertainty, no fabricated performance ranking or changed benchmark inputs. Cost: no stable speed attribution across sessions.\n\nFinal: bridge smoke bounds defect fixed via test_renderer_smoke_uses_actual_bridge_input_limits RED→GREEN and full183 suite; original failed operation preserved. Node default child spawn EPERM; same full suite executed with isolation=none,138/138.\n'
    plan.write_bytes(p.encode('utf-8'))
    code_folder = folder / 'integration-source'
    code_folder.mkdir()
    code_files = [ROOT/'native'/name for name in ('zenkai_1_0.py', 'build_zenkai_1_0.py', 'model_registry.py', 'export_zenkai_skip.py', 'zenkai_model.py', 'zenkai_ablation_skip.py')]
    code_files += [ROOT/'diagnostics/zenkai-overnight'/name for name in ('validate_zenkai_1_0.py', 'run_zenkai_1_0_checks.py', 'finish_zenkai_1_0.py')]
    code_files += [ROOT/'src/core/local-models.js', ROOT/'popup/popup.html']
    source_pins = {}
    for source in code_files:
        dest = code_folder / source.name
        dest.write_bytes(source.read_bytes())
        source_pins[str(source.relative_to(ROOT))] = digest(source)
    write_json(code_folder / 'manifest.json', {'scope': 'Integration/code provenance snapshot; not a standalone dependency installation', 'sha256': source_pins})
    check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    if check.returncode:
        raise ValueError('git diff --check failed: ' + check.stdout + check.stderr)
    validate_package(folder)
    members = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.rglob('*')):
            if not path.is_file():
                continue
            name = 'zenkai-1.0/' + path.relative_to(folder).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
            members[name] = digest(path)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or len(bundle.namelist()) != len(members):
            raise ValueError('Candidate ZIP integrity failed')
        for name, pin in members.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != pin:
                raise ValueError('Candidate ZIP member changed: ' + name)
    write_json(out / 'zip-integrity.json', {'state': 'PASS', 'path': str(archive), 'sha256': digest(archive), 'memberSha256': members, 'releaseApproved': False})
    result = {'state': 'PASS', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'id': MODEL_ID,
              'version': manifest['version'], 'status': 'experimental', 'newTrainingExecuted': False,
              'sourceCheckpointSha256': SOURCE_MASTER_SHA, 'nativeTests': 183, 'nativeTestSeconds': float(native[2]),
              'frontendTests': 138, 'runtime': 'PASS', 'renderer': 'PASS', 'onnxCpu': 'PASS',
              'benchmark': 'PASS', 'selectedPipeline': 'diagnostics/zenkai-1.0-corrected-protocol/pipeline.json',
              'independentReview': review, 'previousImmutableEvidence': old_immutable,
              'sharedDocuments': shared_notes, 'zipSha256': digest(archive),
              'qualityAdvancementAgainstV3': 'FAIL inherited from C5k', 'trackedDiffCheckExitCode': 0,
              'Stage2Started': False, 'releaseApproved': False, 'promoted': False, 'finalHiddenOpened': False}
    selected = [folder/'manifest.json', archive, ROOT/'docs/zenkai-1.0.md', out/'cpu-runtime.json', out/'cpu-onnx-runtime.json',
                out/'zip-integrity.json', out/'review-status.json', log, frontend_log,
                ROOT/'diagnostics/zenkai-1.0-corrected-protocol/pipeline.json',
                Path(pipeline['operations']['runtime']['result']['path']), Path(pipeline['operations']['benchmark']['result']['path']),
                code_folder/'manifest.json', *[ROOT/path for path in shared]]
    result['selectedArtifactsSha256'] = {str(path.relative_to(ROOT)): digest(path) for path in selected}
    write_json(final, result)
    print(json.dumps({'state': 'PASS', 'package': str(folder), 'zip': str(archive), 'zipSha256': digest(archive),
                      'nativeTests': 183, 'frontendTests': 138, 'independentReview': 'NOT EXECUTED', 'releaseApproved': False}))


if __name__ == '__main__':
    main()
