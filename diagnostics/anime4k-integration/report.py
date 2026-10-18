"""Generate the human report from complete measurements, never partial timings."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(name):
    data=json.loads((HERE/name).read_text(encoding='utf-8'))
    if data['status']!='PASS':raise ValueError('Complete PASS operation required: '+name)
    return data
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read('performance-attempt-1.json');q=read('quality-attempt-1.json')
r=read('renderer-attempt-1.json');parity=read('parity-attempt-1.json');stream=read('stream-validation.json')
ids=['realesr-anime','anime4k-s','anime4k-m','realesr-anime4k-s-360']
labels=['RealESR','Anime4K S','Anime4K M','RealESR 360p + Anime4K S']
text=['# Anime4K S/M e combinação com RealESR — 04/10/2026','',
    'Integração experimental no backend ROCm, em FP16, na Radeon RX 5500 XT (`gfx1012`), PyTorch '+p['torch']+', HIP '+p['hip']+'. Os shaders originais e o checkpoint RealESR foram preservados. Nenhum modelo ou preset foi promovido a padrão. Zenkai permaneceu pausado.','',
    'Opções disponíveis: `anime4k-s`, `anime4k-m`, `realesr-anime4k-s-360`. Recarregue a extensão, abra `Iniciar-ROCm.cmd` e selecione uma dessas opções. TensorRT e WebGPU não executam este port; a interface mantém ROCm e desabilita TensorRT para estas opções.','',
    'A combinação reduz a entrada do RealESR para no máximo 640×360, usa sua saída 2× e aplica Anime4K S em até quatro passes 2× para completar a ampliação solicitada, com resize bilinear final quando necessário. Essa redução pode eliminar detalhes da fonte. Em 360→720, Anime4K não precisa executar: a combinação produz exatamente a mesma saída do RealESR 2×, mas nesta implementação sem captura HIP e com possível custo adicional. Não oferece ganho de qualidade ou velocidade nesse caso.','',
    '## Latência','',
    '50 aquecimentos e 500 entradas distintas por caso; sincronização GPU após cada inferência. Mediana / P95 em ms, de uma única execução completa. Inclui o modelo e o resize solicitado, exclui conversão de entrada, cópias, socket e navegador. Inferência sem resize e resize isolado: **NOT MEASURED**. Os números não são FPS do player.','',
    '| Entrada → saída | RealESR | Anime4K S | Anime4K M | Combinação 360p |',
    '|---|---:|---:|---:|---:|']
for size in ([640,360],[1280,720],[1920,1080]):
    rows=[next(row for row in p['cases'] if row['model']==n and row['input']==size) for n in ids]
    text.append('| '+f"{size[1]}p → {rows[0]['output'][1]}p"+' | '+' | '.join(f"{x['medianMs']:.2f} / {x['p95Ms']:.2f}" for x in rows)+' |')
text+=['','A configuração de captura é a mesma usada no renderer: RealESR com HIP graph somente em entradas ≤640×360; novas opções em execução normal, sem graph. Não se usou frame reuse. A variação do ambiente é visível nos mínimos/máximos; consultar os JSON completos. Não misturar esta rodada com números históricos medidos em outro momento.','',
    'Renderer real: 5 aquecimentos + 30 entradas distintas, incluindo conversão, cópias, readback e serialização RGB. Exclui socket e navegador. Mediana / P95 em ms:','',
    '| Entrada → saída | RealESR | Anime4K S | Anime4K M | Combinação 360p |',
    '|---|---:|---:|---:|---:|']
for size in ([640,360],[1280,720],[1920,1080]):
    rows=[next(row for row in r['cases'] if row['model']==n and row['input']==size) for n in ids]
    text.append('| '+f"{size[1]}p → {rows[0]['output'][1]}p"+' | '+' | '.join(f"{x['medianMs']:.2f} / {x['p95Ms']:.2f}" for x in rows)+' |')
text+=['','## Qualidade e estabilidade temporal','',
    'Mesmos três manifests, hashes e 144 frames reais usados na avaliação anterior: 48 frames por trecho, 360→720. Referências 720p provenientes do vídeo local. Todas as opções usam a mesma ordem, inputs e referências. Os arquivos de origem/frames foram verificados antes da execução. Valores normalizados em [0,1]; SSIM usa janela uniforme 11×11, não MS-SSIM. PSNR é a média do PSNR por frame. LPIPS: **NOT MEASURED**.','',
    'Métricas individuais, sem composite. MAE, MSE, Edge MAE, Laplacian MAE e erro temporal: menor é melhor. SSIM e PSNR: maior é melhor. Combinação em 360→720 = RealESR, confirmado pelas métricas por frame.','']
for seq in q['sequences']:
    text+=['### '+Path(seq['manifest']).parent.name,'',
        '| Modelo | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | Edge MAE ↓ | Laplacian MAE ↓ | Temporal MAE ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['bilinear']+ids:
        m=seq['summary'][name]
        text.append('| '+name+' | '+' | '.join(f"{m[k]:.6f}" for k in ['mae','mse','psnr','ssim','edgeMae','laplacianMae','temporalMae'])+' |')
    text+=['']
text+=['O erro temporal compensa a diferença entre referências de frames adjacentes. Não usa optical flow; não certifica ausência de flicker nem mede o modo híbrido de apresentação do navegador. S/M melhoram algumas métricas de reconstrução, mas o RealESR tem erros de borda e temporais menores nos trechos 60s/600s. Não existe vencedor universal de qualidade.','',
    'Inspeção dos crops estáticos: no trecho 60s, RealESR remove melhor os artefatos e reforça olhos/cabelo/contornos; S/M ficam visualmente mais próximos do bilinear, com M melhorando contornos em relação ao S. As métricas de reconstrução favoráveis não devem ser interpretadas sozinhas como preferência visual.','',
    'Crops lado a lado de frames 0 e 24 em cada trecho: `diagnostics/anime4k-integration/clip-*-frame*.png`. Crops 720→1080: `*-1080p-model-difference.png`. Estes últimos comparam **diferenças entre modelos**, sem referência 1080p pareada; não certificam qualidade em 1080p/4K. Não há avaliação temporal 4K nesta rodada.','',
    '## Validação da implementação','',
    '- Oracle independente em NumPy interpreta diretamente matrizes GLSL column-major e amostragem clamp-edge. CPU FP32 S/M coincide dentro de atol 3e-6 / rtol 2e-5 nos exemplos testados. Não utiliza conv2d nem o parser de produção para a convolução de referência.',
    '- Parser rejeita operandos alterados, offsets fracionários, constantes não finitas, mudanças de dimensões e mudanças na equação de saída suportada. O registro também verifica manifests e hashes de cada componente antes de carregar.',
    f"- CPU FP32 vs ROCm FP16: erro máximo {max(x['maxAbsError'] for x in parity['cases']):.8f}; pior erro médio {max(x['meanAbsError'] for x in parity['cases']):.8f}. Tolerância: máximo 2/255 e médio 0.0005; seis casos passaram, incluindo passes repetidos.",
    '- Paridade com execução original mpv/GLSL na GPU: **NOT MEASURED**. A portabilidade de equações não é prova de igualdade com todas as configurações de sampler/resize do mpv.',
    f"- WebSocket autenticado real: {len(stream['cases'])} casos passaram, com inputs distintos, troca de tamanho/modelo e comparação à referência CPU FP32. Não houve reuse.",
    '- Suites: 197 testes Python e 141 testes JavaScript passaram. Revisão independente: duas observações importantes corrigidas e verificadas; nenhum achado crítico restante.',
    '', '## Conclusão prática','',
    'Use Anime4K S quando a prioridade for menor custo; M é uma opção intermediária a comparar visualmente. A combinação oferece redução de custo frente ao RealESR com entrada maior, sacrificando informação de alta resolução. RealESR existente continua disponível para restauração mais forte. Todos os modos novos permanecem experimentais. Não se promete 30 FPS end-to-end em 4K ou em qualquer player sem medir o navegador.','',
    '## Reprodutibilidade','',
    'Reports completos preservados em `diagnostics/anime4k-integration/{parity,quality,performance,renderer}-attempt-1.json`, logs correspondentes, `stream-validation.json` e `final-verification.json`. Cada relatório contém versões, hashes de fontes e artefatos. A falha de um processo HIP invalida apenas aquela operação; retries usam processo novo, até original + 2, sem juntar timings parciais.','',
    'Com o serviço GPU parado, execute `"C:\\Program Files\\Python310\\python.exe" diagnostics/anime4k-integration/run.py`. Escolha um `--attempt-start` novo para preservar os relatórios existentes. Nunca execute o benchmark disputando GPU com o serviço/player ou outro experimento.','',
    'Anime4K original: https://github.com/bloc97/Anime4K . As licenças MIT originais permanecem nos GLSL. O port é PyTorch/ROCm e não requer instalação do mpv.','']
(ROOT/'docs/anime4k-integration.md').write_text('\n'.join(text),encoding='utf-8')
paths=[HERE/n for n in ['parity-attempt-1.json','quality-attempt-1.json','performance-attempt-1.json','renderer-attempt-1.json','stream-validation.json','python-tests.log','js-tests-final.log']]
sources=['native/anime4k_model.py','native/anime4k_registry.py','native/model_registry.py','popup/popup.js','popup/popup.html','src/core/settings.js','src/core/local-models.js','src/engine/model-config.js','diagnostics/anime4k-integration/measure.py','diagnostics/anime4k-integration/run.py']
verification={'status':'PASS','scope':'Complete operation reports, CPU suites and independent review; not model promotion or browser FPS',
    'reports':{str(path.relative_to(ROOT)):sha(path) for path in paths},'sources':{p:sha(ROOT/p) for p in sources},
    'defaultChanged':False,'shaderWeightsChanged':False,'release':'experimental','pythonTests':197,'javascriptTests':141}
(HERE/'final-verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print('Reports generated from complete measured operations.')
