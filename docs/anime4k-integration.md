# Anime4K S/M e combinação com RealESR — 04/10/2026

Integração experimental no backend ROCm, em FP16, na Radeon RX 5500 XT (`gfx1012`), PyTorch 2.12.0+rocm7.15.0a20260728, HIP 7.15.26290. Os shaders originais e o checkpoint RealESR foram preservados. Nenhum modelo ou preset foi promovido a padrão. Zenkai permaneceu pausado.

Opções disponíveis: `anime4k-s`, `anime4k-m`, `realesr-anime4k-s-360`. Recarregue a extensão, abra `Iniciar-ROCm.cmd` e selecione uma dessas opções. TensorRT e WebGPU não executam este port; a interface mantém ROCm e desabilita TensorRT para estas opções.

A combinação reduz a entrada do RealESR para no máximo 640×360, usa sua saída 2× e aplica Anime4K S em até quatro passes 2× para completar a ampliação solicitada, com resize bilinear final quando necessário. Essa redução pode eliminar detalhes da fonte. Em 360→720, Anime4K não precisa executar: a combinação produz exatamente a mesma saída do RealESR 2×, mas nesta implementação sem captura HIP e com possível custo adicional. Não oferece ganho de qualidade ou velocidade nesse caso.

## Latência

50 aquecimentos e 500 entradas distintas por caso; sincronização GPU após cada inferência. Mediana / P95 em ms, de uma única execução completa. Inclui o modelo e o resize solicitado, exclui conversão de entrada, cópias, socket e navegador. Inferência sem resize e resize isolado: **NOT MEASURED**. Os números não são FPS do player.

| Entrada → saída | RealESR | Anime4K S | Anime4K M | Combinação 360p |
|---|---:|---:|---:|---:|
| 360p → 720p | 40.53 / 46.84 | 3.91 / 4.59 | 8.47 / 9.84 | 44.50 / 48.11 |
| 720p → 1080p | 162.54 / 177.35 | 11.48 / 12.28 | 25.58 / 27.07 | 55.84 / 60.20 |
| 1080p → 2160p | 378.29 / 399.34 | 23.96 / 25.22 | 56.03 / 57.86 | 102.35 / 104.15 |

A configuração de captura é a mesma usada no renderer: RealESR com HIP graph somente em entradas ≤640×360; novas opções em execução normal, sem graph. Não se usou frame reuse. A variação do ambiente é visível nos mínimos/máximos; consultar os JSON completos. Não misturar esta rodada com números históricos medidos em outro momento.

Renderer real: 5 aquecimentos + 30 entradas distintas, incluindo conversão, cópias, readback e serialização RGB. Exclui socket e navegador. Mediana / P95 em ms:

| Entrada → saída | RealESR | Anime4K S | Anime4K M | Combinação 360p |
|---|---:|---:|---:|---:|
| 360p → 720p | 49.13 / 50.63 | 7.34 / 9.12 | 11.60 / 13.01 | 51.83 / 53.55 |
| 720p → 1080p | 191.13 / 195.52 | 17.83 / 19.64 | 33.90 / 35.71 | 65.85 / 69.49 |
| 1080p → 2160p | 420.84 / 427.27 | 49.25 / 51.83 | 81.37 / 85.14 | 125.32 / 129.06 |

## Qualidade e estabilidade temporal

Mesmos três manifests, hashes e 144 frames reais usados na avaliação anterior: 48 frames por trecho, 360→720. Referências 720p provenientes do vídeo local. Todas as opções usam a mesma ordem, inputs e referências. Os arquivos de origem/frames foram verificados antes da execução. Valores normalizados em [0,1]; SSIM usa janela uniforme 11×11, não MS-SSIM. PSNR é a média do PSNR por frame. LPIPS: **NOT MEASURED**.

Métricas individuais, sem composite. MAE, MSE, Edge MAE, Laplacian MAE e erro temporal: menor é melhor. SSIM e PSNR: maior é melhor. Combinação em 360→720 = RealESR, confirmado pelas métricas por frame.

### clip-1200s-360-to-720

| Modelo | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | Edge MAE ↓ | Laplacian MAE ↓ | Temporal MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bilinear | 0.002354 | 0.000029 | 46.598447 | 0.994574 | 0.001600 | 0.005604 | 0.000431 |
| realesr-anime | 0.001886 | 0.000032 | 45.639851 | 0.994535 | 0.001733 | 0.005925 | 0.000394 |
| anime4k-s | 0.002262 | 0.000027 | 46.828175 | 0.994064 | 0.001723 | 0.006067 | 0.000425 |
| anime4k-m | 0.002213 | 0.000026 | 46.870851 | 0.995325 | 0.001621 | 0.005649 | 0.000425 |
| realesr-anime4k-s-360 | 0.001886 | 0.000032 | 45.639851 | 0.994535 | 0.001733 | 0.005925 | 0.000394 |

### clip-600s-360-to-720

| Modelo | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | Edge MAE ↓ | Laplacian MAE ↓ | Temporal MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bilinear | 0.012927 | 0.000685 | 31.939438 | 0.923407 | 0.009365 | 0.026722 | 0.010304 |
| realesr-anime | 0.016032 | 0.000607 | 32.404713 | 0.938572 | 0.008703 | 0.025626 | 0.009264 |
| anime4k-s | 0.012514 | 0.000581 | 32.791690 | 0.925781 | 0.009244 | 0.026506 | 0.010061 |
| anime4k-m | 0.012364 | 0.000564 | 32.927261 | 0.927719 | 0.008996 | 0.025785 | 0.009843 |
| realesr-anime4k-s-360 | 0.016032 | 0.000607 | 32.404713 | 0.938572 | 0.008703 | 0.025626 | 0.009264 |

### clip-60s-360-to-720

| Modelo | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | Edge MAE ↓ | Laplacian MAE ↓ | Temporal MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bilinear | 0.010854 | 0.000523 | 32.816761 | 0.944099 | 0.009033 | 0.028928 | 0.001135 |
| realesr-anime | 0.013125 | 0.000379 | 34.219864 | 0.961991 | 0.007642 | 0.025031 | 0.000859 |
| anime4k-s | 0.009742 | 0.000321 | 34.934298 | 0.958210 | 0.008319 | 0.027010 | 0.000977 |
| anime4k-m | 0.009388 | 0.000287 | 35.419222 | 0.961528 | 0.007963 | 0.025995 | 0.000921 |
| realesr-anime4k-s-360 | 0.013125 | 0.000379 | 34.219864 | 0.961991 | 0.007642 | 0.025031 | 0.000859 |

O erro temporal compensa a diferença entre referências de frames adjacentes. Não usa optical flow; não certifica ausência de flicker nem mede o modo híbrido de apresentação do navegador. S/M melhoram algumas métricas de reconstrução, mas o RealESR tem erros de borda e temporais menores nos trechos 60s/600s. Não existe vencedor universal de qualidade.

Inspeção dos crops estáticos: no trecho 60s, RealESR remove melhor os artefatos e reforça olhos/cabelo/contornos; S/M ficam visualmente mais próximos do bilinear, com M melhorando contornos em relação ao S. As métricas de reconstrução favoráveis não devem ser interpretadas sozinhas como preferência visual.

Crops lado a lado de frames 0 e 24 em cada trecho: `diagnostics/anime4k-integration/clip-*-frame*.png`. Crops 720→1080: `*-1080p-model-difference.png`. Estes últimos comparam **diferenças entre modelos**, sem referência 1080p pareada; não certificam qualidade em 1080p/4K. Não há avaliação temporal 4K nesta rodada.

## Validação da implementação

- Oracle independente em NumPy interpreta diretamente matrizes GLSL column-major e amostragem clamp-edge. CPU FP32 S/M coincide dentro de atol 3e-6 / rtol 2e-5 nos exemplos testados. Não utiliza conv2d nem o parser de produção para a convolução de referência.
- Parser rejeita operandos alterados, offsets fracionários, constantes não finitas, mudanças de dimensões e mudanças na equação de saída suportada. O registro também verifica manifests e hashes de cada componente antes de carregar.
- CPU FP32 vs ROCm FP16: erro máximo 0.00384837; pior erro médio 0.00037124. Tolerância: máximo 2/255 e médio 0.0005; seis casos passaram, incluindo passes repetidos.
- Paridade com execução original mpv/GLSL na GPU: **NOT MEASURED**. A portabilidade de equações não é prova de igualdade com todas as configurações de sampler/resize do mpv.
- WebSocket autenticado real: 6 casos passaram, com inputs distintos, troca de tamanho/modelo e comparação à referência CPU FP32. Não houve reuse.
- Suites: 197 testes Python e 141 testes JavaScript passaram. Revisão independente: duas observações importantes corrigidas e verificadas; nenhum achado crítico restante.

## Conclusão prática

Use Anime4K S quando a prioridade for menor custo; M é uma opção intermediária a comparar visualmente. A combinação oferece redução de custo frente ao RealESR com entrada maior, sacrificando informação de alta resolução. RealESR existente continua disponível para restauração mais forte. Todos os modos novos permanecem experimentais. Não se promete 30 FPS end-to-end em 4K ou em qualquer player sem medir o navegador.

## Reprodutibilidade

Reports completos preservados em `diagnostics/anime4k-integration/{parity,quality,performance,renderer}-attempt-1.json`, logs correspondentes, `stream-validation.json` e `final-verification.json`. Cada relatório contém versões, hashes de fontes e artefatos. A falha de um processo HIP invalida apenas aquela operação; retries usam processo novo, até original + 2, sem juntar timings parciais.

Com o serviço GPU parado, execute `"C:\Program Files\Python310\python.exe" diagnostics/anime4k-integration/run.py`. Escolha um `--attempt-start` novo para preservar os relatórios existentes. Nunca execute o benchmark disputando GPU com o serviço/player ou outro experimento.

Anime4K original: https://github.com/bloc97/Anime4K . As licenças MIT originais permanecem nos GLSL. O port é PyTorch/ROCm e não requer instalação do mpv.
