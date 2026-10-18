# FSR 1 — fila GPU e modo nativo 1:1

Data: 10/10/2026. Implementação no caminho existente WebGPU, sem alterar pesos,
presets, denoise padrão ou os shaders EASU/RCAS. Adapter do teste: AMD RDNA1.

## Alterações

`FsrEngine.processFrame` só conclui após a fila GPU terminar o quadro. O
fluxo serial mantém um quadro em voo; os recursos não são reconfigurados
enquanto o quadro anterior ainda está sendo processado. `totalMs` agora inclui
submissão e conclusão GPU, em vez de medir apenas envio de comandos. Ainda
não inclui decodificação, captura externa ou scanout da tela. O FPS do player
deve ser medido separadamente.

Há timeout de 10 segundos, encerramento do device e erro persistente quando
a conclusão falha. Não há retry ou loop de reinicialização. Isso limita o
acúmulo de comandos; não é prova de que todas as causas dos crashes foram
resolvidas. O ASR que compartilha o engine também recebe essa proteção.

No FSR 1 com entrada igual à saída, EASU não é executado. O RCAS recebe a
fonte ou o resultado da limpeza diretamente. Isso elimina uma passagem e
uma textura RGBA8 de saída (aproximadamente 31,6 MiB em 4K). Não modifica
EASU/RCAS em casos de ampliação; ASR mantém seu caminho de reconstrução.

## Qualidade medida

Três trechos de 48 frames, 144 total, provenientes do conjunto preparado
`rezero-real-01`. Nesta avaliação os frames HR 720p são usados como fonte e
referência para avaliar **720p→720p**, não reconstrução 360p→720p. Denoise
desligado; RCAS permanece ativo. Menor MAE é melhor. Erro temporal é a
diferença de deltas entre saídas e referência, sem warping.

| Trecho | MAE anterior | MAE atual | Temporal anterior | Temporal atual |
|---|---:|---:|---:|---:|
| 60 s | 0,004109 | 0,003910 | 0,000440 | 0,000345 |
| 600 s | 0,003520 | 0,003163 | 0,003617 | 0,002706 |
| 1200 s | 0,000400 | 0,000378 | 0,000089 | 0,000066 |

Redução de MAE: 4,8–10,1%; redução de erro temporal: 21,5–26,6%. Não é um
ganho geral de nitidez/restauração nem evidência de remoção adicional de ruído.
O modo 1:1 altera alguns pixels de contorno: diferença máxima anterior/atual
43 / 66 / 28 níveis de 8 bits por trecho, não apenas diferenças de arredondamento.
Os crops fonte/anterior/atual foram gerados e inspecionados; esta rodada não
inclui estudo perceptual humano nem LPIPS/SSIM.

Um teste GPU de ampliação 640×360→1280×720, com limpeza equilibrada, apresentou
diferença máxima **zero** entre versões. É uma verificação de paridade nesse
caso, não uma nova avaliação completa de upscale em 144 frames.

## Tempo medido

Comparação controlada de tempo concluído: baseline antigo recebe uma espera
GPU no diagnóstico; versão nova já espera internamente. Nenhuma recebe duas
esperas. Oito warmups + 30 frames sintéticos distintos por caso; denoise
desligado, preset FSR Nativo. GPU usada sequencialmente, sem frame reuse.

| Fonte → saída | Mediana anterior / atual | P95 anterior / atual |
|---|---:|---:|
| 1080p→1080p | 3,9 / 3,6 ms | 4,1 / 4,1 ms |
| 4K→4K | 7,1 / 6,1 ms | 9,7 / 6,2 ms |
| 1080p→4K | 7,0 / 6,1 ms | 10,2 / 9,3 ms |

Resultados da execução com espera pareada em
`diagnostics/fsr1-improvements-matched-20261010.json`. Houve variação entre
execuções: em uma rodada anterior, 1080p→4K marcou 4,6 / 5,5 ms de mediana.
Não se deve concluir aceleração garantida para upscale, onde a quantidade de
passes permaneceu igual. São amostras curtas, não benchmark de 500 frames e
não FPS de player. Falta teste integrado de reprodução/decodificação em site.

## Verificações e reprodução

- 167 testes JS passaram, incluindo bloqueio do segundo quadro até conclusão,
  falha persistente da GPU, caminho 1:1 e regressões ASR/FSR existentes.
- Browser Chromium: shader compilation, readback, reconfiguração e 4K passaram;
  sem erros de validação GPU. Warnings de powerPreference/readback permanecem
  registrados nos JSON, não ocultados.
- Baseline preservado em `diagnostics/fsr1-baseline-engine-20261010.js`.
- HTTP de teste serve apenas assets permitidos e o conjunto de avaliação;
  não serve credenciais. Nenhum serviço Node permanente foi iniciado.

Para reproduzir, execute `diagnostics/serve-fsr3-check.py` e depois
`node diagnostics/check-fsr-render.cjs http://127.0.0.1:8765/diagnostics/fsr1-improvement-check.html diagnostics/fsr1-check.json`.
Os testes GPU devem rodar isoladamente. A extensão carrega diretamente o
código do projeto; recarregue-a e reabra o viewer para aplicar a mudança.

## Próximos gargalos

O FSR 1 ainda copia a fonte inteira, e a limpeza 5×5 ainda acontece no tamanho
original. Reduzir essas etapas exige avaliar qualidade em material ruidoso e
lineart. Outra hipótese é a atenuação de sharpening em ruído disponível no
RCAS da AMD; ainda não foi incorporada nesta rodada. Referência oficial:
[FSR1](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-spatial/).
