# FSR / ASR: presets e limpeza de vídeo — 2026-10-09

Atualização 10/10: [fila GPU limitada e caminho nativo 1:1](fsr1-improvements-20261010.md).
O tempo por quadro agora inclui conclusão GPU; em FSR 1 sem mudança de tamanho,
EASU é omitido e o RCAS recebe a fonte/limpeza diretamente.

## Controles implementados

O seletor **Resolução / preset de saída** continua definindo 720p, 1080p,
1440p ou 2160p. O novo seletor **Preset FSR / ASR** define a resolução interna:

| Preset | Fator de upscale nominal | Entrada máxima relativa à saída |
| --- | ---: | ---: |
| Performance | 2× | 50% |
| Balanced | 1,7× | 58,8% |
| Quality | 1,5× | 66,7% |
| Ultra Quality | 1,3× | 76,9% |
| Nativo / AA espacial | 1× | 100% |

Os fatores dos quatro presets FSR 1 seguem a
[documentação AMD](https://gpuopen.com/fidelityfx-superresolution/).
Nativo é um modo espacial deste projeto; não é o AA temporal do FSR 2/4.
A entrada nunca é artificialmente aumentada acima da resolução da origem.
Por exemplo, em 720p→1080p, Quality, Ultra Quality e Nativo usam os mesmos
1280×720 disponíveis. Isso explica por que alguns presets podem coincidir.

Quando há redução, ela ocorre em uma textura GPU com quatro amostras bilineares
para aproximar um filtro de área. A resolução final não muda. Nativo é o
padrão para preservar os pixels disponíveis. Performance pode perder detalhes
e não garante a mesma melhoria de FPS observada em jogos: vídeo já chega
decodificado, e EASU/RCAS continuam executando na resolução de saída.

## Redução de ruído

**Desligada**, **Equilibrada** (padrão) e **Forte**, independentes do preset.
Uma passagem bilateral 5×5 atua na resolução original antes da redução/upscale.
Pesos espaciais e distância RGB limitam a mistura entre cores diferentes,
protegendo contornos e texto de alto contraste. Ela reduz ruído fino e algumas
variações leves de compressão; não reconstrói informação perdida em macroblocos.

Pipeline: cópia GPU → limpeza opcional → redução opcional → EASU → RCAS/ASR.
Texturas e buffers são reutilizados; há uma única submissão por frame.
A troca de intensidade não realoca texturas. Não há histórico temporal,
reuso de frames nem leitura de pixels pela CPU no player. As leituras de pixels
dos testes são exclusivamente offline.

Isso é processamento adicional em torno de FSR 1, não treinamento de FSR 4.
FSR 1 fornece EASU e RCAS; o filtro de vídeo é próprio. O
[FidelityFX Denoiser oficial](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/denoiser/)
foi feito para sombras/reflexos de ray tracing, não uma limpeza universal de streams.

Não há promessa de eliminar 100% do ruído/artefatos preservando todos os detalhes.
Forte pode suavizar texturas e detalhes de baixo contraste. Use equilibrada em
material razoável, forte em material ruidoso, desligada para preservar a origem.

## Testes executados

- Teste GPU com ruído determinístico ±12/255 sobre área plana e linha escura:
  MSE 50,8287 desligado; 15,9967 equilibrado; 4,6862 forte. Redução de MSE de
  68,5% e 90,8%, respectivamente. Intensidade da linha: 7/255 nos três modos.
  **Essa porcentagem vale somente para esse teste sintético**, não todos os vídeos.
- 144 frames reais congelados, em três trechos de 48 frames, 360p→720p.
  Avaliados com origem preservada e, separadamente, com ruído RGB determinístico
  ±12/255 mais JPEG qualidade 0,45. Não equivale a testar H264/H265/AV1 ou uma
  stream capturada ao vivo. Mesmo input e referência para todos os modos.
- FSR forte vs desligado, nos trechos degradados: MAE caiu aproximadamente
  19,6%, 14,3% e 0,9%; erro temporal compensado pela referência caiu
  38,6%, 25,0% e 31,8%. O terceiro trecho teve ganho visual pequeno.
- Em origem sem degradação adicional, equilibrado melhorou MAE nos três trechos,
  mas PSNR caiu no terceiro: FSR 46,495→46,422 dB; ASR 46,588→46,455 dB.
  Essa regressão não é escondida na média.
- Os cinco presets foram renderizados no Chromium com WebGPU AMD RDNA1,
  verificação de pixels e sem erros de validação. Teste de popup confirmou
  seleção, persistência e ocultação dos controles para outros modelos.
- Suíte JavaScript: **159/159 testes passaram**, incluindo mudança dos presets,
  persistência das configurações e troca de intensidade sem realocar texturas.
- Em teste sintético 1080p→4K, Nativo + limpeza forte: 3,59 ms/frame de
  throughput em lote. **Não é latência individual nem FPS do player.**

Relatórios completos, com métricas individuais de FSR e ASR:

- `diagnostics/fsr-denoise-noise-edge.json`
- `diagnostics/fsr-denoise-clean-off-144.json`
- `diagnostics/fsr-denoise-clean-balanced-144.json`
- `diagnostics/fsr-denoise-degraded-{off,balanced,strong}-144.json`
- `diagnostics/fsr-denoise-comparison.json`
- `diagnostics/fsr-preset-{performance,balanced,quality,ultra-quality,native-aa}-check.json`
- `diagnostics/fsr-native-4k-denoise-check.json`

## Reproduzir

Com servidor HTTP local na raiz do projeto, porta 8123:

```powershell
node diagnostics/check-fsr-render.cjs http://127.0.0.1:8123/diagnostics/fsr-denoise-check.html
node diagnostics/check-fsr-render.cjs 'http://127.0.0.1:8123/diagnostics/asr-video-quality.html?all=1&denoise=strong&degraded=1'
node diagnostics/check-spatial-popup.cjs
node diagnostics/compare-spatial-denoise.cjs
npm test
```

Os helpers usam Playwright/Chromium locais e aceitam os caminhos nas variáveis
`ANIMESCALER_PLAYWRIGHT` e `ANIMESCALER_CHROMIUM`.
Para carregar os novos controles, recarregue a extensão e a página do vídeo.
