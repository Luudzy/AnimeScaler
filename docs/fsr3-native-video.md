# FSR 3.1 oficial no viewer — experimental

Implementado e verificado em 09/10/2026 na AMD Radeon RX 5500 XT. O modelo
selecionável é **AMD FSR 3.1 · vídeo experimental**, usando provider oficial
**3.1.5**, SDK local 2.3.0 e DirectX 12. Não é o shader FSR 1 renomeado.

## Usar

1. Execute `Iniciar-FSR3.cmd` na raiz do projeto e mantenha o serviço aberto.
2. Recarregue a extensão em `chrome://extensions` ou `edge://extensions`.
3. Escolha o modelo FSR 3.1; o backend DirectX 12 é selecionado automaticamente.
4. Escolha saída 720p, 1080p, 2K/1440p ou 4K/2160p e qualidade FSR.

Performance usa até metade da dimensão de saída; Balanced divide por 1,7;
Quality por 1,5; Ultra Quality por 1,3; Nativo preserva a resolução disponível.
A entrada nunca é ampliada artificialmente antes do FSR3 e fica limitada a
1920 pixels por dimensão e 1920×1080 pixels. Assim, uma fonte 720p para 4K
permanece 720p. Nativo só é verdadeiro 1:1 quando fonte e saída têm o mesmo
tamanho; não fornece o jitter de câmera usado em AA de jogos.

O serviço usa porta 8767 e credencial privada em `native/fsr3-connection.json`.
Não altera o serviço ROCm em 8766 nem o modelo/preset padrão. Usa Python
sistema com aiohttp/numpy, sem instalar pacotes no ComfyUI/ROCm.

## Execução e limitações

O worker mantém contexto FSR3, texturas, staging e readback entre frames.
Movimento atual→anterior e reatividade são estimados por compute shader D3D12,
com busca limitada a aproximadamente sete pixels por quadro de entrada.
O upscale oficial também executa na GPU; não há WARP ou upscale CPU.

Vídeo decodificado não oferece depth e jitter de um motor de jogo. O adapter
usa plano de profundidade constante, movimento estimado e nenhum jitter
artificial. Não equivale à integração FSR3 temporal de um jogo. Não há geração
de frames, treino de FSR3 nem garantia de redução de ruído. Regiões alteradas
são rejeitadas pelas máscaras, mas não existe detector global de corte.

O histórico reinicia no primeiro quadro, troca de tamanho, crop alterado,
timestamp que retrocede/repete, saltos maiores que 250 ms e pausa longa no
worker. O modo híbrido anterior fica desativado para evitar dois históricos.
Cada stream possui seu próprio uso exclusivo do contexto; outro viewer é
rejeitado enquanto o primeiro estiver conectado. Um quadro em voo e limites
de timeout impedem fila ilimitada. Falha de GPU encerra o worker, sem crash loop.

Captura/resize de entrada, readback, pipes e WebSocket ainda cruzam memória
CPU. O canvas final utiliza WebGPU quando disponível. Não é zero-copy.

## Funcionamento verificado

- 164 testes JS e 13 testes Python de contratos/serviço passaram.
- Popup: modelo/backend persistem, cinco qualidades selecionáveis; controles
  incompatíveis ficam ocultos/desativados e trocar de modelo libera D3D12.
- Chromium na GPU real: VideoFrame → worker → WebSocket → D3D12 FSR3 →
  OffscreenCanvas, saída não preta, sem console/page errors.
- Frames distintos e reconfiguração 720p → 1080p → 1440p → 4K passaram.
- Autenticação, origem, transporte, IDs, short writes, frame delta e posse
  exclusiva do histórico cobertos por testes. Não foi testado um player de
  site real nesta rodada; o teste do browser usa quadros sintéticos distintos.

## Qualidade em vídeo real

Mesmo conjunto de três trechos, 48 frames por trecho (144 total), em
`diagnostics/video-sequences/rezero-real-01`, 360p→720p, delta 1001/24 ms.
Comparação com Pillow bilinear e com FSR3 reset a cada quadro. MAE em RGB
normalizado; temporal MAE mede diferença temporal compensada pela referência,
sem warping por fluxo. Menor MAE é melhor. Referências são os frames preparados
do projeto, não demonstração de generalização a outros animes.

| Trecho | Bilinear MAE | FSR3 reset MAE | FSR3 temporal MAE | Bilinear temporal | FSR3 temporal |
|---|---:|---:|---:|---:|---:|
| 60 s | 0,010698 | 0,011106 | 0,011298 | 0,001135 | 0,001194 |
| 600 s | 0,012806 | 0,013222 | 0,013600 | 0,010314 | 0,010623 |
| 1200 s | 0,002357 | 0,004463 | 0,005045 | 0,000432 | 0,000584 |

**Não houve ganho consistente:** o adapter temporal piorou MAE e erro temporal
nos três trechos, especialmente no trecho de gradientes claros. PSNR temporal
também ficou abaixo do bilinear: 32,52 / 31,60 / 42,81 dB, contra
32,83 / 31,95 / 46,61 dB. Esta opção foi disponibilizada para experimentação
conforme autorização do usuário, sem promoção a padrão ou alegação de melhor
qualidade. Relatório completo: `diagnostics/fsr3-native-real-video-144.json`.
Frames 0/16/32 de cada trecho estão salvos em `diagnostics/fsr3-*.png`.

## Performance medida

No conjunto real 360p→720p, mediana nativa temporal 10,00–10,19 ms e P95
10,76–10,87 ms. Inclui pipe, upload, conclusão GPU e readback; **não é tempo
isolado de inferência nem FPS do player**. O primeiro contexto tem aquecimento
separado. Não foi medido VRAM ou tempo GPU isolado.

Smoke no browser, sem reúso, em modo Performance:

| Entrada → saída | Tempo nativo após warmup | Tempo completo do browser |
|---|---:|---:|
| 640×360 → 1280×720 | 8,16–8,88 ms | 32,6–51,0 ms |
| 960×540 → 1920×1080 | 17,65–18,36 ms | 89,3–91,7 ms |
| 1280×720 → 2560×1440 | 35,70–38,70 ms | 178,4–190,7 ms |
| 1920×1080 → 3840×2160 | 78,79–87,46 ms | 366,0–397,6 ms |

São cinco amostras quentes em 720p e apenas duas por tamanho maior; não são
benchmark de 500 frames nem FPS de reprodução. 4K **não atingiu 24 FPS**.
Transporte/readbacks continuam sendo um gargalo grande; reduzir cópias exigirá
um caminho nativo de apresentação/compartilhamento de texturas diferente.
Resultados: `diagnostics/fsr3-browser-worker.json`.

## Reproduzir

`native/fsr3-probe/build.ps1` compila os dois executáveis usando MSVC/Windows
SDK e o SDK externo. `diagnostics/evaluate-fsr3-native.py` avalia os 144 frames.
`diagnostics/serve-fsr3-check.py` serve somente assets de teste na porta 8765;
`node diagnostics/check-fsr3-browser.cjs` verifica o caminho completo usando
Chromium/Playwright instalado. Não execute esses testes GPU simultaneamente.

DLL oficial SHA256:
`d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46`.
Hash do worker utilizado consta no relatório real-video. Credenciais nunca
são incluídas nos relatórios nem servidas pelo HTTP de diagnóstico.
