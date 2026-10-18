# Quadro direto na GPU — 2026-10-10

## Implementação

`GpuPrePost` importa HTMLVideoElement/VideoFrame com `importExternalTexture` e
converte diretamente para o buffer NCHW já usado pelo ONNX Runtime Web.
A textura RGBA intermediária em resolução original passa a ser alocada somente
quando o fallback é necessário. Não mudar modelo, resolução, precisão ou pesos.

O compositor WebGPU do vídeo fluido nativo também importa o VideoFrame e mistura
o original com a IA usando o mesmo recorte/máscara. Evita sua cópia explícita da
fonte inteira. Comparar continua usando a textura da IA por um pipeline separado.

Cada importação recebe um bind group novo, usado e submetido no mesmo task.
As referências externas são removidas após submit, antes de fechar o VideoFrame.
O recorte normalizado limita a amostragem aos centros dos pixels das bordas,
preservando o comportamento da textura recortada anterior. Canvas/ImageBitmap ou
importação incompatível continuam pelo caminho de cópia. SecurityError propaga.

Os diagnósticos registram `captureMethod: external-texture` na IA WebGPU e
`sourceMethod: external-texture` no compositor nativo.

## Verificação

- `node --test`: 182 testes, todos passando.
- `node diagnostics/check-direct-video.cjs`: AMD `rdna-1`, Chromium 1223,
  sem erros WebGPU. Nenhuma flag nova foi adicionada ao produto. O runner
  headless reutiliza `--enable-unsafe-webgpu` apenas na sessão de diagnóstico.
- 144 frames reais: três sequências de 48 quadros, fonte 1280×720 e entrada
  640×360. Máximo e média de erro absoluto dos tensores: **0** nos três trechos.
- Vídeo RGBA, NV12 com crominância variável, HTMLVideoElement via captureStream,
  recorte não central e ampliação do recorte: erro máximo/médio **0** nos casos
  testados. A paridade é do pré-processamento, não uma nova avaliação do modelo.
- Compositor nativo: nenhuma cópia explícita da fonte e cor de saída esperada.
- Regressão fluida: 21 checks WebGPU + 18 Canvas2D passando, além dos quatro
  cenários de entrega do worker com inferência simulada, inclusive retorno de
  Comparar. O serviço HIP do usuário não foi carregado nem reiniciado para isso.

## Medição limitada

Upload + pré-processamento GPU concluído, fonte sintética 1920×1080 → tensor
640×360; 15 warmups e 50 frames distintos medidos por caminho, alternando a ordem.
Leitura do tensor para paridade fica fora dessa medição. Não há inferência.

| Métrica | Cópia anterior | Importação direta |
| --- | ---: | ---: |
| Média | 5,38 ms | 5,19 ms |
| Mediana | 4,20 ms | 4,10 ms |
| P95 | 12,60 ms | 10,60 ms |
| Cópias explícitas da fonte | 50 | 0 |

O ganho de latência é pequeno e sujeito a variação; não é promessa de FPS.
No caminho direto puro, não alocar a antiga textura 1080p economiza 8.294.400
bytes de textura explícita (7,91 MiB). Não é uma medição de VRAM total: a
implementação interna do navegador pode efetuar conversões ou cópias próprias.

## Limitações e ativação

No backend ROCm, inferência ainda recebe pixels enviados pelo navegador ao
serviço nativo. A leitura CPU, transporte WebSocket e transferência para HIP
**não** foram eliminados. A melhoria aplicada ali é no compositor do modo fluido.
FSR/ASR têm pipelines próprios e não foram alterados nesta mudança.

Recarregar a extensão e fechar/reabrir o viewer. Não requer novo treinamento,
troca de modelo ou reinício do serviço ROCm.

Artefatos: `diagnostics/check-direct-video.cjs` e
`diagnostics/direct-video-20261010.json`.

Referências: [vídeo externo no WebGPU](https://webgpufundamentals.org/webgpu/lessons/webgpu-textures-external-video.html),
[tensores GPU no ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
