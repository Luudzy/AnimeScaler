# Viabilidade FSR 3 no WebGPU

**Decisão: NO-GO para portar ou anunciar FSR 3 como rota WebGPU neste projeto.**
O caminho atual de browser não fornece os recursos exigidos pela integração
temporal oficial, e a implementação oficial disponível depende de D3D12 nativo.
Preencher essas entradas com estimativas ou shaders novos mudaria o algoritmo.

## Evidência

- O provider FSR 3 usado pelo projeto recebe recursos e command list D3D12 no
  adapter nativo. WebGPU da extensão não expõe esses objetos nem oferece
  interoperabilidade para entregá-los ao provider. Veja `native/fsr3-probe/`,
  [FSR3 nativo](fsr3-native-video.md) e [contrato oficial](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/).
- FSR 3 temporal usa depth, motion vectors e jitter de câmera. Um vídeo HTML
  decodificado fornece quadros de cor, sem depth do jogo, jitter ou motion
  vectors do renderer. A implementação nativa atual estima movimento e usa
  depth constante; isso é adaptação de vídeo, não paridade da entrada original.
- O código WebGPU presente em `src/engine/fsr-shaders.js` implementa FSR 1
  espacial EASU/RCAS e tem licença MIT registrada em
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Essa licença não cobre
  código FSR 3. O SDK/provider FSR 3 permanece externo; a licença local e os
  termos de redistribuição do binário precisam ser verificados antes de copiar
  ou distribuir qualquer código. Fontes: [interpolação FSR 3](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-interpolation/)
  e [SDK oficial](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK).
- `docs/fsr3-native-experiment.md` registra que a rota temporal experimental
  não venceu bilinear de forma consistente nos três trechos fixos. Isso não
  prova que FSR 3 falha em sua integração própria; não sustenta uma promessa de
  qualidade para uma rota WebGPU diferente.

## Recursos e lacunas

| Recurso | Disponibilidade no browser | Efeito |
|---|---|---|
| Quadro de cor decodificado | Sim | Entrada disponível |
| Depth do jogo | Não | FSR 3 temporal oficial não pode receber depth real |
| Motion vectors do renderer | Não | Vetores estimados seriam outra entrada/algoritmo |
| Jitter subpixel original | Não | Não existe sequência de câmera para reconstruir |
| Command list/provider D3D12 oficial | Não via WebGPU | Provider atual não pode executar nessa rota |
| Shader FSR 3 WebGPU licenciado e auditado | Não encontrado no projeto | Não copiar fontes nem inferir licença de FSR 1 |

Histórico, corte, seek, crop e timestamps exigiriam também estados/reset
temporais confiáveis. A rota nativa já tem reset por crop e timestamp; isso não
resolve a falta dos recursos acima nem autoriza reutilizar o provider por WebGPU.

## Consequência

- Manter FSR 3 apenas na rota nativa D3D12, com estado experimental existente.
- Não criar alias, shader substituto ou modo chamado “FSR 3 WebGPU”.
- Reabrir a decisão somente com API/licença WebGPU oficial adequada e fontes de
  depth, motion e jitter válidas, ou com um pedido separado para avaliar um
  algoritmo de vídeo diferente, nomeado sem alegar paridade FSR 3.
- Não houve compilação, teste browser, benchmark ou carga GPU nesta auditoria.
