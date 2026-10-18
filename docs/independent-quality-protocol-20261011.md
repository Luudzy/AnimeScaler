# Protocolo adicional de qualidade

Esta rodada usa fontes de outras produções, adquiridas depois dos checkpoints existentes. Não há treino, seleção de checkpoint, acesso ao final-hidden ou aprovação automática de modelo.

| Fonte | Origem/licença | SHA-256 do download |
|---|---|---|
| Hero, Blender Foundation, 2018 | [Arquivo e atribuição CC BY 4.0](https://commons.wikimedia.org/wiki/File:HERO_-_Blender_Open_Movie-full_movie.webm) | `986d84723580760cb1dfa73fbca602a0715c491aee9b6f203523514dcf2c9ce3` |
| Namakura Gatana, Jun'ichi Kouchi, 1917 | [Arquivo restaurado, domínio público indicado na página](https://commons.wikimedia.org/wiki/File:Kouichi_Jun%27ichi_-_Namakura_Gatana_(1917)_-_4-minute_restored_version.webm) | `8791c43485d6089da9417e7849a363a7c6ded185f6e20744a784be78a1d5af0c` |

Downloads originais usados: [Hero WebM](https://upload.wikimedia.org/wikipedia/commons/a/a9/HERO_-_Blender_Open_Movie-full_movie.webm), [Namakura Gatana WebM](https://upload.wikimedia.org/wikipedia/commons/b/bf/Kouichi_Jun%27ichi_-_Namakura_Gatana_%281917%29_-_4-minute_restored_version.webm). Salve em `diagnostics/stability-media/hero-original.webm` e `namakura-original.webm`; verifique os hashes antes de preparar. As mídias e imagens derivadas permanecem locais, fora dos commits de código.

```powershell
python native/prepare_stability_media.py --source hero
python native/prepare_stability_media.py --source namakura

$taskQualityArgs = @('-u', 'native/evaluate_stability_quality.py', '--output', 'diagnostics/stability-20261011/quality-new')
Get-ChildItem diagnostics/video-sequences/stability-*-20261011/*/complete.json | ForEach-Object {
    $taskQualityArgs += @('--manifest', $_.FullName)
}
python @taskQualityArgs
python scripts/build-quality-review.py --evaluation diagnostics/stability-20261011/quality-new --output diagnostics/stability-20261011/review-new --region center

# Com servidor diagnóstico ativo; execute separado das medições de desempenho.
node diagnostics/run-spatial-quality.cjs
```

Os preparadores recusam diretórios/arquivos existentes. Para uma repetição integral, preserve a rodada anterior em outro local e use um checkout de avaliação separado. O avaliador aceita `--output` novo e manifests repetidos. Os runners também recusam sobrescrever seus relatórios. `ANIMESCALER_SPATIAL_QUALITY_OUTPUT` muda o destino espacial; `ANIMESCALER_REPORT_DIR` muda o destino dos três checks de regressão GPU existentes.

Cada trecho tem 48 frames com timestamps, hashes, HR/LR alinhados e verificação de GOP interframe 24. Hero usa 25/80/150 segundos, referência 1280×536 e LR 640×268. Namakura usa 60/180 segundos, referência 960×752 e LR 480×376. H264, H265 e AV1 usam CRF24. Três codecs sobre o mesmo trecho não são três fontes independentes. Hero 150s contém créditos: analise texto separadamente das cenas animadas de 25/80s.

A referência é o download comprimido decodificado e redimensionado com Lanczos, preservado em FFV1, não um master de estúdio. Hero é animação 2D de outro estilo; Namakura é anime histórico monocromático. Esses exemplos ampliam a avaliação, mas não comprovam generalização para muitos animes modernos.

Os modelos nativos são avaliados em CPU FP32, usando os artefatos fixados por hash e métricas sobre tensores completos. As saídas espaciais usam WebGPU real e readback RGB8, com teste de saída preta. Não interprete pequenas diferenças entre precisões como uma comparação bit a bit. MAE e deltas temporais compensados pela referência são medidas técnicas; os deltas não fazem alinhamento por fluxo óptico nem certificam ausência de ghosting. O relatório espacial também fornece PSNR; a rodada não mede SSIM.

Os painéis de revisão mantêm recortes sem interpolação, identificação dos modelos, FPS e vídeo FFV1 sem perda. Centro, esquerda e direita são escolhas fixas; não recorte somente a região que favorece um modelo. Imagens estáticas ajudam a inspecionar linhas, halos e textura. A preparação e inspeção pelo agente não substituem revisão humana ou cega de movimento.
