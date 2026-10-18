# AnimeScaler

Extensão Manifest V3 para upscale de vídeo em Chrome e Edge. Usa WebGPU no navegador ou serviços locais para ROCm, TensorRT, OpenCL e DirectX 12.

**Nenhum peso, checkpoint ou shader de modelo é distribuído neste repositório.** Baixe ativos somente das fontes oficiais abaixo e confira licenças antes de usar. O catálogo mostra rotas declaradas pelo código; disponibilidade depende dos arquivos, runtime, GPU e qualificação indicados.

## Instalação

1. Instale Node.js 20+, Python 3.10+ e Git.
2. Clone o projeto e execute `npm install`.
3. Baixe o modelo AnimeJaNai pela fonte oficial abaixo e extraia `2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx` para `models/`.
4. Execute `npm run setup:runtime` e `npm run setup:models`. O setup baixa Adore, valida SHA-256 e prepara as variantes WebGPU. Os pesos ficam locais e são ignorados pelo Git.
5. Abra `chrome://extensions` ou `edge://extensions`, ative o modo de desenvolvedor e carregue esta pasta sem compactação.
6. Para backend nativo, execute `Iniciar-AMD.cmd` ou `Iniciar-NVIDIA.cmd` e siga as instruções do terminal.

Use `npm test` para testes JavaScript e `npm run test:native:cpu` para testes Python sem GPU. Consulte [native/README.md](native/README.md) para runtimes e limites de cada backend.

## Modelos e compatibilidade

| Modelo/família | Backend declarado | Estado e arquivos |
|---|---|---|
| AnimeJaNai SuperUltraCompact 2× | WebGPU, ROCm, TensorRT | WebGPU neural está bloqueado até qualificação. Baixe o ONNX original; `setup:models` prepara variante WebGPU. TensorRT não foi validado em GPU NVIDIA. |
| Adore / Real-CUGAN 2× | ROCm, TensorRT; WebGPU listado, mas bloqueado | `setup:models` baixa as variantes oficiais FP16/FP32 e prepara arquivos WebGPU. Licença CC BY-NC-SA 4.0; uso comercial não permitido sem autorização compatível. TensorRT não foi validado em NVIDIA. |
| FSR 1 Spatial e ASR Anime Spatial | WebGPU | Processamento espacial sem pesos externos. ASR é experimental. |
| ASR Anime Temporal | WebGPU | Experimental e bloqueado até qualificação. |
| Adore Lite, Zenkai 1.0, Zenkai Real Clean 5k, Zenkai AnimeVideo 30k | ROCm; alguns têm rota TensorRT | Checkpoints experimentais ou locais. Não estão incluídos nem há pacote público verificado. Zenkai 1.0 não está aprovado para release. |
| HFA2K Real-CUGAN, RealESR AnimeVideo v3, SPAN MS-SSIM | ROCm, TensorRT | Checkpoints externos necessários. Licença/procedência dos pesos locais não verificada; TensorRT não validado em NVIDIA. |
| Anime4K S/M e RealESR + Anime4K | ROCm | Port experimental. Requer GLSL Anime4K e, no modelo combinado, checkpoint RealESR baixados à parte. |
| Anime4KCPP ACNet, ARNet e ArtCNN | OpenCL | Requer worker/runtime Anime4KCPP e arquivos do projeto upstream. |
| AMD FSR 3.1 | DirectX 12 | Experimental; requer SDK/runtime local. Qualidade e desempenho ainda não qualificados para uso geral. |

Rotas compatíveis não significam validação em toda GPU. Consulte os avisos no popup e [native/README.md](native/README.md). O aplicativo não contorna DRM.

## Downloads oficiais

- **AnimeJaNai SuperUltraCompact:** [release oficial do autor](https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip). Extraia o arquivo ONNX indicado acima. SHA-256 esperado: `f6bbfbed502b8d300ee9d678d2c277d4b56235c36d053c3a1a41be060c54c0e6`.
- **Adore FP16/FP32:** [release oficial Re-SISR](https://github.com/renarchi/Re-SISR/releases/tag/Adore). O script `npm run setup:models` baixa e verifica os arquivos automaticamente.
- **Anime4K GLSL:** [repositório oficial Anime4K](https://github.com/bloc97/Anime4K). Baixe apenas os shaders necessários; coloque-os nos caminhos indicados em `models/anime4k/*.json`.
- **Anime4KCPP:** [repositório oficial](https://github.com/TianZerL/Anime4KCPP). Consulte releases e instruções do upstream para worker e modelos OpenCL.
- **Real-CUGAN:** [repositório oficial](https://github.com/bilibili/ailab/tree/main/Real-CUGAN). Link aponta ao projeto/código; checkpoint compatível deve ser obtido e licenciado por conta do usuário.
- **Real-ESRGAN:** [repositório oficial](https://github.com/xinntao/Real-ESRGAN). Link aponta ao projeto; confira licença e arquitetura do checkpoint antes de uso.
- **SPAN:** [repositório oficial](https://github.com/hongyuanyu/SPAN). Link aponta ao projeto; confira disponibilidade e licença dos pesos no upstream.
- **FSR 3:** [AMD FidelityFX SDK](https://github.com/GPUOpen-Effects/FidelityFX-SDK). Requer configuração local; não há SDK ou modelo incluído.

Não encontramos links oficiais diretos e verificados para todos os checkpoints Real-CUGAN, RealESR e SPAN. O projeto não redistribui esses arquivos.

## Segurança e privacidade

Inferência e diagnósticos ficam locais. O modo nativo comunica-se com `127.0.0.1`. Não são enviados quadros ou URLs de vídeo. Dados de diagnóstico exportados pelo usuário podem conter detalhes do ambiente; revise antes de compartilhar.

## Licenças

O código deste projeto e dependências de terceiros seguem suas licenças próprias. Veja [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Modelos externos seguem os termos de seus autores; o usuário é responsável por baixar e respeitar esses termos.
