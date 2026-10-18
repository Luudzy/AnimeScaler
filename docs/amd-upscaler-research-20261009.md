# Pesquisa GitHub — AMD/RDNA1 para upscale e geração de imagem

Pesquisa realizada em 2026-10-09, com foco em RX 5500 XT (`gfx1012`), Windows e vídeo anime em tempo real.

## O que tem potencial real

| Projeto | O que otimiza | Aplicação no AnimeScaler | Prioridade |
|---|---|---|---|
| [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | NCNN C++ com shaders Vulkan, FP16 e packing | Mesmo AnimeVideo v3 em outro runtime | Alta, mas precisa de worker C++ persistente |
| [Real-CUGAN ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan) | U-Net anime, 2x/3x/4x, denoise configurável | Candidato para 360→720 e restauração de ruído | Alta |
| [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) | NCNN Vulkan, denoise e upscale anime | Preset rápido/limpeza de ruído | Média-alta |
| [SPAN-ncnn-vulkan](https://github.com/TNTwise/SPAN-ncnn-vulkan) | SPAN, arquitetura eficiente de super-resolution | Candidato a Lite 2x/4x | Média-alta |
| [Anime4KCPP](https://github.com/TianZerL/Anime4KCPP) | Filtros/shaders rápidos, sem inferência pesada | Pré/pós-processamento, linha e ruído | Média |
| [AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) | Compilação de grafo ONNX para MIOpen/rocBLAS/HIP | Pode eliminar overhead do PyTorch | Alta, porém experimental em Windows/RDNA1 |
| [TheRock](https://github.com/ROCm/TheRock) | Builds atuais do ROCm por arquitetura | MIOpen/CK/gfx1012 mais completos | Alta, mas exige ambiente separado |

## Achados importantes

O NCNN declara AMD no Windows como plataforma conhecida para Vulkan e usa `use_vulkan_compute`. O Real-ESRGAN oficial distribui modelos AnimeVideo v3 em NCNN (`x2`, `x3`, `x4`). Isso é uma implementação C++ persistente, diferente do binding Python que testamos. [NCNN Vulkan FAQ](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0)

No nosso PC, o modelo AnimeVideo v3 NCNN foi carregado e produziu a forma correta, mas o binding Python ficou em **4603,84 ms por frame**. O executável oficial em lote também inclui inicialização e PNG. Portanto, o ganho potencial está em escrever um processo C++ persistente que mantenha `ncnn::Net`, allocators Vulkan e buffers vivos, recebendo RGB bruto por pipe. Executar o `.exe` por frame não serve.

Real-CUGAN é especialmente relevante para anime: a documentação informa treino em escala de milhões de imagens, 2x/3x/4x, vários pesos de denoise e versão NCNN para placas AMD. Há também modo de baixa memória e um modo de baixa memória mais rápido. [Real-CUGAN](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md)

SPAN é uma alternativa de arquitetura eficiente, com distribuição ncnn Vulkan para Windows/AMD/NVIDIA/Intel. Não é específico para anime, mas é um candidato melhor que RealESR para uma variante Lite se a qualidade nos frames anime for aceitável. [SPAN-ncnn-vulkan](https://github.com/TNTwise/SPAN-ncnn-vulkan)

Waifu2x ncnn Vulkan é antigo, mas continua interessante para limpeza de ruído e linhas. Ele suporta AMD/Vulkan, escala 2x/4x e níveis de denoise; tende a suavizar mais que RealESR, então deve ser avaliado como preset de restauração, não substituto automático. [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan)

## ROCm/MIOpen e `gfx1012`

O repositório comunitário [ROCm-RDNA1](https://github.com/TheTrustedComputer/ROCm-RDNA1) confirma que a RX 5500 XT deve usar `gfx1012`, sem `HSA_OVERRIDE_GFX_VERSION`. Ele também registra limitações de hipBLASLt, Triton e Composable Kernel em RDNA1. O projeto foi arquivado, então é referência histórica, não uma base nova para substituir o ambiente atual.

O [TheRock](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) atual lista `gfx1012` como build passing, sanity tested e release ready em sua matriz. Isso é o caminho de maior potencial para testar uma pilha MIOpen mais completa, mas não prova que o wheel Windows atual funcione sem regressões na 5500 XT.

O nosso serviço imprime `CK grouped conv library not found for device gfx1012`. Portanto, antes de alterar o modelo, há uma hipótese concreta: a instalação não tem todos os kernels MIOpen/CK para `gfx1012`. O teste correto seria uma instalação TheRock separada, medir as mesmas convoluções e comparar saída/latência. Não devemos substituir o ambiente funcionando sem esse A/B.

O [MIGraphX](https://github.com/ROCm/AMDMIGraphX) compila grafos ONNX e emite chamadas MIOpen, rocBLAS ou kernels HIP; a documentação oficial descreve o fluxo principalmente para Linux. O plugin [ONNX Runtime AMD EP](https://github.com/onnxruntime/onnxruntime-ep-amdgpu) exige ORT 1.24+ e é promissor para uma exportação estática, mas não há evidência suficiente de suporte prático para Windows + RDNA1 neste projeto.

## Projetos que não resolvem este alvo

- Os exemplos [AMD Ryzen AI Super-Resolution](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples/super-resolution) usam NPU XDNA/Vitis AI e não se aplicam à RX 5500 XT.
- Projetos de Stable Diffusion via DirectML ou MIGraphX ajudam geração de imagem, mas não reduzem diretamente a latência do RealESR de vídeo.
- `torch.compile` já foi testado no head 2x e ficou em 33,67 ms contra 33,67 ms eager, com paridade exata.
- Vulkan via binding Python já foi medido e ficou em 4603,84 ms; não deve ser usado como conclusão sobre o NCNN C++ nativo.

## Ordem experimental recomendada

1. Baixar e medir Real-CUGAN ncnn Vulkan `pro 2x`, `se 2x` e `nose 2x` nos mesmos frames, com pipe RGB persistente.
2. Medir SPAN ncnn Vulkan como candidato Lite.
3. Criar um worker C++ persistente para Real-ESRGAN NCNN v3; comparar contra os 32,85 ms do ROCm em 640×360→1280×720.
4. Fazer A/B da pilha TheRock/MIOpen com `gfx1012`, sem tocar no ambiente atual.
5. Só depois testar MIGraphX/ONNX no modelo exportado.

O objetivo de 20 ms é plausível para um modelo menor ou para um worker Vulkan C++ bem otimizado em 360→720. Não há evidência de que RealESR v3 completo consiga 20 ms em 720→1080 ou 2K/4K na RX 5500 XT sem reduzir a rede ou a resolução de entrada.

## Revisão específica do Anime4KCPP

O Anime4KCPP v3 é diferente dos shaders GLSL Anime4K que já estão registrados no projeto. Ele implementa um CNN em C++17 e oferece backend OpenCL nativo, com build Windows documentado usando `AC_CORE_WITH_OPENCL=ON`, além de opção de módulo de vídeo e bindings Python. Nesta máquina, o OpenCL AMD está disponível e identifica corretamente a Radeon RX 5500 XT, portanto existe um caminho técnico real para testar a implementação nativa.

O resultado de `anime4k-s` no renderer Python/ROCm, de aproximadamente 871 ms em 1080p→4K, não reprova o Anime4KCPP. É uma rota de execução diferente e não usa os kernels C++ OpenCL do projeto. O experimento correto é compilar o core com OpenCL e manter um worker persistente, medindo entrada de frame decodificado até saída, sem iniciar um processo por frame.

O Anime4KCPP deve ser tratado como candidato de baixa latência para 2× e melhoria de linhas/ruído, não como substituto automático do RealESR em qualidade. Ele só poderá entrar no preset 24 FPS se o benchmark OpenCL nativo ficar abaixo de 41,7 ms em 1080p→4K. O core/CLI sem o módulo de vídeo é MIT; o módulo de vídeo é GPLv3, então a integração precisa manter essa separação de licenças.

O build nativo foi realizado localmente com MSVC Release, OpenCL, CLI e benchmark, sem o módulo de vídeo. A RX 5500 XT foi detectada como `gfx1012:xnack-`. Em 1920×1080, batch 1 e uma thread, os resultados de processamento do modelo foram:

| Modelo | FPS OpenCL | ms/frame | Uso recomendado |
|---|---:|---:|---|
| `acnet-f8b4` | 155,18 | 6,44 | rápido/neutro |
| `acnet-f8b8` | 109,94 | 9,10 | qualidade leve |
| `acnet-f8b8-hdn` | 108,62 | 9,21 | denoise leve |
| `arnet-f8b8` | 67,03 | 14,92 | mais detalhe |
| `artcnn-c4f16` | 65,35 | 15,30 | nitidez/denoise |
| `fsrcnnx-f8b4` | 149,41 | 6,69 | rápido, restauração leve |

Isso é muito melhor que os 48,8 ms do Adore Lite medidos no renderer para 1080p→4K, mas ainda é medição de inferência nativa, não FPS completo da extensão. O próximo passo é envolver o core OpenCL em um worker persistente e medir upload, processamento, download e apresentação. O log bruto está em `diagnostics/anime4kcpp-opencl-benchmark.json`.

O worker persistente foi implementado em `third_party/Anime4KCPP-master/worker` com protocolo binário RGB/RGBA, processo único e stdin/stdout em modo binário do Windows. Com o preset 4K corrigido para alimentar diretamente metade da saída final, `acnet-f8b4` atingiu 39,89 ms em 1712×1080→3424×2160 (25,07 FPS), incluindo transferência do frame e leitura do resultado. `acnet-f8b8-hdn` ficou em 52,56 ms no mesmo tipo de teste. Em 640×360→1280×720, `acnet-f8b8-hdn` ficou em 6,67 ms (149,25 FPS) após o aquecimento. Portanto, `acnet-f8b4` é o primeiro candidato real para o preset 4K de 24 FPS; ainda precisa de validação visual e de P95 antes de virar padrão.

## FSR para streams no navegador

Os arquivos disponibilizados em `Downloads` e a cópia local do FidelityFX SDK
contêm o provedor de FSR 4 para DirectX 12. O cabeçalho da SDK confirma que o
dispatch nativo exige uma command list, recursos de cor, vetores de movimento,
profundidade, exposição e máscaras reativas. Um elemento `<video>` do Chrome
fornece somente a imagem decodificada; portanto, copiar as DLLs ou carregá-las
no serviço ROCm não criaria um FSR 4 funcional e ainda adicionaria cópias entre
APIs.

Foi integrado `AMD FSR 1 Spatial · WebGPU · streams rápidos` como alternativa
separada. Ele faz EASU (reconstrução espacial orientada por bordas) seguido de
RCAS (nitidez adaptativa) em dois render passes WebGPU. O frame sai do decoder
para `copyExternalImageToTexture`, segue por texturas GPU e chega ao canvas;
não há `getImageData`, `mapAsync`, JPEG, WebSocket ou readback nessa rota.

Validação real do navegador em 2026-10-09: ambos shaders compilaram, criaram
pipelines e renderizaram 320×180 → 1920×1080. Um teste sintético separado,
com espera explícita pela fila GPU após aquecimento, mediu 0,47 ms/frame em
1280×720 → 1920×1080 e 0,59 ms/frame em 1920×1080 → 3840×2160. Isso mede os
passes EASU/RCAS, não o FPS completo de um site. FSR 1 melhora escala e percepção de nitidez,
mas não é modelo de restauração: ele não pode recuperar detalhe perdido nem
remover artefato de codec como RealESR/Anime4K/um modelo treinado.
