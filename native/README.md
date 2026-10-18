# Backends nativos

Os backends nativos rodam serviços locais para a extensão. Inicie apenas um por vez; ROCm e TensorRT usam a mesma porta local.

## Inicialização rápida

- **AMD:** execute `Iniciar-AMD.cmd`. Escolha ROCm, OpenCL ou FSR3 no menu.
- **NVIDIA:** execute `Iniciar-NVIDIA.cmd`. Escolha TensorRT ou OpenCL.

Mantenha o terminal aberto. O launcher verifica dependências e informa o que falta. Recarregue a extensão e a página do vídeo após iniciar o serviço. Não execute ROCm e TensorRT ao mesmo tempo.

## Requisitos por backend

- **ROCm:** Windows com GPU AMD compatível e PyTorch ROCm instalado separadamente. Defina `ANIMESCALER_ROCM_PYTHON` para apontar ao Python desse ambiente. Compatibilidade depende da GPU, driver, versão ROCm e build PyTorch. A rota foi exercitada em hardware AMD específico; não há garantia para toda GPU.
- **TensorRT:** GPU NVIDIA, CUDA compatível e TensorRT 10.x em ambiente Python separado. A rota está implementada, mas não foi validada em GPU NVIDIA neste projeto.
- **OpenCL:** driver OpenCL instalado. O launcher prepara o ambiente Python e compila o worker Anime4KCPP quando necessário. Primeiro uso precisa de internet e ferramentas de compilação disponíveis.
- **FSR3 / DirectX 12:** GPU/driver compatível e FidelityFX SDK configurado localmente por `ANIMESCALER_FSR_SDK`. Para o launcher FSR3, configure também `ANIMESCALER_FSR3_PYTHON`. Rota experimental; não há SDK ou DLL incluído.
- **Anime4KCPP OpenCL isolado:** `Iniciar-OpenCL.cmd` cria `native/.venv-opencl` e instala os pins de `native/requirements-native-host.lock` quando não há um Python definido em `ANIMESCALER_OPENCL_PYTHON`. Não importa PyTorch, ROCm ou TensorRT. O ambiente OpenCL usa apenas suas dependências separadas.

O preflight checa arquivos, versões e configuração antes de iniciar. Preflight aprovado não comprova inferência ou qualidade na GPU.

## Checkpoints e shaders

Este repositório não contém pesos, checkpoints ou shaders Anime4K. ROCm e TensorRT precisam dos arquivos externos em `models/`; consulte [catálogo e links oficiais](../README.md#downloads-oficiais). A aplicação registra modelo como indisponível quando faltam arquivos.

Não há download automático verificado para checkpoints HFA2K Real-CUGAN, RealESR, SPAN, Lite ou Zenkai. Confirme procedência, formato, compatibilidade e licença antes de usar. Os checkpoints Lite/Zenkai são experimentais e não fazem parte de release aprovado.

## Diagnósticos locais

Logs do serviço ficam no terminal e diagnósticos ficam na máquina do usuário. Revise arquivos exportados antes de compartilhar: podem registrar versões de runtime, GPU e caminhos locais. Nenhum vídeo ou quadro é enviado para fora por este serviço.
