# Matriz de validação por rota

Esta matriz separa prova automatizada, evidência histórica e validação ainda necessária. Resultado CPU/mock não qualifica driver, navegador real, qualidade visual ou desempenho de uma rota GPU.

Use os [gates de release](release-gates.md) antes de distribuir qualquer build.

| Rota/claim | Prova disponível | Estado e limite |
|---|---|---|
| JavaScript CPU | `npm test`; **515/515 aprovados** em 2026-10-16. | Inclui gate sintético para rotas nativas e integridade do pacote; não substitui site real ou avaliação visual. |
| Backend nativo CPU | `npm run test:native:cpu`; **353 aprovados, 2 ignorados** em 2026-10-16. | Testa lógica CPU e doubles; não prova inferência de todos os modelos. |
| Setup Adore WebGPU | Venv Python 3.10 pinada; conversão temporária validou SHA-256 FP32/FP16; destinos existentes preservados. | Atual para preparação local. Não executou `setup:models` completo nem downloads de pesos. |
| WebGPU neural/shader em navegador | Relatórios e resultados históricos em `diagnostics/` e README; testes CPU usam dispositivos falsos. | Evidência histórica não representa a árvore atual, o Chrome real ou uma nova medição de desempenho. Requer repetir em navegador/GPU e anexar manifesto de escopo. |
| ROCm/Adore no player | E2E de 60 s baseline, E2E Chromium de 10 s com sinal sintético e E2E Chromium de 15 s com AAC real em fixture derivada, RX 5500 XT, `adore-lite-v3`. [Relatórios e limites](playback-evidence-20261015.md). | Áudio AAC decodificado permaneceu ativo enquanto inferência ocorreu; guard escondeu overlay a **4.35 FPS** contra mínimo **19.18 FPS**. Prova cobre fixture curta em loop, sem deriva A/V de longa duração nem saída física de alto-falante. Não qualifica qualidade temporal. |
| Gate de throughput nativo | `tests/controller-throughput-gate.test.js`; Chromium E2E passou **14/14** em 2026-10-16. | Todas as rotas nativas param aprimoramento abaixo do alvo por 5 s, preservando vídeo original. Chromium testa lifecycle com backend OpenCL simulado; a medição OpenCL GPU existente é anterior, 15 s e cerca de **4,6 FPS**. TensorRT/D3D12 e recuperação de sessão continuam sem E2E GPU atual. |
| Inferência nativa ROCm/OpenCL/FSR3 | Em 2026-10-16, cada backend processou um quadro 64×64→128×128 na RX 5500 XT; ROCm e OpenCL emitiram RGBA válido, FSR3 também. Os três serviços responderam em `/health`. | Prova um quadro por backend; não mede playback, qualidade temporal ou estabilidade sustentada. |
| TensorRT | Launcher, preflight e fluxo de menu testados em stubs; preflight real confirma dependências ausentes. | Sem GPU NVIDIA, ambiente CUDA, `tensorrt` e `spandrel`; não funciona nem pode ser qualificado nesta máquina AMD. |
| Captura e apresentação | E2E da extensão com fixture H.264/AAC de 8 s em loop, captura inline e serviço ROCm; teste de STOP durante frame em voo. | Web Audio confirmou AAC decodificado durante inferência e após o guard ocultar o overlay. Long-run A/V sync, saída física de alto-falante, site remoto, DRM, iframe e matriz de browser continuam sem prova nesta rodada. |
| Treino, dataset e qualidade | `npm run test:native:cpu` (353/2 ignorados); manifests reais existentes foram verificados somente em JSON/hash. | Preflight rejeita metadata incompleta, rótulos errados ou vazamento entre `splitGroup`/path/hash. Geradores novos separam grupos de origem, escrevem caminhos relativos e preservam destinos existentes. Os três manifests reais `*-split` são bloqueados; fontes originais ausentes impedem reconstrução, avaliação e treino. Nenhum modelo promovido. |

## Registro mínimo para nova prova

Use o [protocolo de benchmark](benchmark-protocol.md). Registre a árvore pelo manifesto de escopo, classifique a prova como CPU, mock, browser, GPU ou manual e anexe resultado bruto com digests. Ao mudar build, modelo, entrada, rota ou driver, abra uma nova linha de evidência; não transfira o estado de um relatório histórico.
