# Vídeo fluido — referências e resets

10/10/2026. Rodada focada no híbrido nativo ROCm/TensorRT. Não acrescenta
interpolação, geração de frames ou modo temporal ao FSR 1. O checkbox atual
é usado pelo viewer nativo; o FSR 1 mantém processamento por quadro.

## Falhas reproduzidas e corrigidas

O cache de thumbnails guardava o objeto inteiro da captura. Após reset,
`present()` podia reutilizá-lo e mudar seu `epoch`, inclusive quando esse
mesmo objeto acompanhava uma inferência em andamento. O teste demonstrou
que um resultado anterior podia ser relabelado como pertencente à nova cena.

Agora o cache guarda apenas pixels reutilizáveis. Cada amostragem possui
seu próprio instante de captura e epoch. Reset invalida o cache; apresentação
usa uma cópia quando precisa associar a cena a um novo epoch, sem alterar
a referência de uma inferência em andamento. Quadros com timestamp repetido
evitam novo readback, mas não herdam uma data de captura expirada.

A busca de movimento usa identidade dos arrays de pixels, preservando o
cache em quadros repetidos apesar dos objetos de captura independentes.
Não modifica o orçamento de 300 ms nem permite prolongar a validade de
um resultado antigo; apenas capturas novas recebem instantes novos.

O presenter agora também invalida histórico em timestamp regressivo ou
salto superior a 250 ms, mesmo se os dois frames tiverem cores idênticas.
Isso evita depender exclusivamente de um detector de cortes por histograma.
Frames normais e timestamps iguais não disparam esse reset.

## Verificações executadas

- Novos testes reproduziram os três problemas antes da alteração e passaram
  depois: epoch antigo, data de captura em timestamp repetido e seek de
  conteúdo com as mesmas cores.
- Suite JS completa: **170 testes passaram**.
- Browser Chromium real, com VideoFrame e compositor: 21 verificações GPU
  e 18 no fallback Canvas2D passaram, sem page/console errors.
- Cobertura inclui regiões estáveis com IA, movimento usando a origem,
  acompanhamento de detalhe, ausência de rastro no caso sintético, expiração,
  cortes, geometria, cadência estável e rejeição de matches alternantes.
- Relatório: `diagnostics/fluid-browser-20261010.json`.

O teste de movimento apresentou 5,8 ms WebGPU e 5,6 ms Canvas2D nesta amostra;
inclui busca CPU de thumbnail e readback de diagnóstico, não é benchmark de
throughput nem FPS do player. Não foi executada nova inferência de modelo,
avaliação de episódio real ou medição end-to-end de site nesta rodada.
Passar esses casos não prova eliminação universal de flicker.

## Aplicar e reproduzir

Recarregue a extensão e reabra o viewer com Vídeo fluido ativado no backend
ROCm/TensorRT. Nenhum checkpoint ou preset foi alterado.

Diagnóstico local: execute `diagnostics/serve-fsr3-check.py` e
`node diagnostics/check-fluid-browser.cjs`. O servidor permite somente
assets de teste autorizados, sem credenciais. Encerre-o ao terminar.

Próximo trabalho depende do sintoma real: medir contribuições de IA,
rejeições e latência durante reprodução; para FSR 1, avaliar a cadência de
captura/apresentação separadamente do híbrido nativo.
