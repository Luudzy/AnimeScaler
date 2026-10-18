# Entrega da IA no vídeo fluido — 2026-10-10

## Problema reproduzido

O worker aceitava e enviava a imagem da IA à GPU, mas somente uma mensagem
`present` desenhava o compositor. Sem novo callback de captura (fonte parada ou
pausada), a imagem continuava na textura, invisível. Comparar contornava isso
com `presentAI`, desenhando imediatamente. Além disso, resultados com mais de
300 ms eram rejeitados mesmo quando a amostra da fonte permanecia idêntica.

## Correção

- Estabelecer o quadro original no início da inferência quando não existe
  apresentação, inclusive depois de sair de Comparar ou trocar resolução.
- Após o resultado, solicitar apresentação se o relógio estiver ocioso há mais
  de 80 ms. Uma verificação única após 90 ms cobre a captura que para pouco
  antes da conclusão. Não criar outro loop de composição em reprodução ativa.
- Cancelar a verificação ao entrar em Comparar ou destruir a engine.
- Permitir resultado lento somente se as amostras RGBA de 80×45 forem
  exatamente iguais, com mesma época e geometria. Preservar `capturedAt` e a
  idade real; movimentos e cortes continuam sujeitos ao limite de 300 ms.
- Devolver as métricas da apresentação para identificar contribuição real da IA.

## Evidência

`node diagnostics/check-native-fluid-delivery.cjs --baseline` reproduziu o
original sem IA, Comparar com IA e retorno sem IA nos quatro cenários.
`node diagnostics/check-native-fluid-delivery.cjs` mostrou contribuição da IA
igual a 1 e o pixel de saída esperado nos quatro cenários corrigidos:
40/350 ms, tanto no primeiro quadro quanto após um relógio de apresentação
inicialmente ativo parar. Testado também o retorno de Comparar.

Artefatos: `diagnostics/native-fluid-delivery-baseline-20261010.json` e
`diagnostics/native-fluid-delivery-fixed-20261010.json`.

O backend de inferência é simulado; engine, worker, VideoFrame, transporte RGB
e compositor WebGPU são reais. Isso valida a entrega, não a qualidade do modelo,
o desempenho HIP ou um site real. A comparação de amostras é uma verificação
reduzida, não prova de igualdade de todos os pixels da fonte em resolução plena.

Regressões: testes unitários de pausa, relógio ativo, cancelamento em Comparar,
destruição, expiração em movimento e preservação de idade. Verificação visual
sintética adicional: 21 checks WebGPU e 18 Canvas2D, incluindo cortes, movimento,
rejeição de histórico e ausência de pulsação em cenas estáveis.

## Aplicação

Recarregar AnimeScaler em `chrome://extensions` e fechar/reabrir o viewer para
substituir o worker anterior. Não é necessário reiniciar o serviço ROCm.

Uma inferência superior a 300 ms numa fonte em movimento ainda pode voltar ao
original por segurança temporal. Esta correção não aumenta a velocidade do
modelo nem elimina essa limitação.
