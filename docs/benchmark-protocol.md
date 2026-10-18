# Protocolo de benchmark

Registre o protocolo antes de executar uma comparação. Guarde configuração, resultados brutos e identificação dos artefatos juntos. Um resultado sem esses dados é exploratório, não baseline.

## Identidade obrigatória

- Build: revisão Git ou digest do manifesto de escopo; data e sistema operacional.
- Modelo: ID, arquivo, SHA-256, escala e precisão. Registre também preset, backend, provider, layout e parâmetros adaptativos.
- Entrada: arquivo e SHA-256, intervalo de cenas/timestamps, dimensões decodificadas e geometria/crop usados.
- Ambiente: navegador e versão, GPU e driver, runtime/versão, resolução da tela e energia/desempenho do sistema quando disponível.
- Rota: captura, transporte, compositor, origem (aba, player ou fixture) e caminho de apresentação.
- Método: objetivo, unidade, fronteiras do relógio, aquecimentos, quantidade de amostras, concorrência e condição de parada.
- Critérios: gates de qualidade, p95, estabilidade e memória definidos antes da execução. Anexe logs e arquivos de evidência com digest.

## Pré-registro obrigatório do candidato

Preencha este registro e guarde-o com o baseline antes de executar o candidato. Limite sem valor numérico, unidade, fonte e método de medição torna a comparação bloqueada; não substitua campo desconhecido por zero.

| Gate | Valor/limite pré-registrado | Fonte e método | Falha/aborto |
|---|---|---|---|
| Qualidade | Métricas, tolerâncias e revisão visual aplicáveis | Baseline fixo, amostras e regra de comparação | Regressão além da tolerância rejeita candidato |
| Latência/desempenho | p95 máximo e/ou ganho mínimo end-to-end | Janela pareada, quantidade de amostras e ruído do baseline | Gate falha se limite não passar considerando a variação medida |
| Memória | Teto e unidade; indicar CPU, RAM ou VRAM | Ferramenta, intervalo de amostragem e ponto de pico | Teto excedido ou crescimento sem estabilizar aborta |
| Estabilidade | Duração, volume de trabalho e erro permitido | Perfil de reprodução e contagem de frames/operações | Crash, perda de sessão ou erro acima do limite aborta |
| Orçamento | Tempo máximo, tentativas e condição de parada | Responsável e ambiente reservados | Ao esgotar, marcar inconclusivo; não ampliar durante a execução |

Derive cada limite do requisito do perfil alvo e da distribuição observada no baseline, incluindo incerteza de medição. Registre fórmula e dados usados. Se requisito ou baseline não existe, mantenha o limite como pendente e não rode o candidato. Compare um candidato por vez; mudança de threshold exige novo pré-registro antes da próxima execução.

## Rótulos de resultado

| Rótulo | Cálculo | Pode sustentar |
|---|---|---|
| Latência por frame | Duração de uma operação com início/fim declarados; relatório p50/p95/máximo e unidade ms/frame. | Apenas o estágio medido. Inferência isolada não prova FPS do player. |
| Throughput em lote | Frames válidos concluídos ÷ duração total do lote, após aquecimento declarado. | Capacidade do lote e caminho incluído; não latência individual. |
| Inferência FPS | Inferências novas ÷ janela medida. Repita/reuse e espera ficam em contadores separados. | Taxa da engine; não necessariamente taxa de apresentação. |
| Playback FPS | Frames apresentados ÷ tempo de reprodução medido em navegador, com fonte e perdas registradas. | Somente a rota, fonte, ambiente e janela observados. |

Não trate taxa configurada como taxa alcançada. Não inclua frames repetidos como inferências novas. Não misture tempo de fila, captura, inferência, conclusão GPU, transferência e apresentação sem declarar a fronteira. Relate amostra sintética e vídeo real separadamente.

## Execução e comparação

1. Fixe entrada, modelo, ambiente, rota e gates. Mude uma variável por comparação.
2. Declare estado frio ou aquecido e exclua aquecimento da janela medida, salvo quando medir startup.
3. Registre cada amostra válida e descarte motivos; não remova outliers sem regra prévia.
4. Reporte amostras, p50, p95, máximo, FPS correspondente quando válido, erros, perdas, resolução efetiva e memória observada.
5. Pare ao alcançar orçamento, critério de falha ou condição de estabilidade previamente definida.
6. Anexe resultados brutos, protocolo preenchido e digests ao relatório. Marque dimensões não medidas como desconhecidas.

Uma comparação pareada usa a mesma entrada, janela, hardware, runtime e rota. Se qualquer um mudar, classifique como comparação diferente e não atribua a diferença somente ao modelo.
