# Progresso — estabilização de 11/10/2026

Plano: [stability-validation](superpowers/plans/2026-10-11-stability-validation.md). Resultado completo: [relatório](stability-report-20261011.md).

O usuário autorizou os itens 1–4 e a continuação autônoma. Depois informou novo crash e pediu continuidade sem repetir a instabilidade. Desde esse aviso, o trabalho ficou restrito a revisão, documentação e testes CPU com GPU oculta e um thread.

1. **Reprodutibilidade concluída:** snapshot do código anterior, fontes/licenças de terceiros, runners portáteis, HTTP Range, bloqueio de credenciais e skip opcional NCNN explícito. Checkout original preservado; branch `codex/stability-20261011`, base `021d0c7`.
2. **Playback baseline concluído:** quatro modelos × 300 segundos, 20 minutos no total, engines reais sem erros nesses relatórios. Extensão real passou oito checks. Captura completa pelo botão da extensão continua pendente de validação manual; headless não recebeu activeTab.
3. **Experimento de desempenho rejeitado:** ganho isolado de aproximadamente 9,5% do serviço não se sustentou no playback. Candidato caiu para 4,51 FPS e dois timeouts em 43,64 segundos; usuário relatou crash. HIP graph do AnimeVideo desativado em `95a55c8`, inclusive antigos opt-ins. Nenhuma repetição de GPU após o aviso.
4. **Avaliação técnica de qualidade concluída:** Hero e Namakura, 15 sequências, 720 pares H264/H265/AV1; comparação RGB8 comum e painéis preparados. V3 vence MAE em 12/12 casos animados; 30k ganha somente nos três créditos e perde delta temporal em 13/15. Revisão humana/cega de movimento não executada; sem promoção ou treino.
5. **Continuação implementada:** contribuição efetiva da IA e idade do histórico no diagnóstico/HUD; registro independente do AnimeVideo; prontidão autenticada do viewer quando Chrome omite tabs.url; testes de regressão e revisão do código final.

Verificações: Node 193/193; Python após rollback 217 executados, 216 passaram, um skip NCNN. Artefatos selecionados e hashes estão no [índice de evidências](stability-evidence-20261011/index.json). Outros arquivos gerados, mídia, pesos e credenciais permanecem locais e fora dos commits de código.

Não há certificação de estabilidade geral do PC, sites protegidos ou qualidade perceptual em movimento. A causa de driver/Windows do crash não foi estabelecida. A versão final não ativa a otimização rejeitada.
