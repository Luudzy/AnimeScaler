# Gates de release

Esta checklist governa uma release; passar testes CPU não autoriza publicar ou promover uma rota experimental.

## Antes de empacotar

- Registre versão, revisão Git ou manifesto de escopo, arquivos incluídos e SHA-256 dos artefatos distribuídos.
- Confirme allowlist de licença e origem dos pesos. Exclua tokens, credenciais, datasets, mídia privada, checkpoints e resultados locais não autorizados.
- Execute `npm ci`, `npm run test:cpu` e `npm run verify:runtime`. Registre pass/fail/skip e preserve saída bruta.
- Para código nativo alterado, execute a suíte CPU nativa com GPU oculta. Para mudança de rota GPU, exija prova no runtime e hardware alvo; mock não substitui essa prova.
- Para mudança de captura/apresentação, valide browser real na matriz de site, iframe, CORS, seek, pausa, fullscreen e cleanup aplicável.
- Compare qualidade e desempenho com entradas fixas. Registre gates de p95, estabilidade, memória e artefatos antes de rodar.
- Mantenha `releaseStatus: experimental` e rota opt-in até todas as provas específicas da família passarem. Step de treino ou sucesso técnico isolado não promove modelo.

## Preferências e compatibilidade

- Preserve IDs/modelos/backend explícito em preferências existentes. Opção sem suporte segue visível, indisponível e sem substituição silenciosa.
- Versione mudanças de schema; prove leitura de dados anteriores, roundtrip e downgrade suportado antes de publicar.
- Não migre, remova ou reescreva credenciais, modelos locais, datasets, checkpoints ou experiências durante instalação/rollback.
- Identifique o pacote anterior qualificado e mantenha os hashes necessários para restaurá-lo.

## Rollback

1. Interrompa distribuição e desative a rota experimental afetada.
2. Restaure o pacote anterior pelo digest registrado; não sobrescreva dados de usuário.
3. Reverta migração apenas se ela tiver rollback testado. Caso contrário, use leitura compatível com a versão anterior ou mantenha os dados e bloqueie a versão.
4. Verifique runtime, preferências, disponibilidade dos modelos e inicialização da rota restaurada.
5. Registre build, motivo, digest antes/depois e provas de recuperação. Não apague artefatos de experimento para “limpar” o rollback.

## Condição de aprovação

Uma release é aprovada somente quando gates aplicáveis têm evidência ligada ao mesmo build/manifesto, limites restantes estão explícitos e rollback é praticável. Gate não executado permanece pendente; histórico de outra árvore não conta como aprovação atual.
