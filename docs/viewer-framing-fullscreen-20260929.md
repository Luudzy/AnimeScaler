# Enquadramento, comparação e tela cheia

O viewer passa a consultar toda a cadeia de iframes até o vídeo. As coordenadas
do conteúdo são compostas até a aba capturada, descontando bordas, padding e
faixas de `object-fit: contain`. Capturas parcialmente fora da área visível
geram orientação para reabrir em modo exclusivo, em vez de alterar silenciosamente
a proporção. Sessões antigas precisam ser fechadas e reabertas.

O botão **Comparar** (C) suspende o processamento ao vivo, aguarda inferência e
apresentação pendentes, cria um único ImageBitmap e usa essa imagem tanto no
original ampliado pelo navegador quanto na inferência. A comparação usa as mesmas
dimensões de saída, sem composição temporal. A origem continua reproduzindo.
**Voltar ao vídeo** retoma a captura e o modo híbrido configurado.

Tela cheia usa a janela própria do viewer nos dois modos. F ou duplo clique
alternam tela cheia; Esc sai da comparação primeiro, depois da tela cheia.
O estado normal/maximizado anterior é restaurado; janelas normais também
recuperam posição e tamanho. I alterna informações. Controles e cursor somem
após 2,5 segundos de inatividade, preservando acesso por teclado, mensagens de
erro e comparação. A barra quebra linhas em janelas menores.

A expansão do player neutraliza também translate/rotate/scale independentes,
will-change e animações/transições que poderiam reposicionar o vídeo. Ao sair,
a restauração compara valores CSS normalizados pelo navegador, preservando
alterações posteriores feitas pelo site.

## Validação

- Suite JavaScript: 136 testes aprovados.
- `diagnostics/viewer-framing.html`: proporção 16:9, exclusão de faixas, quadro
  congelado idêntico pixel a pixel, captura pausada e retomada: todos aprovados.
  A inferência desse diagnóstico é um substituto determinístico, não uma medida
  de qualidade do modelo ou benchmark do backend.
- `diagnostics/exclusive.html`: iframe com recorte, transformações independentes
  e vídeo com cover; expansão integral e restauração de estilos aprovadas no navegador.
- Estado/bounds de chrome.windows validados com testes de contrato; não foi
  executado um teste completo da extensão no site das capturas do usuário.

Para usar: recarregar a extensão, atualizar a aba do anime e abrir um viewer novo.
