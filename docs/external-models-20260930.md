> Atualização de 04/10/2026: o RealESR ROCm recebeu heads de saída compostos, processamento integral até 1080p e captura HIP limitada a entradas pequenas. As medições abaixo são históricas. Veja [otimização RealESR e benchmarks atuais](realesr-anime-optimization.md).

# Modelos externos integrados

Arquivos originais preservados. Seleções novas: `hfa2k-cugan`, `realesr-anime`, `span-mssim`. Recarregar a extensão e selecionar ROCm; o serviço local foi iniciado com esses modelos disponíveis. TensorRT tem ONNX exportado e escala correta, mas não foi executado em hardware NVIDIA.

O loader usa `torch.load(weights_only=True)` e verifica arquitetura/escala. O SPAN materializa 20 convoluções de inferência uma única vez em FP32, antes de converter a cópia para FP16. A comparação FP32 original/fusão deu erro zero nos exemplos testados; os testes ONNX dinâmicos e FP16 passaram. Não há promessa de melhoria de qualidade em relação aos checkpoints originais.

Modelos 4x produzem 4x neural e são redimensionados à saída do preset. Isso não os transforma em modelos 2x nativos. Não há substituição silenciosa por Adore se o backend nativo falhar.

## Medições na RX 5500 XT

5 aquecimentos e 20 quadros RGB únicos medidos por caso. Mediana do renderer, incluindo cópias/codificação; exclui navegador e socket. Não é o benchmark unificado Zenkai de 50/500 iterações.

| Modelo | 360→720 | 720→1080 | 1080→2160 |
|---|---:|---:|---:|
| HFA2K Real-CUGAN 2x | 73,27 ms | 276,58 ms | 1403,99 ms |
| RealESR AnimeVideo v3 4x | 38,03 ms | 143,48 ms | 736,68 ms |
| SPAN MS-SSIM 4x | 43,12 ms | 163,83 ms | 824,40 ms |

Acima de 1280×720 pixels de entrada, ROCm utiliza blocos de 512 com contexto e sobreposição para conter VRAM. Esse modo é aproximado para redes com estatísticas globais, especialmente Real-CUGAN; não declarar equivalência ao processamento integral. O custo em 4K é alto. HIP graph foi desativado nesses três modelos porque mantém muitos intermediários; a execução normal reutiliza memória.

Evidências: `diagnostics/user-model-export.json`, `diagnostics/user-model-gpu.json`, `diagnostics/external-stream-validation.json`. Testes reais de WebSocket cobrem mudança de tamanho e escala 2x/4x, com comparação à saída ONNX FP32.

Origem dos checkpoints: arquivos fornecidos pelo usuário. Licenças específicas dos pesos não verificadas; não redistribuídos. Real-CUGAN HFA2K não foi apresentado como checkpoint oficial conservative ou como ground truth.
