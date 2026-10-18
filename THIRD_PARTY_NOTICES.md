# Third-party notices

No model weights, checkpoints, or Anime4K GLSL shader files are included in this repository. The user must obtain model assets from their upstream sources and follow each asset's license.

## ESPAN reference for the Zenkai experiment

The official NTIRE2025 ESPAN deployment source is preserved in
`native/vendor/espan/team36_espan.py`, pinned to organizer repository commit
`b69fb668c0362deb696eecbfdebfaa5c2fcdfcb4`. Source:
[NTIRE2025_ESR](https://github.com/Amazingren/NTIRE2025_ESR).
The MIT notice, Copyright (c) 2025 BinRen, is retained in
`native/vendor/espan/LICENSE`. Zenkai's native x2 head, configurable depth and
training branches are documented adaptations; they are not represented as
the authors' unreleased training implementation. No official ESPAN checkpoint
has been downloaded, copied, or redistributed; its weight license remains
unverified. See `docs/zenkai-espan-research.md` for paper and code provenance.

## Adore / Re-SISR

Model: **2x Adore** by **renarchi** (Real-CUGAN architecture).
Source: https://github.com/renarchi/Re-SISR/releases/tag/Adore
License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

The setup script downloads the model as a separate third-party asset into the user's local `models/` folder. Do not use the Adore model commercially unless you obtain rights compatible with that use. If you redistribute the model or an adaptation, preserve the required attribution and license terms.

Expected SHA-256:
- `2x_Adore_renarchi_fp16.onnx`: `79e007eda5721cfa596945d5849320e2fad67f8596f9ac31ed06cc8e3ddaf57d`
- `2x_Adore_renarchi_fp32.onnx`: `532c1b30711c2bb7ea6ddf123fd219df87ca9816fbb8ddaaad5c1f311d5b7c44`

The local experimental `models/adore-lite-experimental.npz` student was distilled
from Adore outputs using user-provided screenshots. It is an experimental Adore
adaptation under the same CC BY-NC-SA 4.0 terms and attribution. Its smaller
architecture and training procedure are in `native/lite_model.py` and
`native/train_lite.py`. Checkpoints, provenance reports, user screenshots, and
visual comparisons are not included in version control.

## User-supplied external checkpoints (2026-09-30)

The local files `2xHFA2kReal-CUGAN.pth`, `realesr-animevideov3.pth` and
`4xmssim_span_pretrain.pth` were supplied by the user. Their exact checkpoint
provenance and redistribution permissions are not established by their names.
Registry metadata deliberately distinguishes code licenses from unverified
checkpoint licenses. These weights have not been published by this task.

Architecture upstreams checked against primary sources:

- Real-CUGAN, bilibili: [MIT code license](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/LICENSE).
- Real-ESRGAN / SRVGGNetCompact, Xintao Wang: [BSD-3-Clause code license](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
- SPAN, Cheng Wan et al.: [Apache-2.0 project license](https://github.com/hongyuanyu/SPAN/blob/main/LICENSE.txt).

The HFA2K checkpoint is not represented as the official conservative Real-CUGAN
checkpoint. Inference uses the installed Spandrel implementations. SPAN deploy
fusion materializes its existing evaluation convolutions once; it does not
rename the model as Zenkai.

## ONNX Runtime Web (runtime)

Microsoft ONNX Runtime Web is installed from npm and copied into `vendor/` by the setup script. See the package's own license files and repository for its license terms.

## AnimeJaNai HD V3 SuperUltraCompact

Model by **the-database**, from https://github.com/the-database/mpv-AnimeJaNai.
License: **CC BY-NC-SA 4.0**, https://creativecommons.org/licenses/by-nc-sa/4.0/.
The original and local WebGPU/FP16 derivatives are not included. The setup prepares local derivatives from the official original after the user downloads it. Local changes correct output dimension metadata and convert internal computation and weights to FP16. See [models/ANIMEJANAI.md](models/ANIMEJANAI.md) for provenance, hashes, conversion steps and validation limitations.


### Anime4K supplied GLSL and PyTorch equation port

Anime4K v3.2 S/M shader sources: copyright (c) 2019–2021 bloc97, MIT. Shader files are not included; obtain them from https://github.com/bloc97/Anime4K . `native/anime4k_model.py` implements their equations in PyTorch; it is not an mpv shader loader. The optional RealESR combination also needs a separate Real-ESRGAN checkpoint; Anime4K's MIT license does not relicense those weights.

## AMD FidelityFX Super Resolution 1 WebGPU path

`src/engine/fsr-shaders.js` adapts the EASU and RCAS algorithms from AMD
FidelityFX Super Resolution 1 for the extension's WebGPU renderer. The FSR 1
source is MIT licensed (copyright AMD 2021). Upstream license: [license.txt](https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/license.txt).
The license text is included in [licenses/FSR1-MIT.txt](licenses/FSR1-MIT.txt).

The user-supplied FSR 4 DLLs are not redistributed or loaded by this
extension. They are DirectX 12 runtime providers intended for a native
application that supplies native GPU resources, depth and motion-vector data;
decoded browser video exposes none of those resources. The selectable stream
mode is therefore accurately labelled **AMD FSR 1 Spatial**, not FSR 4.

## AMD FidelityFX FSR 3 native experimental adapter

`native/fsr3-probe` compiles against the user-supplied FidelityFX SDK 2.3.0
headers and dynamically loads its official FSR 3.1.5 provider from
`Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll`. The SDK/DLL remain
external; they are not copied into the extension. SDK source and license:
https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK . Consult the local
SDK LICENSE and binary redistribution terms before distributing a bundle.
Our video adapter adds GPU-estimated motion/reactivity, constant-plane depth
and no camera jitter; it does not implement frame generation or FSR 4.
