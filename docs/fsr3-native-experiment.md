# Official FSR 3 native compatibility experiment

Historical offline probe. The subsequent extension integration is documented
in [FSR3 native video](fsr3-native-video.md). The status and measurements below
describe the isolated-frame prototype, before that integration.

Date: 2026-10-09. User requested actual FSR 3 integration, rather than renaming
the existing FSR 1 / ASR WebGPU shaders. The approved experiment uses the
user's local SDK, a separate process, bounded GPU work and preserved reports.

## Current state

| Operation | Status |
| --- | --- |
| Build offline DirectX 12 probe | PASS |
| Load official supplied upscaler DLL | PASS |
| Enumerate and explicitly select FSR 3.1.5 | PASS |
| Verify actual context provider ID | PASS |
| Execute and read back 360p→720p on RX 5500 XT | PASS, three isolated frames |
| Genuine temporal video reconstruction | NOT MEASURED / not implemented |
| FSR 3 frame generation | NOT IMPLEMENTED |
| End-to-end browser/player latency and FPS | NOT MEASURED |
| Extension FSR 3 selection | NOT IMPLEMENTED; release gate not met |

The native prototype does run AMD's FSR 3.1.5 provider on the GPU. It is not a
WebGPU shader renamed to FSR 3. It is also not a finished video integration.
FSR/ASR options in the extension remain the existing WebGPU implementations.

## Executed protocol

- MSVC x64, Windows SDK, system D3D12 and DXGI; no driver changes or DLL injection.
- Official `amd_fidelityfx_upscaler_dx12.dll` from the SDK root configured by
  `ANIMESCALER_FSR_SDK` (`Kits/FidelityFX/signedbin`).
- DLL SHA256: `d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46`.
- API returned available providers **3.1.5** and **2.3.4**. Explicit override
  selected 3.1.5; actual context provider ID checked against requested ID.
- Hardware D3D12 adapter **AMD Radeon RX 5500 XT**, no WARP/CPU rendering fallback.
- One GPU submission per process; bounded fence wait (10 seconds) and external
  process timeout (60 seconds). No automatic retry or concurrent GPU runs.
- First frozen LR frame from each existing 60s, 600s and 1200s clip:
  640×360 RGBA input, 1280×720 RGBA output.
- Constant depth 0.5, zero motion vectors, zero camera jitter, reset history on
  the dispatch, sharpening 0.2 and auto exposure. These are compatibility
  placeholders, not inferred depth/motion or a valid jittered video protocol.
- GPU upload/readback through native CPU staging buffers. This offline probe
  does not prove zero-copy browser integration.
- No FFX error callback messages; fence completed, device remained available,
  output byte count and nonblack pixels checked. FFX checks are not a claim of
  full D3D12 debug-layer or GPU-assisted validation.

## Isolated-frame quality smoke test

Both outputs compared against the same frozen 720p reference. Bilinear is
Pillow bilinear in this diagnostic, not the browser's scaler. RGB MAE is
normalized to [0,1] (lower better); PSNR in dB (higher better).

| Clip / first frame only | Bilinear MAE | FSR3 reset MAE | Bilinear PSNR | FSR3 reset PSNR |
| --- | ---: | ---: | ---: | ---: |
| 60s | .010698 | .011115 | 32.8794 | 32.7893 |
| 600s | .014201 | .014726 | 31.3489 | 31.2134 |
| 1200s | .000350 | .000393 | 52.1067 | 51.6036 |

The no-history compatibility adaptation is worse on these isolated-frame
errors. This is not evidence that properly integrated FSR3 fails as a model,
nor a definitive visual judgement. It does block claiming this shortcut
improves anime quality or shipping it as a finished FSR3 temporal backend.
Do not compare these three frames against whole-clip averages in ASR reports.

Raw reports and images:
[60s](../diagnostics/fsr3-native-probe-360p/report.json),
[600s](../diagnostics/fsr3-native-probe-600s/report.json),
[1200s](../diagnostics/fsr3-native-probe-1200s/report.json),
[quality smoke](../diagnostics/fsr3-native-probe-quality.json).
Each directory preserves input/output RGBA, output/bilinear PNGs, stderr,
stdout, command line, exit code and hashes. Existing models were not changed.

## Why this cannot yet be the extension option

The supplied official backend requires a native DirectX 12 device and command
list. WebGPU does not expose those to the extension. The existing native
bridge has transports, but assumes ROCm/TensorRT model services; it is not
currently an FSR3/D3D12 presentation path.

FSR3 temporal reconstruction additionally consumes motion and depth and is
designed for subpixel-jittered rendering. Decoded browser video lacks game
depth, camera jitter and engine motion vectors. Filling these with constants
is useful to prove GPU compatibility, but does not provide the information
needed for its intended temporal reconstruction. Resampling a decoded LR
frame with artificial jitter does not recover missing original detail.

An experimental video adaptation can estimate motion and reject uncertain
history, but must disclose its artificial depth and non-game input semantics.
It requires its own evaluation; the AMD integration contract is not satisfied
merely because the API accepted placeholders.

## Next implementation gate

1. Build an explicit native D3D12 video adapter with persistent contexts,
   bounded frames in flight and reliable reset/reconfiguration. Avoid routing
   it through a service labelled ROCm or claiming native FP16 model inference.
2. Estimate current→previous motion on GPU and build confidence/reactivity
   masks; test vector direction, scale, occlusions, cuts and subtitles. Keep
   estimated motion separate from real game-provided vectors.
3. Define decoded-video sampling/jitter and depth assumptions explicitly.
   Compare temporal adaptation against per-frame reset, ASR spatial and ASR
   temporal on identical decoded inputs. Do not use HR references to prepare
   production LR inputs or synthesize missing jitter data.
4. Evaluate all 144 frozen frames and motion/cut patterns, preserve individual
   regressions, measure native computation and browser transport/presentation
   separately. Do not call isolated API completion player FPS.
5. Expose an experimental option only when output correctness, useful quality
   and bounded playback are demonstrated. Frame generation is a separate
   implementation and evaluation task.

## Reproduce compatibility only

```powershell
& ./native/fsr3-probe/build.ps1
& 'C:/Program Files/Python310/python.exe' native/fsr3-probe/run.py `
  --input diagnostics/video-sequences/rezero-real-01/clip-60s-360-to-720/lr/00000000.png `
  --output-dir diagnostics/fsr3-native-probe-new-run
```

The runner needs Pillow and refuses to overwrite an existing report. SDK path
can be specified with `--sdk`; the build takes `-Sdk`. Both use the official
local headers/DLL and leave the SDK unchanged. The experiment is offline only.

Official AMD references:
[FSR3 upscaler integration](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/),
[FSR3 overview](https://gpuopen.com/fidelityfx-super-resolution-3/),
[official SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK).
