# FSR WebGPU validation — 2026-10-09

## Correction: black screen and superseded measurements

The earlier 0.47–0.61 ms/frame measurements were invalid. Chromium rejected
the bind group because entry 1 (unused sampler) was absent from the automatic
pipeline layout. Queue completion alone did not prove successful rendering.
The sampler declaration, allocation and entry have now been removed.

Initialization checks a validation error scope; asynchronous GPU errors are
reported to diagnostics and surfaced on the next frame rather than silently
continuing with a black canvas. The offline quality page also needed to copy
the transient canvas before yielding to presentation. Player rendering still
uses GPU passes without CPU pixel readback.

The tables below are corrected runs with output pixel checks on the AMD
RDNA1 WebGPU adapter. Raw reports: `fsr-valid-1080p.json`, `fsr-valid-4k.json`,
`asr-valid-1080p.json`, `asr-valid-4k.json`.

## Executed

- WebGPU adapter requested with `powerPreference: high-performance`.
- EASU and RCAS WGSL modules compiled on the browser WebGPU device.
- Both render pipelines were created.
- Synthetic 1280×720 and 1920×1080 canvas frames were copied directly with
  `copyExternalImageToTexture` and rendered to 1920×1080 and 3840×2160.
- `device.queue.onSubmittedWorkDone()` completed after the render.

## Result

`PASS: corrected EASU/RCAS render commands, GPU output pixel sanity and queue drain.`

## Synthetic queue-drain benchmark

The reproducible local page `diagnostics/fsr-webgpu-benchmark.html` was run
on the browser WebGPU device after a 20-frame warm-up. It creates a synthetic
canvas source with a distinct colour marker per frame, submits render passes,
then waits for the queue to drain. It is batch throughput including marker
updates, not individual frame latency or an end-to-end playback measurement.

| Source → output | Completed frames | Batch throughput time | Output pixels |
| --- | ---: | ---: | ---: |
| 1280×720 → 1920×1080 | 120 | 0.86 ms/frame | PASS |
| 1920×1080 → 3840×2160 | 60 | 2.89 ms/frame | PASS |

## ASR Anime Spatial validation

ASR reuses EASU but substitutes an anime-oriented refinement pass: it applies
small smoothing in flat regions, preserves high-contrast lineart and clamps
sharpening to the local colour range. The real browser WebGPU device compiled
and rendered it successfully.

| Source → output | Completed frames | Batch throughput time | Output pixels |
| --- | ---: | ---: | ---: |
| 1280×720 → 1920×1080 | 120 | 0.83 ms/frame | PASS |
| 1920×1080 → 3840×2160 | 60 | 2.92 ms/frame | PASS |

Decoder, capture, presentation and browser composition remain outside this
synthetic measurement. These numbers do not establish player FPS.

## Real-video validation

144 frozen frames (48 per clip), 640×360 inputs and 1280×720 references.
All-black GPU output on a non-black reference now fails the diagnostic.
No GPU validation errors occurred after the correction.

| Clip | Metric | Bilinear | FSR 1 | ASR |
| --- | --- | ---: | ---: | ---: |
| 60s | RGB MAE ↓ | .010909 | .010823 | .010495 |
| 60s | Mean frame PSNR ↑ | 32.796 | 33.379 | 33.511 |
| 60s | Reference-compensated temporal MAE ↓ | .001134 | .001139 | .001087 |
| 600s | RGB MAE ↓ | .012959 | .013063 | .012782 |
| 600s | Mean frame PSNR ↑ | 31.922 | 32.269 | 32.399 |
| 600s | Reference-compensated temporal MAE ↓ | .010308 | .010595 | .010227 |
| 1200s | RGB MAE ↓ | .002362 | .002361 | .002349 |
| 1200s | Mean frame PSNR ↑ | 46.571 | 46.495 | 46.588 |
| 1200s | Reference-compensated temporal MAE ↓ | .000432 | .000444 | .000435 |

Raw report: `fsr-asr-real-video-144.json`; visual evidence:
`fsr-black-screen-fixed.png`. ASR gains are small; temporal error in the
1200s clip remains slightly worse than bilinear. FSR results are mixed.
This does not establish restoration quality on arbitrary noisy streams.

## Reproduction

Start a local HTTP server at project root, then:

```powershell
node diagnostics/check-fsr-render.cjs 'http://127.0.0.1:8123/diagnostics/asr-video-quality.html?all=1' diagnostics/fsr-asr-real-video-144.json
node diagnostics/check-fsr-render.cjs 'http://127.0.0.1:8123/diagnostics/fsr-webgpu-benchmark.html?preset=4k' diagnostics/fsr-valid-4k.json
npm test
```

The helper uses locally installed Playwright/Chromium; override their paths
with `ANIMESCALER_PLAYWRIGHT` and `ANIMESCALER_CHROMIUM` on other machines.
Regression suite: **155/155 JavaScript tests passed**. No native model changed.

## Scope

This validates shader compilation, texture upload, the two GPU passes and
presentation. It does not measure playback FPS on a real site. The runtime HUD
reports CPU submit time for this queued GPU path, not a full GPU completion
latency measurement.

## FSR 4 boundary

The downloaded DLLs are AMD DirectX 12 providers, not browser shaders. Their
native API requires a command list and temporal rendering inputs unavailable to
decoded HTML5 video. The extension therefore exposes FSR 1 Spatial as a fast,
honest stream filter and does not claim to run FSR 4.
