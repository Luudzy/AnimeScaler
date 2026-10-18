# ASR Anime Temporal — experimental

Implemented on 2026-10-09 as a separate, opt-in WebGPU processing option.
Select **ASR Anime Temporal · WebGPU · experimental** after reloading the
extension. Start with balanced noise reduction and Quality or Native spatial
input quality. Existing model defaults remain unchanged.

This is not AMD FSR 3, neural FSR 4, optical flow, temporal AA or frame
generation. It is an anime-oriented spatial upscale with conservative temporal
denoising. No training was performed for this shader.

## GPU pipeline and history

Source upload → optional bilateral cleanup → temporal cleanup → optional input
reduction → EASU → anime refinement → canvas. The processing passes and previous
source copy share one command submission. There is no player pixel readback.

Two reusable rgba16float textures accumulate history at source resolution;
an rgba8 texture retains the previous raw source. Added texture memory is
approximately 20 bytes per source pixel (about 39.6 MiB at 1920×1080), excluding
driver overhead and existing spatial resources. Textures are recreated when
dimensions change, not every frame.

The filter compares raw current/previous pixel, cross-neighbourhood average
and gradient differences. It blends at most 45% accumulated history, clamped
to the current spatial neighbourhood. Movement or changed content reduces
history weight; this is per-pixel rejection, not global scene-cut detection.
There is no motion compensation, so moving objects mostly use the current
spatial result. It does not guarantee removal of ghosting or all codec damage.

History resets on initialization, dimension changes, crop changes, noise mode
changes, backwards/duplicate timestamps, media gaps over 250 ms, wall-clock
gaps over 500 ms, and explicit `resetHistory()`. Model changes recreate the
engine. `temporalHistoryAvailable` reports availability, not the fraction of
pixels actually using history.

## Matched real-video evaluation

Three frozen clips (60s, 600s, 1200s), 48 frames each: 144 distinct 640×360
inputs with 1280×720 references. Balanced spatial denoise and Native spatial
input quality for both ASR variants. History resets between clips; 47/48 frames
per clip have history available. Both runs use timestamps at 24000/1001 fps.
Source frames and references are identical. No final-hidden set was opened.

MAE and reference-compensated temporal MAE are normalized RGB errors (lower
is better); PSNR is the mean frame PSNR in dB (higher is better). Temporal MAE
measures `|(output[t]-output[t-1])-(reference[t]-reference[t-1])|`, without
optical-flow warping. These measurements do not replace visual playback review.

| Input | Clip | Spatial MAE | Temporal MAE | Spatial PSNR | Temporal PSNR | Spatial temporal error | Temporal temporal error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen LR | 60s | .010232 | .010266 | 33.6276 | 33.6154 | .001020 | .001029 |
| Frozen LR | 600s | .012286 | .012410 | 32.6409 | 32.6004 | .009671 | .009478 |
| Frozen LR | 1200s | .002340 | .002224 | 46.4547 | 46.5132 | .000429 | .000339 |
| Noise + JPEG | 60s | .014213 | .013656 | 31.7343 | 31.8805 | .010987 | .007976 |
| Noise + JPEG | 600s | .017382 | .017003 | 30.5008 | 30.5737 | .017057 | .015137 |
| Noise + JPEG | 1200s | .011861 | .011980 | 37.7025 | 37.7588 | .003021 | .001722 |

The damaged version adds deterministic RGB noise ±12/255 and JPEG quality
0.45 to the frozen LR inputs before all processing. It is not H264/H265/AV1
validation or a captured live stream.

Regressions are retained: clean MAE worsens approximately 0.33% and 1.01% in
the first two clips, and clean temporal error worsens approximately 0.86% in
the first. Damaged third-clip MAE worsens approximately 1.01% despite better
PSNR and temporal error. Damaged temporal error falls approximately 27.4%,
11.3% and 43.0% across the three clips. Keep the option experimental; there
is no universal quality winner or 100% denoising claim.

Raw reports (also include bilinear and FSR spatial):
[clean](../diagnostics/asr-temporal-clean-144.json),
[noise/JPEG](../diagnostics/asr-temporal-degraded-144.json).

## Synthetic motion/cut check

24 distinct noisy frames, including a moving two-pixel dark line and abrupt
background change, passed real Chromium WebGPU rendering on AMD RDNA1.
In the static ROI, spatial MSE was 11.7694 and temporal MSE 9.8164: 16.6% extra
reduction over balanced spatial denoise. The exposed old-line location averaged
126.96/255 against a 128/255 background. On the changed-background frame the
maximum temporal/spatial pixel difference was 1/255. This tests those specific
patterns, not all possible motion, text, cuts or ghosting.

[Raw synthetic report](../diagnostics/asr-temporal-noise-motion-cut.json).

## Performance scope

| Source → output | Quality | Frames | Queue-drain batch ms/frame |
| --- | --- | ---: | ---: |
| 1280×720 → 1920×1080 | Performance, balanced denoise | 120 | 1.18 |
| 1920×1080 → 3840×2160 | Native spatial, balanced denoise | 60 | 3.48 |

Distinct synthetic colour markers after warmup; output pixel sanity passed.
These are batch throughput measurements, not per-frame latency or player FPS.
They exclude video decoding, site capture and browser presentation/composition.
Do not compare these as end-to-end speed against the ROCm model HUD.

Raw reports: [1080p](../diagnostics/asr-temporal-performance-batch.json),
[4K](../diagnostics/asr-temporal-4k-batch.json).

## Reproduce

Serve the repository locally on port 8123, then run:

```powershell
node diagnostics/check-fsr-render.cjs 'http://127.0.0.1:8123/diagnostics/asr-temporal-check.html' diagnostics/asr-temporal-noise-motion-cut.json
node diagnostics/check-fsr-render.cjs 'http://127.0.0.1:8123/diagnostics/asr-video-quality.html?all=1&denoise=balanced&temporal=1' diagnostics/asr-temporal-clean-144.json
node diagnostics/check-fsr-render.cjs 'http://127.0.0.1:8123/diagnostics/asr-video-quality.html?all=1&denoise=balanced&temporal=1&degraded=1' diagnostics/asr-temporal-degraded-144.json
node diagnostics/check-spatial-popup.cjs
npm test
```

The helper requires installed Playwright/Chromium; path overrides are
`ANIMESCALER_PLAYWRIGHT` and `ANIMESCALER_CHROMIUM`. Offline evaluation reads
pixels back only for metrics and render validation, unlike the player.

Verification: 161/161 JavaScript tests passed, popup selection/persistence
passed, and synthetic noise/motion/cut rendering passed on the real AMD RDNA1
WebGPU adapter. No native model, training checkpoint or stable preset was
replaced. Live site playback/FPS and broad visual ghosting review remain
unmeasured for this experimental option.
