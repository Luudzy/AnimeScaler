# ASR Anime Spatial

A separate opt-in [ASR Anime Temporal](asr-temporal.md) variant adds GPU history
for stationary-region noise reduction. The spatial option remains available;
the temporal version has measured regressions and is experimental.

ASR is the anime-focused spatial GPU path in AnimeScaler. It is intentionally
not an ONNX model and has no Lite variant: its core uses two WebGPU render
passes. Optional denoising and input reduction add GPU passes, without CPU
pixel readback in the player.

## Pipeline

```text
HTMLVideoElement
  -> copyExternalImageToTexture
  -> optional source-resolution bilateral denoise
  -> optional GPU reduction for selected FSR quality preset
  -> EASU edge-directed upscale
  -> ASR anime refinement
  -> canvas
```

The EASU pass is the FSR 1 spatial reconstruction stage. The ASR refinement
uses a five-tap cross neighbourhood and applies three bounded operations:

1. estimates local luminance range;
2. blends a small amount of cross-neighbour smoothing only in flat areas;
3. sharpens non-flat regions, then clamps the result to the local RGB range.

The clamp and the flatness gate are deliberate. They aim to preserve cel-shaded
lineart and suppress mild flat-area noise without making white/black halos or
inventing texture.

## Scope

ASR can improve scaling appearance, thin lines and mild compression noise. It
cannot recover information that the decoded frame no longer contains, remove
heavy macroblocks reliably, or match a trained restoration model on damaged
material. The option is useful when low latency is more important than AI
reconstruction.

## Validation

For the newer denoising controls, quality presets and matched noise/JPEG tests,
see [FSR video cleanup and presets](fsr-video-denoise-presets.md). The benchmark
table below describes the earlier two-pass path with noise reduction disabled.

On 2026-10-09, after fixing an unused-sampler bind-group mismatch, Chromium
rendered valid output through the AMD RDNA1 WebGPU adapter:

| Source -> output | Queue-drain result |
| --- | ---: |
| 1280x720 -> 1920x1080 | 0.83 ms/frame across 120 frames |
| 1920x1080 -> 3840x2160 | 2.92 ms/frame across 60 frames |

These are synthetic batch throughput measurements with pixel checks, not
individual frame latency or player FPS. They include marker updates,
submission and queue draining after warm-up; exclude site decoding, capture and
browser composition. Earlier 0.49/0.61 ms measurements were invalid because the
old benchmark accepted queue completion with rejected render commands.

The frozen 144-frame 360p→720p test passes render validation. ASR improves RGB
MAE slightly in all three clips; temporal error improves in two clips and is
slightly worse than bilinear in the third. See
[the complete validation](../diagnostics/fsr-webgpu-validation.md) for individual
metrics, regressions and reproduction commands. ASR is a spatial shader,
not a trained neural restoration model.
