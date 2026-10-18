## RealESR AnimeVideo v3 — RX 5500 XT

The production path remains the original RealESR AnimeVideo v3 checkpoint in FP16 through the ROCm service. No weights or output dimensions were changed.

### Controlled ROCm measurements

The benchmark used the actual `infer_to` dispatch, 10 warmups and 30 distinct inputs. It excludes browser, socket and PNG work.

| Input to output | Median |
|---|---:|
| 640x360 to 1280x720 | 32.85 ms |
| 1280x720 to 1920x1080 | 125.53 ms |
| 1280x808 to 3424x2160 | 151.71 ms |

Autotuning did not help: 149.18 ms without it versus 149.43 ms with it for the 1280x808 probe. Channels-last was slower. `torch.compile` preserved exact output (`max_abs_error=0`, `mean_abs_error=0`) but measured 33.67 ms versus 33.67 ms eager on the 2x head.

## 1080p -> 4K target

The practical target is now 24 FPS, which allows at most 41.67 ms per frame. A targeted renderer probe at 1920x1080 -> 3840x2160 measured:

| Path | Median model/renderer time | Equivalent FPS | Result |
|---|---:|---:|---|
| `adore-lite` | 48.84 ms | 18.75 | closest current path |
| `adore-lite-v3` | 116.62 ms | 8.36 | too slow for this target |
| `realesr-anime` | process failed while switching models | NOT MEASURED | `HIP_INFRA_FAILURE`, retry separately |
| `anime4k-s` | 871.38 ms | 1.15 | rejected for this renderer path |
| `hfa2k-cugan` | 2776.81 ms | 0.36 | rejected for this renderer path |
| `span-mssim` | 1924.35 ms | 0.52 | rejected for this renderer path |

These are renderer-only timings with distinct random frames; they are not player FPS. The current Lite path is about 17% over the 24-FPS budget, so the next useful optimization is a dedicated 2x execution path (ONNX/MIGraphX or a persistent native Vulkan worker) rather than another 4x model. Lowering the AI input resolution would meet the timing target but would reduce the detail being requested and is therefore not the quality-preserving default.

The 20 ms target cannot be reached by the current RealESR architecture on this GPU without reducing model work or changing the model. Adore Lite V3 measured 7.71 ms for 640x360 to 1280x720 and remains the appropriate performance option; RealESR remains the quality option.

### NCNN/Vulkan probe

Vulkan 1.3.292 is available on the RX 5500 XT. The official Real-ESRGAN NCNN Vulkan package is stored at `F:/AnimeScaler-vulkan/release-20220424` and contains AnimeVideo v3 x2, x3 and x4 model files. The persistent Python NCNN binding loaded the x2 model and returned the expected 1280x720 shape.

However, five distinct 640x360 frames measured **4603.84 ms/frame median** through the Python binding, including RGB conversion and GPU completion. The standalone executable also includes process/model startup and file I/O, so neither path is suitable for live playback. Vulkan was not registered as a production backend. The experimental code is preserved in `native/ncnn_vulkan_backend.py`, `native/benchmark_ncnn_vulkan.py` and `native/test_ncnn_vulkan.py`.

References: [NCNN Vulkan FAQ](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md), [Real-ESRGAN AnimeVideo model](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md), [Real-ESRGAN NCNN Vulkan release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0).
