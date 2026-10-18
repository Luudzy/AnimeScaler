# Playback evidence — 2026-10-15

## ROCm inline run

Artifact: [`report.json`](../diagnostics/stability-20261015/native-inline-adore-lite-v3-stop-fix-bundle/report.json). The run used the unpacked extension, a local 360p MP4, ROCm on Radeon RX 5500 XT, `adore-lite-v3`, and 360p→720p output for 60 seconds.

- Native service initialized; the inline session remained `running` at all 12 five-second checks.
- Source cadence median: **24.02 FPS**. Processed cadence median: **4.46 FPS** (range **3.29–5.80 FPS**).
- Processing-window mean latency median: **203.9 ms**; worst observed p95: **341.9 ms**.
- No runtime or teardown errors were recorded. Stop completed without `STALE_SESSION` after rebuilding `content-bootstrap.js`.
- The source kept playing and its element was unmuted. This MP4 has a video track only, so the run does **not** establish audible audio continuity. [`report-review.json`](../diagnostics/stability-20261015/native-inline-adore-lite-v3-stop-fix-bundle/report-review.json) records that correction.

The 30-second `fluidVideo=true` comparison had a **5.12 FPS** processed median against **24.08 FPS** source cadence, with no errors. It does not close the playback gap.

## Interpretation

The stop race is fixed and verified by `tests/controller-stop-race.test.js`: cancellation after STOP no longer changes `stopped` to `error`. The live run confirms the session stays alive, but enhanced output still updates far below source cadence. This is enough to explain visibly stalled or juddering enhanced video while the page continues, but it does not reproduce the original error sequence or prove its root cause.

The prior `sourceAudioStillPlaying` check was mislabeled: it tested only that the source element was unmuted, not paused, and advancing. Do not use that field as audio evidence.

## Sustained-throughput guard

The controller now stops enhancement after five seconds below 80% of the lower of configured `maxFps` and observed source cadence. This guard applies only to ROCm. It hides the enhanced overlay and reports `THROUGHPUT_BELOW_TARGET`; it does not pause or mute the source element. Warmup, pause, source replacement, and reconfiguration reset the measurement. Unit tests prove the controller behavior, including a 12 FPS source and unchanged WebGPU behavior.

Real-browser proof: [`native-inline-audio-throughput-guard-v4/report.json`](../diagnostics/stability-20261015/native-inline-audio-throughput-guard-v4/report.json). A 10-second Chromium run on the RX 5500 XT reached **6.55 FPS** against a **19.18 FPS** minimum after a 5-second window. The controller stopped enhancement with `THROUGHPUT_BELOW_TARGET`; the canvas overlay was hidden, the video element remained unpaused and unmuted, and teardown completed. A live 440 Hz page audio signal remained active (RMS **0.028**); its synthetic audio element remained unpaused, unmuted, and live. This proves page audio continues during the guard.

An audio-bearing 15-second Chromium E2E is recorded in [`native-audio-track-live-20261015-01/report.json`](../diagnostics/stability-20261015/native-audio-track-live-20261015-01/report.json). It used an **8.008-second, 480×270 H.264/AAC** derivative of the local HEVC/AAC MKV under `tests/video-test/`; the MKV stayed unchanged. ROCm initialized `adore-lite-v3` and processed at **3.45–4.73 FPS**. The source element stayed unmuted and playing; Web Audio measured decoded AAC RMS **0.034–0.106** across three checks. The throughput guard stopped enhancement at **4.35 FPS** against a **19.18 FPS** minimum and hid the overlay. The runner completed without runtime errors. This verifies decoded source audio during inference and after the overlay stops. It does not measure long-run A/V sync drift or physical speaker output.

## Remaining proof

- Measure A/V sync drift over a long, non-looping audio-bearing source; current AAC evidence covers a short looping fixture and decoded RMS, not lip-sync tolerance.
- Test the 18.43 FPS direct `NativeEngine` result inside the extension content-script context. That earlier result used a local page, had only **0.12** mean AI coverage, and did not prove worker origin, private `connection.json` access, or token isolation.
- Compare both paths with the same clip, model, resolution, and temporal-quality checks. Do not promote the direct path or claim 24 FPS until those checks pass.
