# Native content-context probe — 2026-10-15

## Result

The direct `NativeEngine` content-script route failed its first capability gate in Playwright Chromium. The content script imported the extension module and reached the existing Worker constructor, but Chromium rejected the Worker before service authentication or frame inference:

> `Failed to construct 'Worker': Script at 'chrome-extension://…/src/engine/native-worker.js' cannot be accessed from origin 'http://127.0.0.1:8765'.`

Evidence: [`sanitized report`](../diagnostics/stability-20261015/native-content-context-probe/chromium-final/report.json). No service token, raw frame, or page URL beyond the local diagnostic origin is recorded.

The probe also confirmed that the temporary extension service worker can read the private `native/connection.json` and return it to the isolated content script through extension messaging. A page-context fetch of that file was blocked, and the file remained absent from `web_accessible_resources`. This does not prove the direct route authenticates: Worker creation failed first.

## Limits and decision

- Browser tested: Playwright Chromium only. Chrome Stable and Edge executables were not found at their standard installation paths, so this is not a Chrome/Edge compatibility result.
- No worker, WebSocket, token authentication, GPU frame, or output quality measurement completed.
- Keep the current ROCm routing. Do not add `native/connection.json` to `web_accessible_resources`, relax `allowed_origin`, or claim direct-content performance from the local-page 18.43 FPS result.
- Architecture review can test an extension-origin iframe/offscreen worker with transferable frames, or optimize the existing PNG relay. Any bridge must preserve private config, origin checks, cleanup, and page isolation.

The test-only temporary extension was removed after each run. Product source, manifest, permissions, and runtime routing were not changed.
