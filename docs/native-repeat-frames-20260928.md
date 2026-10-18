# Exact native frame reuse

Native inference now compares every input byte against the last successfully processed input when model, transport and input/output geometry remain compatible. An identical frame reuses the existing output; any changed byte forces inference. Failed computation invalidates reuse before writing the input buffer. This is exact deduplication, not interpolation or approximate motion detection.

WebSocket clients may opt in with `repeat=1`. ASC4 is a 24-byte response containing the new frame ID and output dimensions, without pixels. The service only sends it after the same connection has received the current renderer revision. Other clients and older workers receive complete pixel responses. JPEG does not negotiate compact repeats. The worker retains its last decoded result and validates IDs/dimensions before reuse. A previously rejected hybrid result is uploaded when subsequently accepted; already uploaded pixels do not need another GPU upload. Output resize invalidates that upload identity.

Adaptive and saved performance profiles exclude reused frames so still scenes do not falsely imply that a heavier input is affordable. Logs aggregate reusedFrames and responseBytesTotal without splitting one event per frame. Frame uploads to the service still occur; these optimizations eliminate duplicate GPU inference and most response traffic, not all capture/upload costs. The retained CPU result is bounded; during hybrid rejection an older uploaded image and the latest decoded result may coexist.

RX 5500 XT / Lite measurement, native render duration only (not browser FPS):

| Input → output | Unique frame mean | Identical frame mean | Full RGB response → compact |
|---|---:|---:|---:|
| 360p → 720p | 4.53 ms | 0.14 ms | 2,764,824 → 24 bytes |
| 720p → 1080p | 14.59 ms | 0.94 ms | 6,220,824 → 24 bytes |
| 1080p → 4K | 34.74 ms | 2.06 ms | 24,883,224 → 24 bytes |

Validation: 129 JS tests and 14 native tests passed. Production worker/compositor pixel checks passed in WebGPU and Canvas2D using a local deterministic socket fixture; real GPU render measurements separately verified full-output equality and changed-byte invalidation. Socket integration tests cover per-connection negotiation and revision changes. The worker fixture is a test, not an end-to-end benchmark against a running native service. Real-site repeat frequency and playback FPS remain unmeasured. Additional independent agent review was unavailable due to the account usage limit; no independent review pass is claimed.

Restart the native service, reload the extension, and reopen the viewer to use the complete path. This round changed runtime code; the previously validated 16-channel Lite weights remain installed.
