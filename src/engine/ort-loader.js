let ortPromise;
let importAttempts = 0;

export async function loadOrt() {
  if (!ortPromise) {
    const url = chrome.runtime.getURL('vendor/ort.webgpu.bundle.min.mjs');
    // ESM also caches rejected module loads; a retry needs a fresh module identity.
    const attemptUrl = importAttempts++ ? `${url}#ort-retry-${importAttempts}` : url;
    ortPromise = import(attemptUrl).then((ort) => {
      ort.env.logLevel = 'warning';
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = chrome.runtime.getURL('vendor/');
      return ort;
    }).catch(error => {
      ortPromise = undefined;
      throw error;
    });
  }
  return ortPromise;
}
