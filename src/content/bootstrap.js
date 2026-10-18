import { bootstrapAnimeScaler } from './main.js';

// Registered content scripts run in the isolated world, without fetching an ESM entry through the page.
(() => {
  if (globalThis.__animeScalerBootstrapRequested) return;
  globalThis.__animeScalerBootstrapRequested = true;
  try {
    bootstrapAnimeScaler();
  } catch (error) {
    globalThis.__animeScalerBootstrapRequested = false;
    console.error('[AnimeScaler] Falha ao carregar:', error);
  }
})();
