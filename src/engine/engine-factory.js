import { isSpatialUpscalerModel } from '../core/local-models.js';
import { resolveBackendDescriptor } from '../core/backend-descriptor.js';
import { AdoreEngine } from './adore-engine.js';
import { FsrEngine } from './fsr-engine.js';
import { NativeInlineEngine } from './native-inline-engine.js';

export function createProcessingEngine(settings) {
  const backend = resolveBackendDescriptor(settings);
  if (backend.available === false) {
    const error = new Error(backend.reason);
    error.code = 'MODEL_BACKEND_UNAVAILABLE';
    throw error;
  }
  if (backend.executionHost === 'extension-offscreen') return new NativeInlineEngine();
  return isSpatialUpscalerModel(backend.modelId) ? new FsrEngine() : new AdoreEngine();
}

export function requiresEngineRecreation(previous, next) {
  const previousBackend = resolveBackendDescriptor(previous);
  const nextBackend = resolveBackendDescriptor(next);
  if (previousBackend.executionHost !== nextBackend.executionHost) return true;
  if (previousBackend.executionHost === 'extension-offscreen') return false;
  if (previousBackend.id !== nextBackend.id) return true;
  const previousSpatial = isSpatialUpscalerModel(previousBackend.modelId);
  const nextSpatial = isSpatialUpscalerModel(nextBackend.modelId);
  return previousSpatial !== nextSpatial || (previousSpatial && nextSpatial && previous?.model !== next?.model);
}
