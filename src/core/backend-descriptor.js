import { normalizeSettings } from './settings.js';
import { resolveModelCompatibility } from './local-models.js';

export function resolveBackendDescriptor(value = {}) {
  const modelId = value?.model;
  const settings = normalizeSettings(value);
  const requestedBackend = settings.backend;
  if (!['webgpu', 'rocm', 'tensorrt', 'opencl', 'd3d12'].includes(requestedBackend)) {
    const error = new Error(`Backend ${requestedBackend} indisponível para o modelo ${modelId ?? settings.model}.`);
    error.code = 'BACKEND_UNAVAILABLE';
    throw error;
  }
  const modelBackend = resolveModelCompatibility(settings.model, requestedBackend);
  const descriptor = {
    id: requestedBackend,
    modelId: settings.model,
    executionHost: requestedBackend === 'webgpu' ? 'content' : 'extension-offscreen',
    outputFormat: 'rgba8',
    requiresLocalService: requestedBackend !== 'webgpu',
  };
  return {
    ...descriptor,
    ...(modelBackend.available ? {} : {available:false, reason:modelBackend.reason}),
    ...(modelBackend.releaseStatus ? {releaseStatus:modelBackend.releaseStatus} : {}),
  };
}
