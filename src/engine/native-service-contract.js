export function validateNativeService(result, requested) {
  if (result.backend !== requested.backend || result.protocol !== 1 || result.model !== requested.model) {
    throw new Error('Serviço nativo não corresponde ao backend/modelo solicitado.');
  }
  const modelMetadata = result.metadata?.[requested.model];
  const revision = result.revisions?.[requested.model];
  if (!Array.isArray(result.models) || !result.models.includes(requested.model) ||
      typeof revision !== 'string' || !revision.trim() ||
      !modelMetadata || typeof modelMetadata.status !== 'string' || !modelMetadata.status.trim() ||
      (modelMetadata.id !== undefined && modelMetadata.id !== requested.model)) {
    throw new Error('Servi\u00e7o nativo n\u00e3o confirmou identidade, revis\u00e3o e status do modelo.');
  }
  if (requested.backend === 'd3d12') {
    if (requested.model !== 'fsr3-native' || result.precision !== 'native' || !/^3\.1\.\d+$/.test(result.provider ?? '')) {
      throw new Error('Provider oficial FSR 3.1 não confirmado pelo serviço DirectX 12.');
    }
  } else if (requested.backend === 'opencl') {
    if (!requested.model.startsWith('anime4kcpp-') || result.precision !== 'native' || modelMetadata.provider !== 'Anime4KCPP OpenCL') {
      throw new Error('Provider Anime4KCPP OpenCL ou precis\u00e3o nativa n\u00e3o confirmados pelo servi\u00e7o.');
    }
  } else if (result.precision !== 'fp16') throw new Error('Precisão do serviço nativo incompatível.');
}
function nativeServiceName(backend, model) {
  if (typeof model === 'string' && model.startsWith('anime4kcpp-')) return 'OpenCL';
  if (backend === 'opencl') return 'OpenCL';
  if (backend === 'd3d12') return 'FSR 3';
  if (backend === 'tensorrt') return 'TensorRT';
  if (backend === 'rocm') return 'ROCm';
  return 'nativo';
}

export function serviceErrorText(backend, kind, model) {
  const isD3d12 = backend === 'd3d12';
  const service = nativeServiceName(backend, model);
  if (kind === 'timeout') return `Servi\u00e7o ${service} excedeu o tempo de resposta.`;
  if (kind === 'disconnected') {
    if (typeof model === 'string' && model.startsWith('anime4kcpp-')) {
      if (backend === 'opencl') return 'Servi\u00e7o OpenCL desconectado. Abra Iniciar-OpenCL.cmd.';
      return 'Worker OpenCL desconectado. Verifique o servi\u00e7o nativo configurado.';
    }
    if (isD3d12) return 'Servi\u00e7o FSR 3 desconectado. Abra Iniciar-FSR3.cmd.';
    if (backend === 'tensorrt') return 'Servi\u00e7o TensorRT desconectado. Abra Iniciar-TensorRT.cmd.';
    if (backend === 'opencl') return 'Servi\u00e7o OpenCL desconectado. Abra Iniciar-OpenCL.cmd.';
    if (backend === 'rocm') return 'Servi\u00e7o ROCm desconectado. Abra Iniciar-ROCm.cmd.';
    return 'Servi\u00e7o nativo desconectado.';
  }
  if (kind === 'invalid-response') return `Resposta ${service} inv\u00e1lida.`;
  throw new TypeError('Unknown native service error kind');
}

export function nativeServiceFailure(backend, kind, model) {
  const service = nativeServiceName(backend, model);
  if (kind === 'busy') return { code: 'NATIVE_SESSION_BUSY', message: `${service} est\u00e1 ocupado por outra sess\u00e3o.` };
  if (kind === 'timeout') return { code: 'BACKEND_TIMEOUT', message: serviceErrorText(backend, kind, model) };
  if (kind === 'disconnected') return { code: 'BACKEND_UNAVAILABLE', message: serviceErrorText(backend, kind, model) };
  if (kind === 'invalid-response') return { code: 'BACKEND_INVALID_RESPONSE', message: serviceErrorText(backend, kind, model) };
  if (kind === 'unavailable') return { code: 'BACKEND_UNAVAILABLE', message: `Falha no servi\u00e7o ${service}.` };
  throw new TypeError('Unknown native service failure kind');
}

export function classifyNativeServiceError(backend, message, model) {
  if (typeof message !== 'string') return nativeServiceFailure(backend, 'invalid-response', model);
  return message === 'GPU ocupada.'
    ? nativeServiceFailure(backend, 'busy', model)
    : nativeServiceFailure(backend, 'unavailable', model);
}

export function classifyNativeServiceClose(backend, close, model) {
  return close?.code === 1013 || close?.reason === 'FSR3 viewer already active'
    ? nativeServiceFailure(backend, 'busy', model)
    : nativeServiceFailure(backend, 'disconnected', model);
}
