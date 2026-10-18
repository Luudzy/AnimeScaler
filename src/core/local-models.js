export const ADDED_MODELS = Object.freeze({
  'fsr3-native': {label:'AMD FSR 3.1 · vídeo experimental',scale:1,releaseStatus:'experimental'},
  'zenkai-1-0': {label:'Zenkai 1.0',scale:2},
  'zenkai-realesr-clean5k': {label:'Zenkai Real Clean 5k',scale:2},
  'zenkai-animevideo-30k': {label:'Zenkai AnimeVideo 30k FP16',scale:2,releaseStatus:'experimental'},
  'hfa2k-cugan': {label:'HFA2K Real-CUGAN',scale:2},
  'realesr-anime': {label:'RealESR AnimeVideo v3',scale:4},
  'anime4k-s': {label:'Anime4K S',scale:2,rocmOnly:true},
  'anime4k-m': {label:'Anime4K M',scale:2,rocmOnly:true},
  'realesr-anime4k-s-360': {label:'RealESR 360p + Anime4K S',scale:4,rocmOnly:true},
  'span-mssim': {label:'SPAN MS-SSIM',scale:4},
  'anime4kcpp-acnet-f8b8-hdn': {label:'Anime4KCPP ACNet f8b8 Denoise · OpenCL',scale:2,openclOnly:true},
  'anime4kcpp-acnet-f8b4': {label:'Anime4KCPP ACNet f8b4 · OpenCL',scale:2,openclOnly:true},
  'anime4kcpp-arnet-f8b8': {label:'Anime4KCPP ARNet f8b8 · OpenCL',scale:2,openclOnly:true},
  'anime4kcpp-artcnn-c4f16': {label:'Anime4KCPP ArtCNN C4F16 · OpenCL',scale:2,openclOnly:true},
});
const WEBGPU_MODELS = new Set(['animejanai','adore','fsr1-spatial','asr-anime-spatial','asr-anime-temporal']);
const LOCAL_MODELS = new Set(['adore-lite','adore-lite-v3','hfa2k-cugan','realesr-anime','span-mssim']);
const UNKNOWN_MODEL_REASON = 'Este modelo salvo não consta no catálogo disponível. A seleção foi preservada, mas não pode ser ativada.';
const MODEL_BACKENDS = Object.freeze({
  adore: ['rocm', 'tensorrt'],
  animejanai: ['rocm', 'tensorrt'],
  'adore-lite': ['rocm', 'tensorrt'],
  'adore-lite-v3': ['rocm', 'tensorrt'],
  'fsr1-spatial': ['webgpu'],
  'asr-anime-spatial': ['webgpu'],
  'asr-anime-temporal': [],
  'zenkai-1-0': ['rocm', 'tensorrt'],
  'zenkai-realesr-clean5k': ['rocm'],
  'zenkai-animevideo-30k': ['rocm'],
  'hfa2k-cugan': ['rocm', 'tensorrt'],
  'realesr-anime': ['rocm', 'tensorrt'],
  'anime4k-s': ['rocm'],
  'anime4k-m': ['rocm'],
  'realesr-anime4k-s-360': ['rocm'],
  'span-mssim': ['rocm', 'tensorrt'],
  'anime4kcpp-acnet-f8b8-hdn': ['opencl'],
  'anime4kcpp-acnet-f8b4': ['opencl'],
  'anime4kcpp-arnet-f8b8': ['opencl'],
  'anime4kcpp-artcnn-c4f16': ['opencl'],
  'fsr3-native': ['d3d12'],
});
const WEBGPU_PENDING_MODELS = new Set(['adore','animejanai','asr-anime-temporal']);
const BACKEND_LABELS = Object.freeze({
  webgpu: 'WebGPU', rocm: 'ROCm local', tensorrt: 'TensorRT local',
  opencl: 'OpenCL local', d3d12: 'DirectX 12',
});
const BACKEND_IDS = new Set(Object.keys(BACKEND_LABELS));
const modelLabel = model => ADDED_MODELS[model]?.label ?? model;
const routeNames = (model, backends) => backends.map(backend => BACKEND_LABELS[backend]).join(' ou ')
  || (WEBGPU_MODELS.has(model)
    ? `somente WebGPU${WEBGPU_PENDING_MODELS.has(model) ? ', com qualificação pendente e ativação bloqueada' : ''}`
    : 'nenhum backend compatível registrado');
export const isAddedModel = model => Object.hasOwn(ADDED_MODELS,model);
export const requiresNative = model => isAddedModel(model) || String(model).startsWith('adore-lite');

export const isRocmOnly = model => ADDED_MODELS[model]?.rocmOnly === true;
export const isOpenclOnly = model => ADDED_MODELS[model]?.openclOnly === true;
export const isFsrSpatialModel = model => model === 'fsr1-spatial';
export const isSpatialUpscalerModel = model => isFsrSpatialModel(model) || model === 'asr-anime-spatial' || model === 'asr-anime-temporal';

export function resolveModelCompatibility(model, backend) {
  const releaseStatus = ADDED_MODELS[model]?.releaseStatus;
  const known = isAddedModel(model) || LOCAL_MODELS.has(model) || WEBGPU_MODELS.has(model);
  const unavailable = (status, reason) => ({available:false, status, reason, releaseStatus});
  if (!known || !Object.hasOwn(MODEL_BACKENDS, model)) return unavailable('unknown-model', UNKNOWN_MODEL_REASON);
  if (!BACKEND_IDS.has(backend)) return unavailable('unknown-backend', `Backend ${String(backend)} não consta no catálogo.`);

  if (backend === 'webgpu' && WEBGPU_PENDING_MODELS.has(model)) {
    const family = model === 'adore' || model === 'animejanai' ? 'engine ONNX WebGPU' : 'pipeline de shaders WebGPU';
    return unavailable('qualification-pending', `${modelLabel(model)}: WebGPU ainda não foi qualificado para ativação deste modelo (${family} existe).`);
  }

  const allowedBackends = MODEL_BACKENDS[model];
  if (allowedBackends.includes(backend)) return {available:true, status:'route-available', reason:null, releaseStatus};

  if (backend === 'webgpu') {
    return unavailable('unsupported', `${modelLabel(model)} não tem executor WebGPU. Este modelo exige ${routeNames(model, allowedBackends)}.`);
  }
  return unavailable('unsupported', `${modelLabel(model)} não tem rota ${BACKEND_LABELS[backend]}. Este modelo exige ${routeNames(model, allowedBackends)}.`);
}
