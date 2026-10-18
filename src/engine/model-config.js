import { isAddedModel, isOpenclOnly, isRocmOnly } from '../core/local-models.js';
const MODELS = Object.freeze([
  {
    id: 'adore-fp16',
    label: 'Adore',
    precision: 'fp16',
    path: 'models/2x_Adore_renarchi_webgpu_fp16.onnx',
    sha256: 'e503b357fe6d6ce34c3db9980cc2ed37dd1f23d4772b352c8a4f2befb151f2b5',
  },
  {
    id: 'adore-fp32',
    label: 'Adore',
    precision: 'fp32',
    path: 'models/2x_Adore_renarchi_webgpu.onnx',
    sha256: 'b0a76dff7816816dee2c9a19f58bb1a81dd68178547f9518b64bc0257205749d',
  },
]);

function runtimeUrl(path) {
  const getter = globalThis.chrome?.runtime?.getURL;
  return getter ? getter(path) : path;
}

export function getModelCandidates(getUrl = runtimeUrl, selection = 'adore') {
  if (selection === 'fsr3-native') throw new Error('FSR 3 exige DirectX 12 local. Abra Iniciar-FSR3.cmd.');
  if (isOpenclOnly(selection)) throw new Error('Anime4KCPP requires local OpenCL. Open Iniciar-OpenCL.cmd.');
  if (isRocmOnly(selection)) throw new Error('Este port do Anime4K suporta apenas ROCm. Abra o serviço local e selecione ROCm.');
  if (isAddedModel(selection)) throw new Error('Este modelo exige ROCm ou TensorRT. Abra o serviço local e selecione o backend nativo.');
  const models = selection === 'animejanai' ? [{
    id: 'animejanai-fp16', label: 'AnimeJaNai SuperUltraCompact', precision: 'fp16',
    path: 'models/2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu_fp16.onnx',
    sha256: '1c897f35e51fd4016392fd658888e7603d9ae95acace640fee9872a9c6147172',
  }, {
    id: 'animejanai', label: 'AnimeJaNai SuperUltraCompact', precision: 'fp32',
    path: 'models/2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu.onnx',
    sha256: '3eca0dee498cc0bfb311a6ed97e23bf225f7660de033ca9683ffefd155c0b164',
  }] : MODELS;
  return models.map((model) => ({ ...model, url: getUrl(model.path) }));
}


export function supportsFloat32GpuIo(inputMetadata, outputMetadata, inputShape, outputShape) {
  const validDimensions = metadata => Array.isArray(metadata?.shape) && metadata.shape.length === 4 &&
    metadata.shape[0] === 1 && metadata.shape[1] === 3 && metadata.shape.slice(2).every(value =>
      (Number.isSafeInteger(value) && value > 0) || (typeof value === 'string' && value.trim().length > 0));
  if (!Boolean(
    inputMetadata?.isTensor
    && outputMetadata?.isTensor
    && inputMetadata.type === 'float32'
    && outputMetadata.type === 'float32'
  ) || !validDimensions(inputMetadata) || !validDimensions(outputMetadata)) return false;
  for (const axis of [2, 3]) {
    const input = inputMetadata.shape[axis], output = outputMetadata.shape[axis];
    if (typeof input === 'number' && typeof output === 'number' && output !== input * 2) return false;
    if (typeof input === 'string' && input === output) return false;
  }
  const matches = (metadata, shape) => !shape || metadata.shape.every((value, axis) =>
    typeof value === 'string' || value === shape[axis]);
  return matches(inputMetadata, inputShape) && matches(outputMetadata, outputShape);
}
