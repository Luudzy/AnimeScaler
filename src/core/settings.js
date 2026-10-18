import { isOpenclOnly, isRocmOnly, requiresNative } from './local-models.js';
const PRESET_MAX_WIDTH = Object.freeze({
  '720p': 640,
  // Native x2 path: 540p input -> 1080p output.
  '1080p': 960,
  '1080p-fast': 960,
  '1440p': 1280,
  '4k': 1920,
  performance: 320,
  balanced: 960,
  quality: 1280,
  native: 1920,
});

const STEP_WIDTHS = Object.freeze([1920, 1280, 960, 640, 480, 320, 240, 160]);
const MAX_MODEL_EDGE = 1920;
const MAX_MODEL_PIXELS = 1920 * 1080;
const VALID_PRESETS = new Set(Object.keys(PRESET_MAX_WIDTH));
export const FSR_SCALE_FACTORS = Object.freeze({performance:2,balanced:1.7,quality:1.5,'ultra-quality':1.3,'native-aa':1});

export function resolveFsrInputSize(sourceWidth, sourceHeight, outputWidth, outputHeight, quality = 'native-aa') {
  const divisor = Object.hasOwn(FSR_SCALE_FACTORS, quality) ? FSR_SCALE_FACTORS[quality] : 1;
  const scale = Math.min(1, outputWidth / divisor / sourceWidth, outputHeight / divisor / sourceHeight);
  return {width:Math.max(1,Math.floor(sourceWidth*scale)),height:Math.max(1,Math.floor(sourceHeight*scale))};
}

export function normalizeSettings(value = {}) {
  const source = value && typeof value === 'object' ? value : {};
  const model = typeof source.model === 'string' && source.model ? source.model : 'animejanai';
  const defaultBackend = model === 'fsr3-native' ? 'd3d12' : isOpenclOnly(model) ? 'opencl' : isRocmOnly(model) || requiresNative(model) ? 'rocm' : 'webgpu';
  return {
    backend: ['webgpu', 'rocm', 'tensorrt', 'opencl', 'd3d12'].includes(source.backend) ? source.backend : defaultBackend,
    exclusiveViewer: typeof source.exclusiveViewer === 'boolean' ? source.exclusiveViewer : false,
    fluidVideo: source.model !== 'fsr3-native' && source.fluidVideo === true,
    enabled: typeof source.enabled === 'boolean' ? source.enabled : false,
    preset: VALID_PRESETS.has(source.preset) ? source.preset : 'performance',
    model,
    maxFps: [12, 24, 30, 60].includes(source.maxFps) ? source.maxFps : 24,
    adaptive: typeof source.adaptive === 'boolean' ? source.adaptive : true,
    showHud: typeof source.showHud === 'boolean' ? source.showHud : true,
    spatialDenoise: ['off', 'balanced', 'strong'].includes(source.spatialDenoise) ? source.spatialDenoise : 'balanced',
    fsrQuality: Object.hasOwn(FSR_SCALE_FACTORS, source.fsrQuality) ? source.fsrQuality : 'native-aa',
  };
}

function toModelDimension(value) {
  // Adore's down/up-sampling skip connections require multiples of four.
  // Merely even sizes (e.g. 480x270) fail in the model's Add nodes.
  return Math.max(4, Math.floor(value / 4) * 4);
}

function toModelSize(width, height) {
  const scale = Math.min(1, MAX_MODEL_EDGE / Math.max(width, height),
    Math.sqrt(MAX_MODEL_PIXELS / (width * height)));
  return {
    width: toModelDimension(width * scale),
    height: toModelDimension(height * scale),
  };
}

export function resolveInputSize(sourceWidth, sourceHeight, preset = 'balanced', adaptiveLevel = 0) {
  if (!Number.isFinite(sourceWidth) || !Number.isFinite(sourceHeight) || sourceWidth <= 0 || sourceHeight <= 0) {
    throw new RangeError('source dimensions must be positive finite numbers');
  }

  const safePreset = VALID_PRESETS.has(preset) ? preset : 'balanced';
  const level = Math.max(0, Math.floor(Number.isFinite(adaptiveLevel) ? adaptiveLevel : 0));
  // 4K is a native x2 target. A 720p source must first be resized to 1080p
  // so the model produces 2160p, instead of producing 1440p and enlarging it.
  // For sources between 720p and 1080p, keep the source width at level 0.
  if (safePreset === '4k') {
    const nativeWidth = sourceWidth <= 1280 ? 1920 : Math.min(1920, sourceWidth);
    const eligible = [nativeWidth, 1280, 960, 640, 480, 320, 240, 160]
      .filter((width, index, values) => width > 0 && values.indexOf(width) === index);
    const targetWidth = eligible[Math.min(level, eligible.length - 1)];
    const width = toModelDimension(targetWidth);
    const height = toModelDimension((width / sourceWidth) * sourceHeight);
    return toModelSize(width, height);
  }
  const baseMax = Math.min(PRESET_MAX_WIDTH[safePreset], sourceWidth);
  const eligible = STEP_WIDTHS.filter((width) => width <= baseMax);
  if (!eligible.length) eligible.push(baseMax);
  const targetWidth = Math.min(sourceWidth, eligible[Math.min(level, eligible.length - 1)] ?? baseMax);
  const width = toModelDimension(targetWidth);
  const height = toModelDimension((width / sourceWidth) * sourceHeight);

  return toModelSize(width, height);
}

export const PRESET_WIDTHS = PRESET_MAX_WIDTH;

export function resolveOutputSize(sourceWidth, sourceHeight, preset, inputWidth, inputHeight) {
  const bounds = ({'720p':[1280,720],'1080p':[1920,1080],'1080p-fast':[1920,1080],
    '1440p':[2560,1440],'4k':[3840,2160]})[preset] ?? null;
  if (!bounds) return {width:inputWidth*2,height:inputHeight*2};
  const scale = Math.min(bounds[0]/sourceWidth,bounds[1]/sourceHeight);
  return {width:Math.max(2,Math.round(sourceWidth*scale/2)*2),height:Math.max(2,Math.round(sourceHeight*scale/2)*2)};
}
