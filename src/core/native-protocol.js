const SETTINGS_KEYS = new Set(['backend', 'model', 'preset', 'maxFps', 'adaptive', 'showHud', 'exclusiveViewer',
  'fluidVideo', 'enabled', 'spatialDenoise', 'fsrQuality']);

const MESSAGE_FIELDS = {
  START: ['backend', 'model', 'settings', 'captureMode', 'sourceWidth', 'sourceHeight'],
  FRAME_IN: ['sequence', 'mediaTime', 'width', 'height', 'mime', 'dataUrl'],
  CONFIGURE: ['revision', 'settings', 'adaptiveLevel', 'sourceWidth', 'sourceHeight'],
  STOP: ['reason'],
};

export function isNativeMessageShape(message) {
  if (!message || typeof message !== 'object' || Array.isArray(message)) return false;
  const fields = MESSAGE_FIELDS[message.type];
  if (!fields) return false;
  const allowed = new Set(['protocol', 'type', 'sessionId', ...fields]);
  if (Object.keys(message).some(key => !allowed.has(key))) return false;
  const settings = message.settings;
  return settings == null || (typeof settings === 'object' && !Array.isArray(settings) &&
    Object.keys(settings).every(key => SETTINGS_KEYS.has(key)));
}
