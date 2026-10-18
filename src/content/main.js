import { AnimeScalerController } from './controller.js';

export async function handleAnimeScalerMessage(controller, message) {
  switch (message.type) {
    case 'ANIMESCALER_EXCLUSIVE_ENTER':
      return controller.exclusiveSource.enter(message.sessionId, message.captureFrame);
    case 'ANIMESCALER_EXCLUSIVE_EXIT':
      return controller.exclusiveSource.exit(message.sessionId);
    case 'ANIMESCALER_PROBE':
      return controller.probe();
    case 'ANIMESCALER_START':
      return controller.start(message.settings);
    case 'ANIMESCALER_START_TAB_CAPTURE':
      return controller.startTabCapture({
        streamId: message.streamId,
        captureFrame: message.captureFrame,
        settings: message.settings,
      });
    case 'ANIMESCALER_CAPTURE_GEOMETRY':
      return controller.getCaptureGeometry(message.captureFrame,message.sessionId);
    case 'ANIMESCALER_PLAYER_COMMAND':
      return controller.playerCommand(message.command,message.sessionId);
    case 'ANIMESCALER_STOP':
      return controller.stop();
    case 'ANIMESCALER_SETTINGS':
      return controller.updateSettings(message.settings);
    case 'ANIMESCALER_STATUS':
      return controller.getStatus();
    default:
      return { error: { code: 'UNKNOWN_MESSAGE', message: message.type } };
  }
}

export function bootstrapAnimeScaler() {
  if (globalThis.__animeScalerController) return globalThis.__animeScalerController;
  const controller = new AnimeScalerController();

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message?.type?.startsWith?.('ANIMESCALER_')) return false;
    handleAnimeScalerMessage(controller, message)
      .then(sendResponse)
      .catch((error) => sendResponse({
        state: 'error',
        error: { code: 'MESSAGE_ERROR', message: error?.message ?? String(error) },
      }));
    return true;
  });

  globalThis.__animeScalerController = controller;
  return controller;
}
