/** Choose the original-player activation action without browser or backend dependencies. */
export function resolveActivationRoute(settings, status) {
  if (status?.state === 'running' || status?.state === 'starting') return 'stop';
  return ['rocm', 'tensorrt', 'opencl', 'd3d12'].includes(settings?.backend) ? 'viewer' : 'inline';
}
