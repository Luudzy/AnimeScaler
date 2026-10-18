export class AnimeScalerHud {
  constructor() {
    this.video = null;
    this.visible = true;
    this.root = document.createElement('div');
    this.root.dataset.animescaler = 'hud';
    Object.assign(this.root.style, {
      position: 'fixed',
      display: 'none',
      minWidth: '178px',
      maxWidth: '260px',
      padding: '9px 11px',
      borderRadius: '9px',
      background: 'rgba(8, 10, 14, 0.82)',
      color: '#f6f7fb',
      font: '600 11px/1.45 system-ui, -apple-system, Segoe UI, sans-serif',
      letterSpacing: '.01em',
      pointerEvents: 'none',
      zIndex: '2147483001',
      boxShadow: '0 5px 22px rgba(0,0,0,.35)',
      backdropFilter: 'blur(10px)',
      whiteSpace: 'pre-line',
    });
    document.documentElement.appendChild(this.root);
  }

  attach(video) {
    this.video = video;
    this.sync();
  }

  sync() {
    if (!this.video?.isConnected) return;
    const rect = this.video.getBoundingClientRect();
    this.root.style.left = `${Math.max(8, rect.right - 270)}px`;
    this.root.style.top = `${Math.max(8, rect.top + 10)}px`;
  }

  setVisible(visible) {
    this.visible = Boolean(visible);
    this.root.style.display = this.visible ? 'block' : 'none';
  }

  update(status) {
    if (!this.visible) return;
    const lines = ['AnimeScaler'];
    if (status.state === 'running') {
      const backend = status.stats?.backend === 'webgpu' ? 'WebGPU' : String(status.stats?.backend ?? 'WebGPU').toUpperCase();
      lines.push(`${status.stats?.modelLabel || "Modelo"} ${String(status.stats?.modelPrecision ?? '').toUpperCase()} · ${backend}`);
      if (status.stats) {
        lines.push(`${status.stats.sourceWidth}×${status.stats.sourceHeight} → ${status.stats.outputWidth}×${status.stats.outputHeight}`);
        if (status.stats.fsrQuality) lines.push(`FSR ${status.stats.fsrQuality} · ruído ${status.stats.spatialDenoise}`);
        lines.push(`Frame ${status.stats.totalMs?.toFixed?.(1) ?? '-'} ms · ${status.processedFps?.toFixed?.(1) ?? '-'} fps processados (inclui quadros reutilizados)`);
      } else {
        lines.push('Inicializando…');
      }
    } else if (status.state === 'error') {
      lines.push(`Erro: ${status.error?.message ?? 'falha desconhecida'}`);
    } else {
      lines.push(status.state === 'starting' ? 'Inicializando…' : 'Parado');
    }
    this.root.textContent = lines.join('\n');
    this.sync();
  }

  destroy() {
    this.video = null;
    this.root.remove();
  }
}
