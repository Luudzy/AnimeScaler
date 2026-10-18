import { FsrEngine } from '../src/engine/fsr-engine.js';

window.evaluateSpatialSequences = async manifestPaths => {
  const report = { state: 'running', scope: 'Actual WebGPU spatial outputs read back for paired offline quality; no player throughput claim', sequences: [], errors: [] };
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
  for (const manifestPath of manifestPaths) {
    const metadata = await fetch(manifestPath).then(response => { if (!response.ok) throw new Error('Missing manifest'); return response.json(); });
    const base = manifestPath.slice(0, manifestPath.lastIndexOf('/') + 1);
    const [width, height] = metadata.hrSize;
    const source = new OffscreenCanvas(...metadata.lrSize), sourceContext = source.getContext('2d', { alpha: false });
    const targetCanvas = new OffscreenCanvas(width, height), targetContext = targetCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    const modes = [];
    for (const model of ['fsr1-spatial', 'asr-anime-spatial']) {
      const canvas = document.createElement('canvas'); document.body.append(canvas);
      const engine = new FsrEngine(); engine.diagnostics = { event() {}, frame() {}, flush() {} };
      await engine.initialize({ canvas, video: source, settings: { model, preset: '720p', fsrQuality: 'native-aa', spatialDenoise: 'balanced' } });
      // Force the manifest's exact paired x2 output; the product preset otherwise
      // fits unusual aspect ratios into a 720p display box.
      await engine.renderer.configure(...metadata.lrSize, width, height, ...metadata.lrSize);
      const renderer=engine.renderer, device=engine.device;
      renderer.context.configure({device,format:renderer.format,alphaMode:'opaque',usage:GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC});
      const rowBytes=Math.ceil(width*4/256)*256;
      const gpuRead=device.createBuffer({size:rowBytes*height,usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      const render=renderer.render.bind(renderer);
      // Snapshot in the same task as render submission. A displayed WebGPU
      // canvas may discard its current texture before an awaited drawImage.
      renderer.render=(...args)=>{
        render(...args);
        const copy=device.createCommandEncoder();
        copy.copyTextureToBuffer({texture:renderer.context.getCurrentTexture()},{buffer:gpuRead,bytesPerRow:rowBytes},[width,height]);
        device.queue.submit([copy.finish()]);
      };
      modes.push({ model, engine, canvas, gpuRead, rowBytes, rows: [], previous: null });
    }
    let previousReference = null;
    try {
      for (const frame of metadata.frames) {
        const load = async name => { const response = await fetch(base + name); if (!response.ok) throw new Error('Missing paired frame'); return createImageBitmap(await response.blob()); };
        const [low, high] = await Promise.all([load(frame.lr), load(frame.hr)]);
        sourceContext.drawImage(low, 0, 0); targetContext.drawImage(high, 0, 0); low.close(); high.close();
        source.currentTime = frame.index / (Number(metadata.fps.split('/')[0]) / Number(metadata.fps.split('/')[1]));
        const reference = targetContext.getImageData(0, 0, width, height).data;
        for (const mode of modes) {
          await mode.engine.processFrame(source);
          await mode.gpuRead.mapAsync(GPUMapMode.READ);
          const bytes=new Uint8Array(mode.gpuRead.getMappedRange()), output=new Uint8ClampedArray(width*height*4);
          const bgra=mode.engine.renderer.format.startsWith('bgra');
          for(let y=0;y<height;y++)for(let x=0;x<width;x++){
            const src=y*mode.rowBytes+x*4, dst=(y*width+x)*4;
            output[dst]=bytes[src+(bgra?2:0)];output[dst+1]=bytes[src+1];output[dst+2]=bytes[src+(bgra?0:2)];output[dst+3]=255;
          }
          mode.gpuRead.unmap();
          let mae = 0, mse = 0, temporal = 0, brightness = 0, referenceBrightness = 0;
          for (let i = 0; i < output.length; i += 4) for (let c = 0; c < 3; c++) {
            const delta = output[i + c] - reference[i + c]; mae += Math.abs(delta); mse += delta * delta;
            brightness += output[i + c]; referenceBrightness += reference[i + c];
            if (mode.previous) temporal += Math.abs((output[i + c] - mode.previous[i + c]) - (reference[i + c] - previousReference[i + c]));
          }
          const samples = width * height * 3;
          if (referenceBrightness / samples > 10 && brightness / samples < 1) throw new Error(`${mode.model}: unexpected black output`);
          mode.rows.push({ mae: mae / samples / 255, psnr: 10 * Math.log10(1 / Math.max(1e-12, mse / samples / 255 ** 2)), temporal: mode.previous ? temporal / samples / 255 : null });
          mode.previous = output;
        }
        previousReference = reference;
      }
      report.sequences.push({ manifest: manifestPath, frames: metadata.frames.length, sourceSha256: metadata.sourceSha256,
        models: Object.fromEntries(modes.map(mode => [mode.model, { mae: mean(mode.rows.map(row => row.mae)), psnr: mean(mode.rows.map(row => row.psnr)), referenceCompensatedTemporalMae: mean(mode.rows.map(row => row.temporal).filter(value => value !== null)) }])) });
      document.querySelector('#report').textContent = JSON.stringify(report, null, 2);
    } finally { for (const mode of modes) { mode.gpuRead.destroy();await mode.engine.destroy(); mode.canvas.remove(); } }
  }
  report.state = 'complete'; return report;
};
