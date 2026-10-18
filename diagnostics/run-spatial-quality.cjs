const fs = require('node:fs');
const crypto = require('node:crypto');
const { browserRuntime } = require('./browser-runtime.cjs');
const { chromium, executablePath } = browserRuntime();
const roots = ['stability-hero-20261011', 'stability-namakura-20261011'];
const manifests = roots.flatMap(root => fs.readdirSync('diagnostics/video-sequences/' + root).filter(name => fs.existsSync(`diagnostics/video-sequences/${root}/${name}/complete.json`)).map(name => `/diagnostics/video-sequences/${root}/${name}/complete.json`));
const output = process.env.ANIMESCALER_SPATIAL_QUALITY_OUTPUT || 'diagnostics/stability-20261011/spatial-quality.json';
if (fs.existsSync(output)) throw new Error('Choose a new quality report path; previous results are preserved');
const hash = filename => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
for (const manifestPath of manifests) {
  const filename = manifestPath.slice(1), manifest = JSON.parse(fs.readFileSync(filename, 'utf8'));
  const base = filename.slice(0, filename.lastIndexOf('/') + 1);
  if (manifest.frames.length !== 48) throw new Error('Incomplete quality sequence');
  for (const frame of manifest.frames) {
    for (const side of ['lr', 'hr']) if (hash(base + frame[side]) !== frame[side + 'Sha256']) throw new Error('Paired frame hash mismatch');
  }
}

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath, args: ['--enable-unsafe-webgpu'] });
  try {
    const page = await browser.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/diagnostics/stability-quality.html');
    const result = await page.evaluate(paths => window.evaluateSpatialSequences(paths), manifests);
    result.browserErrors = errors;
    result.manifestHashes = Object.fromEntries(manifests.map(name => [name, hash(name.slice(1))]));
    result.precision = 'Browser RGB8 readback; CPU model report uses FP32. Cross-backend metrics are not bit-identical.';
    fs.writeFileSync(output, JSON.stringify(result, null, 2), { flag: 'wx' });
    console.log(JSON.stringify({ state: result.state, sequences: result.sequences.length, errors }));
    if (errors.length || result.state !== 'complete') throw new Error('Spatial quality validation failed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
