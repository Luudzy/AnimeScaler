const fs = require('node:fs');
const path = require('node:path');

function browserRuntime(env = process.env) {
  const { chromium } = require(env.ANIMESCALER_PLAYWRIGHT || 'playwright-core');
  if (env.ANIMESCALER_CHROMIUM) {
    if (!fs.existsSync(env.ANIMESCALER_CHROMIUM)) throw new Error('ANIMESCALER_CHROMIUM does not point to an executable');
    return { chromium, executablePath: env.ANIMESCALER_CHROMIUM };
  }
  const cache = env.LOCALAPPDATA ? path.join(env.LOCALAPPDATA, 'ms-playwright') : '';
  const versions = cache && fs.existsSync(cache) ? fs.readdirSync(cache).filter(name => /^chromium-\d+$/.test(name)).sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1])) : [];
  const candidates = [chromium.executablePath(), ...versions.map(name => path.join(cache, name, 'chrome-win64', 'chrome.exe'))];
  const executablePath = candidates.find(candidate => fs.existsSync(candidate));
  if (!executablePath) throw new Error('Install Chromium with npx playwright-core install chromium or set ANIMESCALER_CHROMIUM');
  return { chromium, executablePath };
}

module.exports = { browserRuntime };
