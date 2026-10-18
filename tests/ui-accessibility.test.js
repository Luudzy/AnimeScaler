import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

function cssColor(source, name) {
  const match = source.match(new RegExp(`${name}:\\s*(#[0-9a-f]{6})`, 'i'));
  assert.ok(match, `missing CSS color ${name}`);
  return match[1];
}

function luminance(hex) {
  const channels = hex.slice(1).match(/.{2}/g).map(value => parseInt(value, 16) / 255)
    .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrast(foreground, background) {
  const values = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
  return (values[0] + 0.05) / (values[1] + 0.05);
}

test('popup subtle text meets AA contrast on primary and card backgrounds', async () => {
  const css = await readFile(new URL('../popup/popup.css', import.meta.url), 'utf8');
  const foreground = cssColor(css, '--text-subtle');
  for (const backgroundName of ['--bg-primary', '--bg-card']) {
    const background = cssColor(css, backgroundName);
    assert.ok(contrast(foreground, background) >= 4.5,
      `${foreground} on ${background} must reach 4.5:1`);
  }
});

test('popup and viewer keep visible keyboard focus styles', async () => {
  const popup = await readFile(new URL('../popup/popup.css', import.meta.url), 'utf8');
  const viewer = await readFile(new URL('../viewer/viewer.css', import.meta.url), 'utf8');
  assert.match(popup, /:focus-visible\s*,/);
  assert.match(viewer, /button:focus-visible\s*\{/);
});

test('popup and viewer use native tab order without positive tabindex overrides', async () => {
  for (const path of ['../popup/popup.html', '../viewer/viewer.html']) {
    const html = await readFile(new URL(path, import.meta.url), 'utf8');
    assert.doesNotMatch(html, /tabindex\s*=\s*["'](?:[1-9]\d*)["']/i);
  }
});
