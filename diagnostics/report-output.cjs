const fs = require('node:fs');
const path = require('node:path');

function writeDiagnosticReport(defaultPath, report) {
  const filename = process.env.ANIMESCALER_REPORT_DIR
    ? path.join(process.env.ANIMESCALER_REPORT_DIR, path.basename(defaultPath)) : defaultPath;
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  fs.writeFileSync(filename, JSON.stringify(report, null, 2), { flag: 'wx' });
  return filename;
}

module.exports = { writeDiagnosticReport };
