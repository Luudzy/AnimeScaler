$ErrorActionPreference = "Stop"
Write-Host "AnimeScaler setup" -ForegroundColor Cyan
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Node.js 22+ nao encontrado." }
$nodeVersionText = node -p "process.versions.node"
if ($LASTEXITCODE -ne 0 -or $nodeVersionText -notmatch '^\d+\.\d+\.\d+') { throw "Nao foi possivel validar a versao do Node.js." }
$nodeVersion = [version]$matches[0]
if ($nodeVersion.Major -lt 22) { throw "Node.js 22+ necessario; detectado $nodeVersionText." }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm nao encontrado." }
npm ci
if ($LASTEXITCODE -ne 0) { throw "npm ci falhou." }
npm run setup
if ($LASTEXITCODE -ne 0) { throw "Setup falhou." }
Write-Host "`nPronto. Abra chrome://extensions ou edge://extensions, ative Modo do desenvolvedor e use 'Carregar sem compactacao' nesta pasta." -ForegroundColor Green
