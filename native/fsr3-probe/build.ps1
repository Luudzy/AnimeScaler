param([string]$Sdk = $env:ANIMESCALER_FSR_SDK)
$ErrorActionPreference = 'Stop'
if (-not $Sdk) { throw 'ANIMESCALER_FSR_SDK path is required' }
if (-not (Test-Path -LiteralPath $Sdk -PathType Container)) { throw "FidelityFX SDK directory not found: $Sdk" }
$Sdk = (Resolve-Path -LiteralPath $Sdk).Path
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'MSVC x64 development tools not found' }
$msvc = Get-ChildItem (Join-Path $vs 'VC\Tools\MSVC') -Directory | Sort-Object Name -Descending | Select-Object -First 1
$kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10'
$version = Get-ChildItem (Join-Path $kits 'Include') -Directory | Sort-Object Name -Descending | Select-Object -First 1
$build = Join-Path $PSScriptRoot 'build'
New-Item -ItemType Directory -Path $build -Force | Out-Null
$oldInclude = $env:INCLUDE
$oldLib = $env:LIB
try {
    $env:INCLUDE = @((Join-Path $msvc.FullName 'include'), (Join-Path $version.FullName 'ucrt'), (Join-Path $version.FullName 'shared'), (Join-Path $version.FullName 'um'), (Join-Path $version.FullName 'winrt')) -join ';'
    $env:LIB = @((Join-Path $msvc.FullName 'lib\x64'), (Join-Path $kits "Lib\$($version.Name)\um\x64"), (Join-Path $kits "Lib\$($version.Name)\ucrt\x64")) -join ';'
    $compiler = Join-Path $msvc.FullName 'bin\Hostx64\x64\cl.exe'
    foreach ($target in @('probe', 'worker')) {
        & $compiler /nologo /EHsc /std:c++17 /O2 /D_WINDOWS /DNOMINMAX "/I$Sdk\Kits\FidelityFX\api\include" "/I$Sdk\Kits\FidelityFX\upscalers\include" "/Fe$build\fsr3-$target.exe" "/Fo$build\fsr3-$target.obj" (Join-Path $PSScriptRoot "$target.cpp") /link d3d12.lib dxgi.lib d3dcompiler.lib
        if ($LASTEXITCODE -ne 0) { throw "FSR3 $target build failed: $LASTEXITCODE" }
    }
} finally { $env:INCLUDE = $oldInclude; $env:LIB = $oldLib }
