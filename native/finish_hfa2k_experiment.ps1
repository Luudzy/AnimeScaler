param([Parameter(Mandatory=$true)][int]$TrainingProcessId)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$candidate = Join-Path $projectRoot 'models/lite-v3-24-hfa2k-experiment'
$statusFile = Join-Path $candidate 'postprocessing.json'
$stages = @(
    @('C:/Program Files/Python310/python.exe', 'native/evaluate_lite_perceptual.py', 'perceptual'),
    @('C:/Program Files/Python310/python.exe', 'native/generate_lite_comparison.py', 'comparisons'),
    @('F:/comfyui-rocm-071926/python_env/python.exe', 'native/evaluate_lite_temporal.py', 'temporal'),
    @('F:/comfyui-rocm-071926/python_env/python.exe', 'native/benchmark_lite_deployment.py', 'performance')
)
$results = @()
try {
    if (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
        Wait-Process -Id $TrainingProcessId
    }
    if (-not (Test-Path -LiteralPath (Join-Path $candidate 'report.json'))) {
        throw 'Training ended without its final report; postprocessing not run.'
    }
    foreach ($stage in $stages) {
        $log = Join-Path $projectRoot ('diagnostics/hfa2k-' + $stage[2] + '.log')
        $evaluation = Start-Process -FilePath $stage[0] -ArgumentList @($stage[1], '--candidate', ('"' + $candidate + '"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.stderr') -Wait -PassThru
        $results += @{stage=$stage[2]; exitCode=$evaluation.ExitCode; log=$log}
        @{state='running'; results=$results} | ConvertTo-Json -Depth 5 | Set-Content $statusFile
        if ($evaluation.ExitCode -ne 0) { throw ('Evaluation failed: ' + $stage[2]) }
    }
    @{state='complete'; results=$results; promoted=$false} | ConvertTo-Json -Depth 5 | Set-Content $statusFile
} catch {
    @{state='failed'; error=$_.Exception.Message; results=$results; promoted=$false} | ConvertTo-Json -Depth 5 | Set-Content $statusFile
    exit 1
}
