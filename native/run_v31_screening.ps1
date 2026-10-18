$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$python = 'F:/comfyui-rocm-071926/python_env/python.exe'
$statusPath = Join-Path $projectRoot 'diagnostics/v31-screening-status.json'
$runs = @(@('b','128'), @('a','128'), @('c','128'), @('b','96'))
$completed = @()
function Invoke-Stage($executable, $stageArgs, $logName) {
    $process = Start-Process -FilePath $executable -ArgumentList $stageArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput ('diagnostics/'+$logName+'.log') -RedirectStandardError ('diagnostics/'+$logName+'-errors.log') -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw ('Stage failed: '+$logName+'; see error log') }
}
try {
    if (-not (Test-Path 'native/training-data/v31/codec-full/complete.json')) { throw 'Codec preparation must finish first' }
    foreach ($run in $runs) {
        $name = 'lite-v31-'+$run[0]+'-p'+$run[1]+'-detail-s929'
        $candidate = 'models/'+$name
        @{state='training'; current=$name; completed=$completed} | ConvertTo-Json -Depth 5 | Set-Content $statusPath
        Invoke-Stage $python @('-u','native/train_lite_v31.py','--variant',$run[0],'--patch-lr',$run[1],'--steps','5000','--output',$candidate) $name
        @{state='evaluation'; current=$name; completed=$completed} | ConvertTo-Json -Depth 5 | Set-Content $statusPath
        Invoke-Stage 'C:/Program Files/Python310/python.exe' @('native/compare_lite_v31.py','--candidate',$candidate) ($name+'-visual')
        Invoke-Stage $python @('native/benchmark_lite_v31.py','--candidate',$candidate) ($name+'-benchmark')
        $completed += $name
    }
    @{state='screening-complete'; completed=$completed; release='PENDING visual review and real video validation; no promotion'} | ConvertTo-Json -Depth 5 | Set-Content $statusPath
} catch {
    @{state='failed'; error=$_.Exception.Message; completed=$completed} | ConvertTo-Json -Depth 5 | Set-Content $statusPath
    exit 1
}
