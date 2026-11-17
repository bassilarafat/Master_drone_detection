param(
    [ValidateSet('E0','E1')][string]$Arm = 'E0',
    [ValidateSet(17,42,123)][int]$Seed = 17
)
$ErrorActionPreference = 'Stop'
$controlledRoot = Split-Path $PSScriptRoot -Parent
$controlledPython = Join-Path $controlledRoot '.venv/Scripts/python.exe'
$controlledLogs = Join-Path $controlledRoot '04_experiments/controlled_yolo26_v2/logs'
New-Item -ItemType Directory -Path $controlledLogs -Force | Out-Null
$controlledPidFile = Join-Path $controlledLogs 'active_training.pid'
if (Test-Path -LiteralPath $controlledPidFile) {
    $existingTrainingPid = [int](Get-Content -LiteralPath $controlledPidFile)
    if (Get-Process -Id $existingTrainingPid -ErrorAction SilentlyContinue) {
        throw "A controlled training launcher is already active: $existingTrainingPid"
    }
}
$env:PYTHONPATH = Join-Path $controlledRoot 'src'
$env:PYTHONUNBUFFERED = '1'
$env:YOLO_CONFIG_DIR = Join-Path $controlledRoot '.cache/paper/ultralytics'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$controlledStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$controlledOut = Join-Path $controlledLogs "$($Arm)_seed$($Seed)_$controlledStamp.stdout.log"
$controlledErr = Join-Path $controlledLogs "$($Arm)_seed$($Seed)_$controlledStamp.stderr.log"
$controlledProcess = Start-Process -FilePath $controlledPython `
    -ArgumentList @('-u','-m','drone_detection.yolo26_controlled','--arm',$Arm,'--seed',"$Seed",'--execute-training') `
    -WorkingDirectory $controlledRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $controlledOut -RedirectStandardError $controlledErr
$controlledProcess.Id | Set-Content -LiteralPath $controlledPidFile
[ordered]@{ arm=$Arm; seed=$Seed; pid=$controlledProcess.Id; stdout=$controlledOut; stderr=$controlledErr } | ConvertTo-Json
