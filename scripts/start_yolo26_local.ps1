# Run from PowerShell. Training continues in a hidden process; logs stay local.
param([string]$ResumeCheckpoint)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
$logRoot = Join-Path $projectRoot '04_experiments/local_clean_v2/logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$pidFile = Join-Path $logRoot 'training.pid'
if (Test-Path -LiteralPath $pidFile) {
    $trainingPid = [int](Get-Content -LiteralPath $pidFile)
    if (Get-Process -Id $trainingPid -ErrorAction SilentlyContinue) {
        throw "Training process $trainingPid is already running."
    }
}
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot '.cache/ultralytics-local'
$env:PYTHONUNBUFFERED = '1'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "$stamp.stdout.log"
$stderr = Join-Path $logRoot "$stamp.stderr.log"
$trainingArgs = @(
    '-u', '-m', 'drone_detection', 'train', '--config',
    'configs/experiments/yolo26s_clean_v2_local.yaml'
)
if ($ResumeCheckpoint) {
    $trainingArgs = @('-u', 'scripts/resume_yolo26_local.py', '--checkpoint', $ResumeCheckpoint)
}
$process = Start-Process -FilePath $python -ArgumentList $trainingArgs `
  -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $pidFile
Write-Output "Training PID: $($process.Id)"
Write-Output "Output log: $stdout"
Write-Output "Progress/error log: $stderr"
