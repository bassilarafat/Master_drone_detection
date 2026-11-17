# Start the three controlled Clean_v2 runs sequentially in a hidden process.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
$logRoot = Join-Path $projectRoot '04_experiments/clean_v2_standard_protocol/logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$pidFile = Join-Path $logRoot 'training.pid'
if (Test-Path -LiteralPath $pidFile) {
    $trainingPid = [int](Get-Content -LiteralPath $pidFile)
    if (Get-Process -Id $trainingPid -ErrorAction SilentlyContinue) {
        throw "Controlled seed training process $trainingPid is already running."
    }
}
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot '.cache/ultralytics-local'
$env:PYTHONUNBUFFERED = '1'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "$stamp.stdout.log"
$stderr = Join-Path $logRoot "$stamp.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList @('-u', 'scripts/train_clean_v2_standard_seeds.py') `
  -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $pidFile
Write-Output "Training PID: $($process.Id)"
Write-Output "Output log: $stdout"
Write-Output "Progress/error log: $stderr"
