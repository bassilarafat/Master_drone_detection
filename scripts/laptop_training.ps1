param(
    [ValidateSet('check','seal','start','resume','pause')][string]$Mode='check',
    [ValidateSet('E0','E1')][string]$Arm='E0',
    [ValidateSet(17,42,123)][int]$Seed=17
)
$ErrorActionPreference='Stop'
$laptopRoot=Split-Path $PSScriptRoot -Parent
$laptopPython=Join-Path $laptopRoot '.venv/Scripts/python.exe'
$env:PYTHONPATH=Join-Path $laptopRoot 'src'
$env:YOLO_CONFIG_DIR=Join-Path $laptopRoot '.cache/paper/ultralytics'
$env:PYTHONUNBUFFERED='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
if ($Mode -in @('check','seal','pause')) {
    & $laptopPython -m drone_detection.yolo26_laptop --mode $Mode --arm $Arm --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw 'The requested operation failed; training was not launched.' }
    return
}
$laptopLogs=Join-Path $laptopRoot '04_experiments/controlled_yolo26_v3/logs'
New-Item -ItemType Directory -Path $laptopLogs -Force | Out-Null
$laptopPidFile=Join-Path $laptopLogs 'active.pid'
if (Test-Path -LiteralPath $laptopPidFile) {
    $existingLaptopPid=[int](Get-Content -LiteralPath $laptopPidFile)
    if (Get-Process -Id $existingLaptopPid -ErrorAction SilentlyContinue) { throw 'A laptop training process is already active.' }
}
$laptopStamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$laptopOut=Join-Path $laptopLogs "$($Arm)_seed$($Seed)_$laptopStamp.stdout.log"
$laptopErr=Join-Path $laptopLogs "$($Arm)_seed$($Seed)_$laptopStamp.stderr.log"
$laptopProcess=Start-Process -FilePath $laptopPython `
    -ArgumentList @('-u','-m','drone_detection.yolo26_laptop','--mode',$Mode,'--arm',$Arm,'--seed',"$Seed",'--execute-training') `
    -WorkingDirectory $laptopRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $laptopOut -RedirectStandardError $laptopErr
$laptopProcess.Id | Set-Content -LiteralPath $laptopPidFile
[ordered]@{mode=$Mode;arm=$Arm;seed=$Seed;pid=$laptopProcess.Id;stdout=$laptopOut;stderr=$laptopErr} | ConvertTo-Json
