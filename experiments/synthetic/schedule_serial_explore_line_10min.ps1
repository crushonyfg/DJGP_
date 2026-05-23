# Wait 10 minutes, then run serial 5-seed explore+line experiment (single-GPU friendly).
# Intended launch: Start-Process powershell -WindowStyle Hidden -File this_script.ps1
#
# Log: experiments/synthetic/explore_line_serial_5seeds_600/scheduler.log

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutRoot = Join-Path $Root "experiments\synthetic\explore_line_serial_5seeds_600"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$SchedLog = Join-Path $OutRoot "scheduler.log"

function Write-Sched($t) {
    "$(Get-Date -Format o)  $t" | Out-File -FilePath $SchedLog -Append -Encoding utf8
}

Write-Sched "Scheduler started. Sleeping 600 seconds before serial run..."
Start-Sleep -Seconds 600
Write-Sched "Sleep finished. Launching run_explore_line_serial_5seeds.ps1"

$RunScript = Join-Path $PSScriptRoot "run_explore_line_serial_5seeds.ps1"
try {
    & $RunScript *>&1 | Tee-Object -FilePath $SchedLog -Append
    Write-Sched "Serial script finished exit=$LASTEXITCODE"
} catch {
    Write-Sched "ERROR: $($_.Exception.Message)"
    throw
}
