# Launches 3-way UCI benchmark with --resume; logs to run_resume.log in this folder.
# Repo layout: .../DJGP/experiments/uci/compare_3way_10seeds/  (this script + outputs)
#              .../DJGP/experiments/uci/compare_3way_10seeds.py (entrypoint)
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
# DJGP repo root: from experiments/uci/compare_3way_10seeds go up 3 levels
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $root
$pyExe = Join-Path $env:LOCALAPPDATA "miniconda3\envs\jumpGP\python.exe"
if (-not (Test-Path $pyExe)) {
    $pyExe = Join-Path $env:USERPROFILE "miniconda3\envs\jumpGP\python.exe"
}
$main = Join-Path $root "experiments\uci\compare_3way_10seeds.py"
$log = Join-Path $PSScriptRoot "run_resume.log"
& $pyExe -u $main `
    --resume `
    --num_exp 10 `
    --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" `
    --max_anchors 200 `
    --out_csv (Join-Path $PSScriptRoot "results.csv") `
    --out_json (Join-Path $PSScriptRoot "results.json") `
    *>> $log
