# UCI 4 methods x 3 datasets x 10 seeds, full test set, empirical std preset.
# Log: uci4_empstd_10seeds_fulltest_<timestamp>.log
# Out: uci4_methods_empstd_10seeds_fulltest.json / .csv
#
# Monitor:  Get-Content <log> -Wait -Tail 30
# Or:        Get-Content <log> -Tail 5

$ErrorActionPreference = "Stop"
# PSScriptRoot = .../DJGP/experiments/uci/inductive_projection_compare
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not (Test-Path (Join-Path $root "djgp"))) { throw "djgp not found under $root" }
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $PSScriptRoot "uci4_empstd_10seeds_fulltest_$ts.log"
$state = Join-Path $PSScriptRoot "RUN_10seeds_state.txt"
$relOut = "experiments/uci/inductive_projection_compare/uci4_methods_empstd_10seeds_fulltest"
$cmd = @"
cd /d `"$root`" && python -u experiments\uci\compare_inductive_projection.py --num_exp 10 --datasets `"Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction`" --max_test_anchors 999999 --max_train_anchors 128 --n 35 --m2 40 --lmjgp_steps 200 --n_outer 5 --n_m_steps 80 --MC_num 3 --lmjgp_supervised_train_mc 2 --lmjgp_supervised_grad_check 0 --scaling_preset uci_empirical --out_dir experiments/uci/inductive_projection_compare --out_json ${relOut}.json --out_csv ${relOut}.csv > `"$log`" 2>&1
"@
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Hidden
@"
Started: $(Get-Date -Format o)
Log: $log
Output: $root\$($relOut -replace '/','\').json
Command: $cmd
"@ | Set-Content -Path $state -Encoding utf8
Write-Host "Background job started. State file: $state"
Write-Host "Log: $log"
