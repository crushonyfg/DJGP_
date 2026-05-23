# Profile W exploration vs baselines: 5 seeds, max_test=200, 600 steps.
# Logs: experiments/synthetic/explore_w_5seeds_600/seed{0..4}.log

$ErrorActionPreference = "Stop"
# Script lives in experiments/synthetic -> repo root is two levels up
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$Variants = @(
    "D3_snfix0.3_exploreW_inner20_lrW1_decay",
    "D3_snfix0.3_exploreW_inner20_lrW0.5_decay",
    "D3_snfix0.3_exploreW_inner20_lrW0.5_trust",
    "D3_snfix0.3_exploreW_inner20_lrW0.5_acceptCEM",
    "D3_snfix0.3_warm120_blend0.5_slowW5x",
    "D3_snfix0.3_warm120_blend_to0.5_freeze_W"
) -join ","

$OutRoot = Join-Path $Root "experiments\synthetic\explore_w_5seeds_600"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$Py = Join-Path $Root "experiments\synthetic\test_remedy_d_structured_vi.py"
$ArgsBase = @(
    $Py,
    "--setting", "lh_q5_train500",
    "--variants", $Variants,
    "--max_test", "200",
    "--num_steps", "600",
    "--eval_every", "150",
    "--lr", "0.01"
)

for ($s = 0; $s -lt 5; $s++) {
    $log = Join-Path $OutRoot "seed$s.log"
    $err = Join-Path $OutRoot "seed$s.err"
    $outDir = Join-Path $OutRoot "metrics_seed$s"
    $allArgs = $ArgsBase + @("--seed", "$s", "--out_dir", $outDir)
    Write-Host "Starting seed $s -> $log"
    Start-Process -FilePath "python" -ArgumentList $allArgs `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $log `
        -RedirectStandardError $err `
        -WindowStyle Hidden
}

Write-Host "Launched 5 jobs. Tail logs: Get-Content $OutRoot\seed0.log -Tail 40 -Wait"
