# Serial 5-seed run: ONE python process per seed (variants run in-process order).
# GPU: only one training job at a time. Logs per seed under explore_line_serial_5seeds_600/.
#
# Usage (from repo root or any cwd):
#   powershell -NoProfile -ExecutionPolicy Bypass -File experiments/synthetic/run_explore_line_serial_5seeds.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$Variants = @(
    "D3_snfix0.3_exploreW_inner20_lrW1_decay",
    "D3_snfix0.3_exploreW_inner20_lrW0.5_decay",
    "D3_snfix0.3_exploreW_inner20_lrW0.5_trust",
    "D3_snfix0.3_exploreW_inner20_lrW0.5_acceptCEM",
    "D3_snfix0.3_lineW_elbo_inner20",
    "D3_snfix0.3_lineW_elbo_inner20_decay",
    "D3_snfix0.3_lineW_holdout_inner20",
    "D3_snfix0.3_lineW_cem_accept20",
    "D3_snfix0.3_lineW_elbo_trust_strong",
    "D3_snfix0.3_warm120_blend0.5_slowW5x",
    "D3_snfix0.3_warm120_blend_to0.5_freeze_W"
) -join ","

$OutRoot = Join-Path $Root "experiments\synthetic\explore_line_serial_5seeds_600"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$Py = Join-Path $Root "experiments\synthetic\test_remedy_d_structured_vi.py"
$MasterLog = Join-Path $OutRoot "serial_master.log"
"========================================`n$(Get-Date -Format o)  SERIAL 5-seed explore+line+baselines`n========================================" | Out-File -FilePath $MasterLog -Encoding utf8

for ($s = 0; $s -lt 5; $s++) {
    $log = Join-Path $OutRoot "seed$s.log"
    $outDir = Join-Path $OutRoot "metrics_seed$s"
    $msg = "`n$(Get-Date -Format o)  >>> START seed=$s  out_dir=$outDir"
    $msg | Tee-Object -FilePath $MasterLog -Append
    $msg | Out-File -FilePath $log -Append -Encoding utf8

    $pyArgs = @(
        "--setting", "lh_q5_train500",
        "--variants", $Variants,
        "--max_test", "200",
        "--num_steps", "600",
        "--eval_every", "150",
        "--lr", "0.01",
        "--seed", "$s",
        "--out_dir", $outDir,
        "--eval_sampleW_cem",
        "--MC_num", "5"
    )
    & python $Py @pyArgs *>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    "$(Get-Date -Format o)  >>> END seed=$s  exit=$code" | Tee-Object -FilePath $MasterLog -Append
    if ($code -ne 0) {
        "$(Get-Date -Format o)  ABORT: seed $s failed with exit $code" | Tee-Object -FilePath $MasterLog -Append
        exit $code
    }
    try { nvidia-smi | Out-File -FilePath (Join-Path $OutRoot "nvidia_after_seed$s.txt") -Encoding utf8 } catch {}
}

"$(Get-Date -Format o)  ALL 5 seeds completed." | Tee-Object -FilePath $MasterLog -Append
Write-Host "Done. Logs: $OutRoot"
