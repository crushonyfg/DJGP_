$ErrorActionPreference = "Continue"
$root = "experiments/uci/cem_em_park_grid"
New-Item -ItemType Directory -Force -Path $root | Out-Null

$alphas    = @(1, 3, 10)
$lambdas   = @(0.0001, 0.001, 0.01)

foreach ($a in $alphas) {
    foreach ($l in $lambdas) {
        $tag = "a${a}_l${l}"
        $tag = $tag.Replace('.', 'p')
        $log = "$root/$tag.log"
        $csv = "$root/$tag.csv"
        $jsn = "$root/$tag.json"
        Write-Host "==> running $tag"
        python experiments/uci/compare_cem_em_projection.py `
            --num_exp 1 --datasets "Parkinsons Telemonitoring" `
            --n 35 --m1 2 --max_anchors 128 `
            --n_outer 5 --n_m_steps 80 --lr 0.01 `
            --lambda_gate 0.1 --lambda_smooth $l --lambda_scale 0 `
            --coeff_scale $a --row_normalize_W 1 `
            --normalize_smoothness 1 --track_best_W 1 `
            --include_free 0 --include_lmjgp 0 `
            --out $csv --out_json $jsn 2>&1 | Tee-Object -FilePath $log | Out-Null
        Write-Host "    saved $log"
    }
}
Write-Host "GRID DONE."
