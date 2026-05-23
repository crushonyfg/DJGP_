# compare_3way_10seeds — 输出与断点续跑

- **标准输出**（与 `compare_3way_10seeds.py` 的 `--out_csv` / `--out_json` 一致）  
  - `results.csv`：长表，便于 pandas 画图。  
  - `results.json`：嵌套结构，含 CEM-EM 的 `diagnostics`。  
  - 运行过程中：`results.json.partial` 与 `results.partial.csv`（每跑完一个 seed 更新一次）。

- **断点续跑**（不重复已完成的 dataset×seed）  
  与**第一次**使用**相同**的 `--out_json` / `--out_csv` 路径，并加上 `--resume`：

  ```bash
  conda activate jumpGP
  cd <DJGP 仓库根目录>

  python experiments/uci/compare_3way_10seeds.py --resume ^
      --num_exp 10 ^
      --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
      --max_anchors 200 ^
      --out_csv experiments/uci/compare_3way_10seeds/results.csv ^
      --out_json experiments/uci/compare_3way_10seeds/results.json
  ```

  脚本会优先读取 `results.json.partial`（若存在）；否则会读已有的 `results.json`。  
  已完成的 `(数据集, seed)` 会打印 `[resume: skip, already complete]` 并跳过。  
  **未完成**的 seed（例如中途崩溃）会从 checkpoint 里删掉并重新算一整轮。

- **建议**：续跑时不要再用 Excel 打开 `results.partial.csv`，以免 Windows 锁文件导致 checkpoint 写入失败（现已改为先写 `.tmp` 再替换，但仍可能被占用）。
