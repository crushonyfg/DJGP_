"""Quick Φᵀr alignment diagnostic on D3 (no RMSE focus).

```powershell
python experiments/synthetic/diagnose_phi_alignment.py ^
    --max_test 40 --num_steps 120 --variants D3,D3_cov1
```
"""

from experiments.synthetic.test_remedy_d_structured_vi import main

if __name__ == "__main__":
    import sys

    if "--variants" not in sys.argv:
        sys.argv.extend(["--variants", "D3,D3_cov0.1,D3_cov1,D3_ridge0.1"])
    main()
