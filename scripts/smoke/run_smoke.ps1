$ErrorActionPreference = "Stop"

conda run -n jumpGP python -m unittest tests.test_imports
conda run -n jumpGP python -m unittest tests.test_smoke_jumpgp
conda run -n jumpGP python -m unittest tests.test_smoke_djgp
