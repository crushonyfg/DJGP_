from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class PDEBenchRunnerTest(unittest.TestCase):
    def test_runner_on_fake_dataset_xgb_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_dir = root / "fake_burgers"
            ds_dir.mkdir()
            rng = np.random.default_rng(123)
            X_train = rng.normal(size=(32, 8)).astype(np.float32)
            X_test = rng.normal(size=(12, 8)).astype(np.float32)
            y_train = (X_train[:, 0] + (X_train[:, 1] > 0).astype(np.float32)).astype(np.float32)
            y_test = (X_test[:, 0] + (X_test[:, 1] > 0).astype(np.float32)).astype(np.float32)
            shock_test = np.linspace(0.35, 0.65, len(X_test)).astype(np.float32)
            dataset = {
                "X_train": torch.from_numpy(X_train),
                "Y_train": torch.from_numpy(y_train),
                "X_test": torch.from_numpy(X_test),
                "Y_test": torch.from_numpy(y_test),
                "args": {
                    "source": "fake-pdebench",
                    "jump_threshold": 0.5,
                    "near_threshold_delta": 0.08,
                    "shock_location": shock_test,
                },
            }
            with (ds_dir / "dataset.pkl").open("wb") as f:
                pickle.dump(dataset, f)

            out_json = root / "result.json"
            cmd = [
                sys.executable,
                "-m",
                "experiments.pdebench.compare_pdebench_burgers",
                "--dataset-folder",
                str(ds_dir),
                "--num_exp",
                "1",
                "--max_anchors",
                "10",
                "--include_xgb",
                "--no-include_dkl",
                "--no-include_lmjgp",
                "--no-include_cemem",
                "--out",
                str(root / "result.pkl"),
                "--out_json",
                str(out_json),
            ]
            subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
            result = json.loads(out_json.read_text())
            metrics = result["runs"]["0"]["metrics"]["xgb_bootstrap"]
            self.assertIn("all", metrics)
            self.assertIn("near_threshold", metrics)
            self.assertIn("far_from_threshold", metrics)
            self.assertEqual(len(metrics["all"]), 7)


if __name__ == "__main__":
    unittest.main()
