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


class PDEBenchInspectTest(unittest.TestCase):
    def test_inspect_fake_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_dir = root / "fake_burgers"
            ds_dir.mkdir()

            X_train = np.arange(24, dtype=np.float32).reshape(4, 6)
            X_test = np.arange(18, dtype=np.float32).reshape(3, 6)
            y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            y_test = np.array([1.5, 2.5, 3.5], dtype=np.float32)
            shock = np.array([0.45, 0.51, 0.62], dtype=np.float32)
            threshold_distance = np.abs(shock - 0.5)

            dataset = {
                "X_train": torch.from_numpy(X_train),
                "Y_train": torch.from_numpy(y_train),
                "X_test": torch.from_numpy(X_test),
                "Y_test": torch.from_numpy(y_test),
                "args": {
                    "source": "fake-pdebench",
                    "qoi": "shock_jump",
                    "jump_threshold": 0.5,
                    "threshold_distance": threshold_distance,
                    "shock_location": shock,
                    "split_seed": 7,
                    "test_size": 0.2,
                    "x_standardized": True,
                    "y_standardized": True,
                    "y_mean": 1.5,
                    "y_std": 1.0,
                },
            }
            with (ds_dir / "dataset.pkl").open("wb") as f:
                pickle.dump(dataset, f)

            out_json = root / "inspection.json"
            cmd = [
                sys.executable,
                "scripts/data/inspect_pdebench_burgers.py",
                "--dataset-folder",
                str(ds_dir),
                "--out",
                str(out_json),
                "--bins",
                "4",
            ]
            subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

            report = json.loads(out_json.read_text())
            self.assertEqual(report["shapes"]["X_train"], [4, 6])
            self.assertEqual(report["qoi"], "shock_jump")
            self.assertIn("shock_location", report)
            self.assertIn("near_threshold_counts", report)
            self.assertEqual(report["near_threshold_counts"]["test"]["0.05"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
