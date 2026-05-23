from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.pdebench import (
    downsample_space,
    make_burgers_surrogate,
    save_dataset,
    split_and_standardize,
    standardize_burgers_tensor,
)


class PDEBenchSurrogateTest(unittest.TestCase):
    def test_burgers_shock_jump_surrogate(self):
        U = np.zeros((6, 4, 8, 1), dtype=np.float32)
        U[:, -1, 4:, 0] = 1.0
        U[3:, -1, 6:, 0] = 3.0

        U_std = downsample_space(standardize_burgers_tensor(U), D=4)
        surrogate = make_burgers_surrogate(U_std, qoi="shock_jump", noise_std=0.0)

        self.assertEqual(surrogate.X.shape, (6, 4))
        self.assertEqual(surrogate.y.shape, (6,))
        self.assertEqual(surrogate.meta["D"], 4)
        self.assertEqual(surrogate.meta["qoi"], "shock_jump")
        self.assertIn("shock_location", surrogate.meta)
        self.assertIn("threshold_distance", surrogate.meta)
        self.assertIn("near_threshold_mask", surrogate.meta)
        self.assertEqual(len(surrogate.meta["shock_location"]), 6)
        self.assertEqual(len(surrogate.meta["threshold_distance"]), 6)
        self.assertEqual(len(surrogate.meta["near_threshold_mask"]), 6)
        self.assertEqual(surrogate.meta["initial_time_index"], 0)
        self.assertEqual(surrogate.meta["final_time_index"], -1)
        self.assertEqual(surrogate.meta["threshold"], 0.5)
        self.assertTrue(np.all(np.isfinite(surrogate.y)))

    def test_save_djgp_dataset(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(20, 5)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32) + X[:, 1]
        dataset = split_and_standardize(X, y, seed=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_dataset(Path(tmp), dataset, {"source": "unit-test"})
            self.assertTrue(path.exists())
            self.assertIn("raw_y_mean", dataset["args"])
            self.assertIn("raw_y_std", dataset["args"])
            self.assertIn("split_seed", dataset["args"])
            self.assertIn("train_indices", dataset["args"])
            self.assertIn("test_indices", dataset["args"])


if __name__ == "__main__":
    unittest.main()
