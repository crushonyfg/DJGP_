from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pdebench import (
    downsample_space,
    load_burgers_hdf5,
    make_near_enriched_split_indices,
    make_near_enriched_subset_split_indices,
    make_burgers_surrogate,
    save_dataset,
    split_and_standardize,
    split_and_standardize_by_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDEBench Burgers HDF5 shard into DJGP dataset.pkl format."
    )
    parser.add_argument("--hdf5", required=True, help="Path to a PDEBench Burgers .hdf5 file.")
    parser.add_argument(
        "--output-folder",
        default="data/processed/pdebench_burgers_shockjump",
        help="Output folder that will contain dataset.pkl.",
    )
    parser.add_argument("--qoi", default="shock_jump", choices=["probe", "shock_location", "shock_jump", "right_energy"])
    parser.add_argument("--D", type=int, default=128, help="Spatial dimension after uniform downsampling.")
    parser.add_argument("--max-samples", type=int, default=2000, help="Read only the first N samples from the HDF5 shard.")
    parser.add_argument(
        "--candidate-samples",
        type=int,
        default=None,
        help="Read this many candidates before optional near-threshold subsampling. Defaults to --max-samples.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t0", type=int, default=0, help="Time index used as the input initial/early field.")
    parser.add_argument("--t-final", type=int, default=-1, help="Time index used for the final-field QoI.")
    parser.add_argument("--probe-idx", type=int, default=None, help="Spatial probe index for qoi=probe.")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--jump-threshold", type=float, default=0.5)
    parser.add_argument("--jump-size", type=float, default=2.0)
    parser.add_argument(
        "--near-threshold-delta",
        type=float,
        default=0.05,
        help="Default band for near_threshold_mask when shock_location is available.",
    )
    parser.add_argument(
        "--min-test-near-threshold",
        type=int,
        default=0,
        help="If >0, force at least this many near-threshold samples into the test split.",
    )
    parser.add_argument(
        "--min-train-near-threshold",
        type=int,
        default=0,
        help="If >0 with --candidate-samples, force at least this many near-threshold samples into the train split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_samples = args.candidate_samples or args.max_samples
    U = load_burgers_hdf5(args.hdf5, max_samples=candidate_samples, D=args.D)
    U = downsample_space(U, args.D)

    surrogate = make_burgers_surrogate(
        U,
        qoi=args.qoi,
        t0=args.t0,
        t_final=args.t_final,
        probe_idx=args.probe_idx,
        jump_threshold=args.jump_threshold,
        jump_size=args.jump_size,
        near_threshold_delta=args.near_threshold_delta,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    split_meta = {}
    meta_for_save = dict(surrogate.meta)
    if args.min_test_near_threshold > 0 and candidate_samples > args.max_samples:
        if "near_threshold_mask" not in surrogate.meta:
            raise SystemExit("--min-test-near-threshold requires a QoI with shock_location metadata.")
        train_candidate_idx, test_candidate_idx, split_meta = make_near_enriched_subset_split_indices(
            surrogate.meta["near_threshold_mask"],
            total_samples=args.max_samples,
            test_size=args.test_size,
            min_test_near=args.min_test_near_threshold,
            min_train_near=args.min_train_near_threshold,
            seed=args.seed,
        )
        selected_idx = np.concatenate([train_candidate_idx, test_candidate_idx])
        local_train_idx = np.arange(len(train_candidate_idx))
        local_test_idx = np.arange(len(train_candidate_idx), len(selected_idx))
        X_selected = surrogate.X[selected_idx]
        y_selected = surrogate.y[selected_idx]
        for key in ("shock_location", "threshold_distance", "near_threshold_mask"):
            if key in meta_for_save:
                meta_for_save[key] = meta_for_save[key][selected_idx]
        meta_for_save["N"] = int(len(selected_idx))
        meta_for_save["candidate_samples"] = int(candidate_samples)
        meta_for_save["candidate_indices"] = selected_idx.astype(int).tolist()
        dataset = split_and_standardize_by_indices(
            X_selected,
            y_selected,
            train_indices=local_train_idx,
            test_indices=local_test_idx,
            test_size=args.test_size,
            seed=args.seed,
        )
    elif args.min_test_near_threshold > 0:
        if "near_threshold_mask" not in surrogate.meta:
            raise SystemExit("--min-test-near-threshold requires a QoI with shock_location metadata.")
        train_idx, test_idx, split_meta = make_near_enriched_split_indices(
            surrogate.meta["near_threshold_mask"],
            test_size=args.test_size,
            min_test_near=args.min_test_near_threshold,
            seed=args.seed,
        )
        dataset = split_and_standardize_by_indices(
            surrogate.X,
            surrogate.y,
            train_indices=train_idx,
            test_indices=test_idx,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        dataset = split_and_standardize(
            surrogate.X,
            surrogate.y,
            test_size=args.test_size,
            seed=args.seed,
        )
    meta = {
        **meta_for_save,
        **split_meta,
        "hdf5": str(Path(args.hdf5).resolve()),
        "hdf5_path": str(Path(args.hdf5).resolve()),
        "original_hdf5_path": str(Path(args.hdf5).resolve()),
        "max_samples": args.max_samples,
    }
    output = save_dataset(args.output_folder, dataset, meta)
    print(f"Wrote DJGP dataset: {output}")


if __name__ == "__main__":
    main()
