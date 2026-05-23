from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


DATASET_CANDIDATES = [
    "Wine Quality",
    "Parkinsons Telemonitoring",
    "Appliances Energy Prediction",
]

OUTPUT_DIR = Path(__file__).resolve().parent / "pca_nonstationary_plots"
RANDOM_SEED = 42


def _to_numeric_numpy(frame_like) -> np.ndarray:
    array = np.asarray(frame_like)
    if np.issubdtype(array.dtype, np.number):
        return array.astype(float)

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("需要 pandas 来处理非数值列。") from exc

    numeric_df = pd.DataFrame(frame_like).apply(pd.to_numeric, errors="coerce")
    return numeric_df.to_numpy(dtype=float)


def _impute_nan_with_column_mean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    col_means = np.nanmean(x, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(x))
    if nan_rows.size > 0:
        x = x.copy()
        x[nan_rows, nan_cols] = col_means[nan_cols]
    return x


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name == "Appliances Energy Prediction":
        dataset = fetch_ucirepo(id=374)
        x = dataset.data.features.iloc[:, 1:]
        y = dataset.data.targets
    elif dataset_name == "Parkinsons Telemonitoring":
        dataset = fetch_ucirepo(id=189)
        x = dataset.data.features.iloc[:, :-1]
        y = dataset.data.targets.iloc[:, 0]
    elif dataset_name == "Wine Quality":
        dataset = fetch_ucirepo(id=186)
        x = dataset.data.features
        y = dataset.data.targets.iloc[:, 0]
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    x_np = _to_numeric_numpy(x)
    y_np = np.asarray(y, dtype=float).reshape(-1)

    finite_mask = np.isfinite(y_np)
    x_np = x_np[finite_mask]
    y_np = y_np[finite_mask]

    x_np = _impute_nan_with_column_mean(x_np)
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    y_median = np.nanmedian(y_np)
    y_np = np.nan_to_num(y_np, nan=y_median, posinf=y_median, neginf=y_median)

    return x_np, y_np


def compute_pca_embeddings(x: np.ndarray) -> tuple[np.ndarray, PCA]:
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    z_2d = pca_2d.fit_transform(x_std)
    return z_2d, pca_2d


def compute_binned_surface(
    z_2d: np.ndarray,
    y: np.ndarray,
    grid_size: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    x_axis = z_2d[:, 0]
    y_axis = z_2d[:, 1]

    x_edges = np.linspace(np.min(x_axis), np.max(x_axis), grid_size + 1)
    y_edges = np.linspace(np.min(y_axis), np.max(y_axis), grid_size + 1)

    counts, _, _ = np.histogram2d(x_axis, y_axis, bins=[x_edges, y_edges])
    weighted_sum, _, _ = np.histogram2d(x_axis, y_axis, bins=[x_edges, y_edges], weights=y)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_surface = weighted_sum / counts

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

    masked_surface = np.ma.masked_invalid(mean_surface)
    return x_grid, y_grid, masked_surface


def prepare_dataset_result(dataset_name: str, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    z_2d, pca_2d = compute_pca_embeddings(x)
    grid_x, grid_y, grid_z = compute_binned_surface(z_2d, y, grid_size=80)

    explained = pca_2d.explained_variance_ratio_

    return {
        "z_2d": z_2d,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "y": y,
        "explained": explained,
    }


def plot_summary(all_results: dict[str, dict[str, np.ndarray]], output_dir: Path) -> None:
    n = len(DATASET_CANDIDATES)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), constrained_layout=True)

    if n == 1:
        axes = [axes]

    for idx, dataset_name in enumerate(DATASET_CANDIDATES):
        result = all_results[dataset_name]
        ax = axes[idx]

        z_2d = result["z_2d"]
        grid_x = result["grid_x"]
        grid_y = result["grid_y"]
        grid_z = result["grid_z"]
        y = result["y"]
        explained = result["explained"]

        contour = ax.contourf(grid_x, grid_y, grid_z, levels=25, cmap="viridis")
        scatter = ax.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            c=y,
            cmap="viridis",
            s=8,
            alpha=0.30,
            edgecolors="none",
        )

        ax.set_title(
            f"{dataset_name}\nPC1={explained[0]:.3f}, PC2={explained[1]:.3f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.20)

        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("mean y")

    fig.suptitle("UCI datasets: PCA 2D views for potential nonstationary jump inspection", fontsize=14)
    fig.savefig(output_dir / "all_datasets_pca_2d_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, np.ndarray]] = {}

    for dataset_name in DATASET_CANDIDATES:
        print(f"Processing: {dataset_name}")
        x, y = load_dataset(dataset_name)
        result = prepare_dataset_result(dataset_name, x, y)
        all_results[dataset_name] = result

    plot_summary(all_results, OUTPUT_DIR)
    print(f"Finished. Figure saved to: {OUTPUT_DIR / 'all_datasets_pca_2d_summary.png'}")


if __name__ == "__main__":
    main()