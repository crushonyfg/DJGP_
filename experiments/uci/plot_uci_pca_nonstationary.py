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
    y_np = np.nan_to_num(y_np, nan=np.nanmedian(y_np), posinf=np.nanmedian(y_np), neginf=np.nanmedian(y_np))
    return x_np, y_np


def compute_pca_embeddings(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA, PCA]:
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    pca_1d = PCA(n_components=1, random_state=RANDOM_SEED)
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    z_1d = pca_1d.fit_transform(x_std).reshape(-1)
    z_2d = pca_2d.fit_transform(x_std)
    return z_1d, z_2d, pca_1d, pca_2d


def compute_binned_curve(x_1d: np.ndarray, y: np.ndarray, num_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(np.min(x_1d), np.max(x_1d), num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_mean = np.full(num_bins, np.nan)

    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (x_1d >= edges[i]) & (x_1d <= edges[i + 1])
        else:
            mask = (x_1d >= edges[i]) & (x_1d < edges[i + 1])
        if np.any(mask):
            y_mean[i] = np.mean(y[mask])

    return centers, y_mean


def compute_binned_surface(
    z_2d: np.ndarray,
    y: np.ndarray,
    grid_size: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    x = z_2d[:, 0]
    y_axis = z_2d[:, 1]

    x_edges = np.linspace(np.min(x), np.max(x), grid_size + 1)
    y_edges = np.linspace(np.min(y_axis), np.max(y_axis), grid_size + 1)

    counts, _, _ = np.histogram2d(x, y_axis, bins=[x_edges, y_edges])
    weighted_sum, _, _ = np.histogram2d(x, y_axis, bins=[x_edges, y_edges], weights=y)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_surface = weighted_sum / counts

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
    masked_surface = np.ma.masked_invalid(mean_surface)
    return x_grid, y_grid, masked_surface


def plot_dataset_panels(dataset_name: str, x: np.ndarray, y: np.ndarray, output_dir: Path) -> dict[str, np.ndarray]:
    z_1d, z_2d, pca_1d, pca_2d = compute_pca_embeddings(x)
    sort_idx = np.argsort(z_1d)
    curve_x, curve_y = compute_binned_curve(z_1d, y, num_bins=80)
    grid_x, grid_y, grid_z = compute_binned_surface(z_2d, y, grid_size=80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].scatter(z_1d, y, s=10, alpha=0.22, color="tab:blue", edgecolors="none")
    axes[0].plot(z_1d[sort_idx], y[sort_idx], color="lightgray", alpha=0.15, linewidth=0.8)
    axes[0].plot(curve_x, curve_y, color="tab:red", linewidth=2.2, label="bin mean of y")
    axes[0].set_title(f"{dataset_name}: 1D PCA")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    contour = axes[1].contourf(grid_x, grid_y, grid_z, levels=25, cmap="viridis")
    axes[1].scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=y,
        cmap="viridis",
        s=8,
        alpha=0.3,
        edgecolors="none",
    )
    fig.colorbar(contour, ax=axes[1], label="mean y")
    axes[1].set_title(f"{dataset_name}: 2D PCA contour")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    explained = pca_2d.explained_variance_ratio_
    fig.suptitle(
        f"{dataset_name} | standardized features -> PCA "
        f"(PC1={explained[0]:.3f}, PC2={explained[1]:.3f})",
        fontsize=12,
    )
    fig.tight_layout()

    filename = dataset_name.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(output_dir / f"{filename}_pca_nonstationary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "z_1d": z_1d,
        "z_2d": z_2d,
        "curve_x": curve_x,
        "curve_y": curve_y,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
    }


def plot_summary(all_results: dict[str, dict[str, np.ndarray]], output_dir: Path) -> None:
    fig, axes = plt.subplots(len(all_results), 2, figsize=(14, 4.6 * len(all_results)))
    if len(all_results) == 1:
        axes = np.array([axes])

    for row_idx, dataset_name in enumerate(DATASET_CANDIDATES):
        result = all_results[dataset_name]

        ax_1d = axes[row_idx, 0]
        ax_2d = axes[row_idx, 1]

        z_1d = result["z_1d"]
        z_2d = result["z_2d"]
        curve_x = result["curve_x"]
        curve_y = result["curve_y"]
        grid_x = result["grid_x"]
        grid_y = result["grid_y"]
        grid_z = result["grid_z"]

        sort_idx = np.argsort(z_1d)
        ax_1d.scatter(z_1d, result["y"], s=8, alpha=0.18, color="tab:blue", edgecolors="none")
        ax_1d.plot(z_1d[sort_idx], result["y"][sort_idx], color="lightgray", alpha=0.12, linewidth=0.7)
        ax_1d.plot(curve_x, curve_y, color="tab:red", linewidth=2.0)
        ax_1d.set_title(f"{dataset_name}: 1D PCA")
        ax_1d.set_xlabel("PC1")
        ax_1d.set_ylabel("y")
        ax_1d.grid(alpha=0.25)

        contour = ax_2d.contourf(grid_x, grid_y, grid_z, levels=25, cmap="viridis")
        ax_2d.scatter(z_2d[:, 0], z_2d[:, 1], c=result["y"], cmap="viridis", s=7, alpha=0.22, edgecolors="none")
        fig.colorbar(contour, ax=ax_2d, label="mean y")
        ax_2d.set_title(f"{dataset_name}: 2D PCA contour")
        ax_2d.set_xlabel("PC1")
        ax_2d.set_ylabel("PC2")

    fig.suptitle("UCI datasets: PCA views for potential nonstationary jump inspection", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "all_datasets_pca_nonstationary_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, np.ndarray]] = {}

    for dataset_name in DATASET_CANDIDATES:
        print(f"Processing: {dataset_name}")
        x, y = load_dataset(dataset_name)
        result = plot_dataset_panels(dataset_name, x, y, OUTPUT_DIR)
        result["y"] = y
        all_results[dataset_name] = result

    plot_summary(all_results, OUTPUT_DIR)
    print(f"Finished. Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
