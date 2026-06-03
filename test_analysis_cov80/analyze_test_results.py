#!/usr/bin/env python3
"""Generate summary plots for the current cov80 model outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = "results_masked_cov80_full_scaled"
DEFAULT_OUT_DIR = "test_analysis_cov80/figures_pred"


def infer_feature_names(results_dir: Path, num_features: int) -> list[str]:
    if num_features == 2:
        return ["freq_dev", "volt_dev"]
    config_path = results_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        metrics = config.get("test_metrics_original") or config.get("test_metrics_scaled") or {}
        if isinstance(metrics, dict) and num_features == 2:
            return ["freq_dev", "volt_dev"]
    fallback = ["freq_dev", "angle_delta", "volt_dev"]
    return fallback[:num_features]


def load_split_arrays(results_dir: Path, split_name: str, use_original: bool) -> tuple[np.ndarray, np.ndarray, str]:
    suffix = "_original" if use_original else ""
    pred_path = results_dir / f"{split_name}_predictions{suffix}.npy"
    true_path = results_dir / f"{split_name}_targets{suffix}.npy"
    if pred_path.exists() and true_path.exists():
        return np.load(pred_path), np.load(true_path), "original" if use_original else "scaled"
    pred_path = results_dir / f"{split_name}_predictions.npy"
    true_path = results_dir / f"{split_name}_targets.npy"
    return np.load(pred_path), np.load(true_path), "scaled"


def flatten_all(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pred.reshape(-1), true.reshape(-1)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred_flat, true_flat = flatten_all(pred, true)
    err = pred_flat - true_flat
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true_flat - np.mean(true_flat)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    denom_mape = np.abs(true_flat)
    nz = denom_mape > 1e-8
    mape = float(np.mean(abs_err[nz] / denom_mape[nz]) * 100.0) if np.any(nz) else 0.0
    smape_denom = np.abs(pred_flat) + np.abs(true_flat)
    smape_mask = smape_denom > 1e-8
    smape = float(np.mean(2.0 * abs_err[smape_mask] / smape_denom[smape_mask]) * 100.0) if np.any(smape_mask) else 0.0
    wape_denom = float(np.sum(np.abs(true_flat)))
    wape = float(np.sum(abs_err) / wape_denom * 100.0) if wape_denom > 1e-8 else 0.0
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "wape": wape,
    }


def choose_windows(true: np.ndarray, pred: np.ndarray, num_windows: int = 3, window_samples: int = 180) -> list[tuple[int, int, str]]:
    sample_abs_err = np.mean(np.abs(pred - true), axis=(1, 2, 3))
    best_idx = int(np.argmin(sample_abs_err))
    worst_idx = int(np.argmax(sample_abs_err))
    mid_idx = int(np.argsort(sample_abs_err)[len(sample_abs_err) // 2])
    selected = [(best_idx, "best"), (mid_idx, "median"), (worst_idx, "worst")][:num_windows]
    windows = []
    max_idx = true.shape[0] - 1
    half = window_samples // 2
    for idx, tag in selected:
        start = max(0, idx - half)
        end = min(max_idx, idx + half)
        windows.append((start, end, tag))
    return windows


def plot_learning_curves(history: dict, out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    plt.plot(epochs, history["test_loss"], label="Test", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_parity(true_flat: np.ndarray, pred_flat: np.ndarray, r2: float, out_path: Path, max_points: int = 50000) -> None:
    if true_flat.size > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(true_flat.size, size=max_points, replace=False)
        true_show = true_flat[idx]
        pred_show = pred_flat[idx]
    else:
        true_show = true_flat
        pred_show = pred_flat
    lo = min(np.min(true_show), np.min(pred_show))
    hi = max(np.max(true_show), np.max(pred_show))
    plt.figure(figsize=(6, 6))
    plt.scatter(true_show, pred_show, s=5, alpha=0.25)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Parity Plot (R^2={r2:.4f})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_residuals(true_flat: np.ndarray, pred_flat: np.ndarray, out_path: Path, max_points: int = 70000) -> None:
    residuals = pred_flat - true_flat
    if true_flat.size > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(true_flat.size, size=max_points, replace=False)
        true_show = true_flat[idx]
        res_show = residuals[idx]
    else:
        true_show = true_flat
        res_show = residuals
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(res_show, bins=100, alpha=0.8)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (Pred - True)")
    axes[0].set_ylabel("Count")
    axes[1].scatter(true_show, res_show, s=5, alpha=0.2)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.5)
    axes[1].set_title("Residual vs Actual")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Residual")
    for ax in axes:
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_timeseries_windows(true: np.ndarray, pred: np.ndarray, out_path: Path, feature_names: list[str], sensor_idx: int, feature_idx: int) -> None:
    windows = choose_windows(true, pred, num_windows=3, window_samples=200)
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 9), sharex=False)
    if len(windows) == 1:
        axes = [axes]
    for ax, (start, end, tag) in zip(axes, windows):
        true_line = true[start:end + 1, 0, sensor_idx, feature_idx]
        pred_line = pred[start:end + 1, 0, sensor_idx, feature_idx]
        x = np.arange(start, end + 1)
        ax.plot(x, true_line, label="Actual", linewidth=1.8)
        ax.plot(x, pred_line, label="Predicted", linewidth=1.6)
        ax.set_title(f"{tag.capitalize()} Window | Sensor {sensor_idx} | Feature {feature_names[feature_idx]}")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Sample index")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_sensor_horizon_heatmap(true: np.ndarray, pred: np.ndarray, out_path: Path, feature_names: list[str], feature_idx: int) -> None:
    abs_err = np.abs(pred[..., feature_idx] - true[..., feature_idx])
    err_map = np.mean(abs_err, axis=0).T
    plt.figure(figsize=(9, 6))
    im = plt.imshow(err_map, aspect="auto", origin="lower")
    plt.colorbar(im, label="MAE")
    plt.xlabel("Sensor index")
    plt.ylabel("Horizon step")
    plt.title(f"MAE Heatmap by Horizon x Sensor ({feature_names[feature_idx]})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_error_cdf(true_flat: np.ndarray, pred_flat: np.ndarray, out_path: Path) -> None:
    abs_err = np.abs(pred_flat - true_flat)
    x = np.sort(abs_err)
    y = np.linspace(0, 1, len(x), endpoint=False)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=2)
    for q in [0.5, 0.8, 0.9, 0.95]:
        v = np.quantile(abs_err, q)
        plt.axvline(v, linestyle="--", linewidth=1, alpha=0.6)
        plt.text(v, q, f"Q{int(q*100)}={v:.3f}", rotation=90, va="bottom", ha="right")
    plt.xlabel("Absolute Error")
    plt.ylabel("Fraction of predictions <= x")
    plt.title("CDF of Absolute Error")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze current cov80 model predictions and create figures")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split_name", type=str, default="test", choices=["test", "pred"])
    parser.add_argument("--sensor_idx", type=int, default=0)
    parser.add_argument("--feature_idx", type=int, default=0)
    parser.add_argument("--use_original", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = np.load(results_dir / "history.npy", allow_pickle=True).item()
    pred, true, data_variant = load_split_arrays(results_dir, args.split_name, args.use_original)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (samples, horizon, sensors, features), got {pred.ndim}D")

    feature_names = infer_feature_names(results_dir, pred.shape[-1])
    feature_idx = int(np.clip(args.feature_idx, 0, pred.shape[-1] - 1))
    sensor_idx = int(np.clip(args.sensor_idx, 0, pred.shape[2] - 1))

    metrics_all = compute_metrics(pred, true)
    metrics_by_feature = {}
    for idx in range(pred.shape[-1]):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        metrics_by_feature[name] = compute_metrics(pred[..., idx:idx+1], true[..., idx:idx+1])

    pred_flat, true_flat = flatten_all(pred, true)
    plot_learning_curves(history, out_dir / "01_learning_curves.png")
    plot_parity(true_flat, pred_flat, metrics_all["r2"], out_dir / "02_parity_plot.png")
    plot_residuals(true_flat, pred_flat, out_dir / "03_residual_plots.png")
    plot_timeseries_windows(true, pred, out_dir / "04_timeseries_overlays.png", feature_names, sensor_idx, feature_idx)
    for idx, feature_name in enumerate(feature_names):
        plot_timeseries_windows(true, pred, out_dir / f"04_{idx+1}_{feature_name}.png", feature_names, sensor_idx, idx)
    plot_sensor_horizon_heatmap(true, pred, out_dir / "05_sensor_horizon_mae_heatmap.png", feature_names, feature_idx)
    plot_error_cdf(true_flat, pred_flat, out_dir / "06_abs_error_cdf.png")

    summary = {
        "shapes": {"predictions": list(pred.shape), "targets": list(true.shape)},
        "overall_metrics": metrics_all,
        "metrics_by_feature": metrics_by_feature,
        "selected_indices": {
            "sensor_idx": sensor_idx,
            "feature_idx": feature_idx,
            "feature_name": feature_names[feature_idx],
            "split_name": args.split_name,
            "data_variant": data_variant,
        },
    }
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Saved outputs:")
    for name in [
        "01_learning_curves.png",
        "02_parity_plot.png",
        "03_residual_plots.png",
        "04_timeseries_overlays.png",
        "05_sensor_horizon_mae_heatmap.png",
        "06_abs_error_cdf.png",
        "summary_metrics.json",
    ]:
        print(f"  - {out_dir / name}")
    for idx, feature_name in enumerate(feature_names):
        print(f"  - {out_dir / ('04_' + str(idx + 1) + '_' + feature_name + '.png')}")


if __name__ == "__main__":
    main()