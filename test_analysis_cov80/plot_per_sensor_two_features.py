#!/usr/bin/env python3
"""Create one plot per sensor for the current two-target cov80 model outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = "results_masked_cov80_full_scaled"
DEFAULT_OUT_DIR = "test_analysis_cov80/per_sensor_two_features_pred"


def infer_feature_names(num_features: int) -> list[str]:
    return ["freq_dev", "volt_dev"] if num_features == 2 else [f"feature_{idx}" for idx in range(num_features)]


def load_split_arrays(results_dir: Path, split_name: str, use_original: bool) -> tuple[np.ndarray, np.ndarray, str]:
    suffix = "_original" if use_original else ""
    pred_path = results_dir / f"{split_name}_predictions{suffix}.npy"
    true_path = results_dir / f"{split_name}_targets{suffix}.npy"
    if pred_path.exists() and true_path.exists():
        return np.load(pred_path), np.load(true_path), "original" if use_original else "scaled"
    return np.load(results_dir / f"{split_name}_predictions.npy"), np.load(results_dir / f"{split_name}_targets.npy"), "scaled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-sensor plots for the current cov80 model")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split_name", type=str, default="pred", choices=["test", "pred"])
    parser.add_argument("--use_original", action="store_true", default=True)
    parser.add_argument("--horizon_reduce", type=str, default="mean", choices=["mean", "first", "last"])
    parser.add_argument("--max_sensors", type=int, default=0)
    return parser.parse_args()


def reduce_horizon(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.mean(arr, axis=1)
    if mode == "first":
        return arr[:, 0, :, :]
    return arr[:, -1, :, :]


def compute_metrics_1d(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, true, data_variant = load_split_arrays(results_dir, args.split_name, args.use_original)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (samples, horizon, sensors, features), got {pred.ndim}D")
    pred_r = reduce_horizon(pred, args.horizon_reduce)
    true_r = reduce_horizon(true, args.horizon_reduce)
    num_samples, num_sensors, num_features = pred_r.shape
    feature_names = infer_feature_names(num_features)
    sensor_count = num_sensors if args.max_sensors <= 0 else min(args.max_sensors, num_sensors)

    summary = {
        "input_shapes": {"predictions": list(pred.shape), "targets": list(true.shape), "reduced": list(pred_r.shape)},
        "horizon_reduce": args.horizon_reduce,
        "num_sensors_plotted": int(sensor_count),
        "sensor_metrics": {},
        "split_name": args.split_name,
        "data_variant": data_variant,
    }
    x = np.arange(num_samples)

    for sensor_idx in range(sensor_count):
        fig, axes = plt.subplots(num_features, 1, figsize=(14, 3 * num_features), sharex=True)
        if num_features == 1:
            axes = [axes]
        feature_metrics = {}
        for feature_idx, feature_name in enumerate(feature_names):
            y_true = true_r[:, sensor_idx, feature_idx]
            y_pred = pred_r[:, sensor_idx, feature_idx]
            metrics = compute_metrics_1d(y_pred, y_true)
            feature_metrics[feature_name] = metrics
            ax = axes[feature_idx]
            ax.plot(x, y_true, label="Actual", linewidth=1.4, alpha=0.9)
            ax.plot(x, y_pred, label="Predicted", linewidth=1.2, alpha=0.85)
            ax.set_ylabel(feature_name)
            ax.set_title(f"Sensor {sensor_idx} | {feature_name} | MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.3f}")
            ax.grid(alpha=0.25)
            if feature_idx == 0:
                ax.legend(loc="upper right")
        axes[-1].set_xlabel("Sample index")
        plt.tight_layout()
        fig.savefig(out_dir / f"sensor_{sensor_idx:03d}_features.png", dpi=200)
        plt.close(fig)
        summary["sensor_metrics"][f"sensor_{sensor_idx}"] = feature_metrics

    with open(out_dir / "summary_per_sensor.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved {sensor_count} sensor figures to {out_dir}")
    print("Example file:", out_dir / "sensor_000_features.png")
    print("Summary file:", out_dir / "summary_per_sensor.json")


if __name__ == "__main__":
    main()