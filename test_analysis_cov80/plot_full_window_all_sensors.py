#!/usr/bin/env python3
"""Plot full-window all-sensor overlays and heatmaps for the current cov80 run."""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = "results_masked_cov80_full_scaled"
DEFAULT_OUT_DIR = "test_analysis_cov80/full_window_all_sensors_pred"
DEFAULT_FEATURE_NAMES = ["freq_dev", "angle_delta", "volt_dev"]


def infer_feature_names(results_dir: Path, num_features: int) -> list[str]:
    if num_features == 2:
        return ["freq_dev", "volt_dev"]
    return DEFAULT_FEATURE_NAMES[:num_features]


def load_split_arrays(results_dir: Path, split_name: str, use_original: bool) -> tuple[np.ndarray, np.ndarray, str]:
    suffix = "_original" if use_original else ""
    pred_path = results_dir / f"{split_name}_predictions{suffix}.npy"
    true_path = results_dir / f"{split_name}_targets{suffix}.npy"
    if pred_path.exists() and true_path.exists():
        return np.load(pred_path), np.load(true_path), "original" if use_original else "scaled"
    return np.load(results_dir / f"{split_name}_predictions.npy"), np.load(results_dir / f"{split_name}_targets.npy"), "scaled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize all sensors across the full prediction window")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split_name", type=str, default="pred", choices=["test", "pred"])
    parser.add_argument("--use_original", action="store_true", default=True)
    parser.add_argument("--horizon_reduce", type=str, default="mean", choices=["mean", "first", "last"])
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--sensor_ordering", type=str, default="graph_spectral", choices=["none", "graph_spectral", "longitude", "latitude", "sensor_id"])
    parser.add_argument("--adjacency_file", type=str, default="results/A_geo.npy")
    parser.add_argument("--sensor_order_file", type=str, default="results/sensor_order.npy")
    parser.add_argument("--metadata_file", type=str, default="data/FDRLocation.xlsx")
    return parser.parse_args()


def reduce_horizon(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.mean(arr, axis=1)
    if mode == "first":
        return arr[:, 0, :, :]
    return arr[:, -1, :, :]


def plot_overlay_all_sensors(true_mat: np.ndarray, pred_mat: np.ndarray, feature_name: str, alpha: float, out_path: Path) -> None:
    samples, sensors = true_mat.shape
    x = np.arange(samples)
    fig, ax = plt.subplots(figsize=(14, 6))
    for sensor_idx in range(sensors):
        ax.plot(x, true_mat[:, sensor_idx], color="tab:blue", alpha=alpha, linewidth=0.6)
    for sensor_idx in range(sensors):
        ax.plot(x, pred_mat[:, sensor_idx], color="tab:orange", alpha=alpha, linewidth=0.6)
    true_proxy = plt.Line2D([0], [0], color="tab:blue", linewidth=2, label="Actual (all sensors)")
    pred_proxy = plt.Line2D([0], [0], color="tab:orange", linewidth=2, label="Predicted (all sensors)")
    ax.legend(handles=[true_proxy, pred_proxy], loc="upper right")
    ax.set_title(f"Full Window Overlay | Feature: {feature_name} | Sensors: {sensors}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_heatmap(mat: np.ndarray, title: str, cbar_label: str, out_path: Path, x_label: str = "Sensor index") -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sample index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def load_sensor_ids(sensor_order_file: Path, expected_sensors: int) -> np.ndarray:
    sensor_ids = np.asarray(np.load(sensor_order_file)).astype(int)
    if sensor_ids.shape[0] != expected_sensors:
        raise ValueError(f"Sensor order length mismatch: expected {expected_sensors}, got {sensor_ids.shape[0]}")
    return sensor_ids


def compute_graph_spectral_order(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=1)
    laplacian = np.diag(degree) - adjacency
    _, eigenvectors = np.linalg.eigh(laplacian)
    return np.argsort(eigenvectors[:, 1]) if adjacency.shape[0] > 1 else np.arange(adjacency.shape[0])


def compute_metadata_order(sensor_ids: np.ndarray, metadata_file: Path, primary_key: str, secondary_key: str) -> Tuple[np.ndarray, list]:
    import pandas as pd

    metadata = pd.read_excel(metadata_file)
    subset = metadata[["FDRID", primary_key, secondary_key] + [c for c in ["GridName"] if c in metadata.columns]].copy()
    subset["FDRID"] = subset["FDRID"].astype(int)
    subset = subset.drop_duplicates(subset="FDRID").set_index("FDRID")
    rows = []
    for original_index, sensor_id in enumerate(sensor_ids.tolist()):
        row = subset.loc[sensor_id]
        rows.append({
            "original_index": original_index,
            "sensor_id": sensor_id,
            "primary": float(row[primary_key]),
            "secondary": float(row[secondary_key]),
            "grid_name": row.get("GridName", ""),
        })
    rows.sort(key=lambda item: (item["primary"], item["secondary"], item["sensor_id"]))
    return np.asarray([row["original_index"] for row in rows], dtype=int), rows


def get_sensor_reordering(ordering: str, adjacency_file: Path, sensor_order_file: Path, metadata_file: Path, expected_sensors: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, list]:
    if ordering == "none":
        return None, None, "raw index", []
    sensor_ids = load_sensor_ids(sensor_order_file, expected_sensors)
    if ordering == "graph_spectral":
        adjacency = np.load(adjacency_file)
        permutation = compute_graph_spectral_order(adjacency)
        rows = [{"original_index": int(idx), "sensor_id": int(sensor_ids[idx])} for idx in permutation.tolist()]
        return permutation, sensor_ids, "graph spectral order", rows
    if ordering == "longitude":
        permutation, rows = compute_metadata_order(sensor_ids, metadata_file, "Longitude", "Latitude")
        return permutation, sensor_ids, "longitude/latitude order", rows
    if ordering == "latitude":
        permutation, rows = compute_metadata_order(sensor_ids, metadata_file, "Latitude", "Longitude")
        return permutation, sensor_ids, "latitude/longitude order", rows
    permutation = np.argsort(sensor_ids)
    rows = [{"original_index": int(idx), "sensor_id": int(sensor_ids[idx])} for idx in permutation.tolist()]
    return permutation, sensor_ids, "sensor ID order", rows


def save_sensor_reordering(rows: list, out_path: Path) -> None:
    if not rows:
        return
    fieldnames = ["reordered_index", "original_index", "sensor_id", "grid_name", "primary", "secondary"]
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for reordered_index, row in enumerate(rows):
            writer.writerow({
                "reordered_index": reordered_index,
                "original_index": row.get("original_index", ""),
                "sensor_id": row.get("sensor_id", ""),
                "grid_name": row.get("grid_name", ""),
                "primary": row.get("primary", ""),
                "secondary": row.get("secondary", ""),
            })


def summarize_feature(true_mat: np.ndarray, pred_mat: np.ndarray) -> dict:
    err = pred_mat - true_mat
    abs_err = np.abs(err)
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_actual": float(np.mean(true_mat)),
        "mean_predicted": float(np.mean(pred_mat)),
        "p50_abs_error": float(np.quantile(abs_err, 0.50)),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
    }


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
    feature_names = infer_feature_names(results_dir, pred.shape[-1])
    pred_reduced = reduce_horizon(pred, args.horizon_reduce)
    true_reduced = reduce_horizon(true, args.horizon_reduce)

    permutation, sensor_ids, ordering_label, ordering_rows = get_sensor_reordering(
        ordering=args.sensor_ordering,
        adjacency_file=Path(args.adjacency_file),
        sensor_order_file=Path(args.sensor_order_file),
        metadata_file=Path(args.metadata_file),
        expected_sensors=pred_reduced.shape[1],
    )
    if permutation is not None:
        save_sensor_reordering(ordering_rows, out_dir / f"sensor_order_{args.sensor_ordering}.csv")

    feature_stats = {}
    for feature_idx in range(pred_reduced.shape[-1]):
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
        true_mat = true_reduced[:, :, feature_idx]
        pred_mat = pred_reduced[:, :, feature_idx]
        abs_err = np.abs(pred_mat - true_mat)
        stem = f"{feature_idx + 1:02d}_{feature_name}"
        plot_overlay_all_sensors(true_mat, pred_mat, feature_name, args.alpha, out_dir / f"{stem}_overlay_all_sensors.png")
        plot_heatmap(true_mat, f"Actual Heatmap | {feature_name} | Full Window", "Actual value", out_dir / f"{stem}_actual_heatmap.png")
        plot_heatmap(pred_mat, f"Predicted Heatmap | {feature_name} | Full Window", "Predicted value", out_dir / f"{stem}_pred_heatmap.png")
        plot_heatmap(abs_err, f"Absolute Error Heatmap | {feature_name} | Full Window", "|pred - actual|", out_dir / f"{stem}_abs_error_heatmap.png")
        if permutation is not None:
            ordered_abs_err = abs_err[:, permutation]
            plot_heatmap(ordered_abs_err, f"Absolute Error Heatmap | {feature_name} | Full Window | {ordering_label}", "|pred - actual|", out_dir / f"{stem}_abs_error_heatmap_{args.sensor_ordering}.png", x_label=f"Sensor index ({ordering_label})")
        feature_stats[feature_name] = summarize_feature(true_mat, pred_mat)

    summary = {
        "input_shapes": {"predictions": list(pred.shape), "targets": list(true.shape), "reduced": list(pred_reduced.shape)},
        "horizon_reduce": args.horizon_reduce,
        "alpha": args.alpha,
        "sensor_ordering": args.sensor_ordering,
        "sensor_ordering_label": ordering_label,
        "ordered_sensor_count": int(len(sensor_ids)) if sensor_ids is not None else 0,
        "feature_stats": feature_stats,
        "split_name": args.split_name,
        "data_variant": data_variant,
    }
    with open(out_dir / "full_window_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print("Saved full-window all-sensor plots to:", out_dir)


if __name__ == "__main__":
    main()