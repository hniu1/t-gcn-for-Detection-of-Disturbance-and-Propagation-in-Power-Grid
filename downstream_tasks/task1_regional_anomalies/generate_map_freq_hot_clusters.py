#!/usr/bin/env python3
"""Generate map-style hot frequency cluster CSVs used by episode collapsing."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = "results_masked_cov80_full_scaled"
DEFAULT_OUT_DIR = "downstream_tasks/task1_regional_anomalies/outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate map-style hot frequency cluster CSV files")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split_name", type=str, default="pred", choices=["test", "pred"])
    parser.add_argument("--use_original", action="store_true", default=True)
    parser.add_argument("--horizon_reduce", type=str, default="mean", choices=["mean", "first", "last"])
    parser.add_argument("--feature_idx", type=int, default=0, help="Feature index; 0 is freq_dev in current setup")
    parser.add_argument(
        "--color_max_quantile",
        type=float,
        default=0.90,
        help="Match map logic: hot sensors are abs error >= this quantile",
    )
    parser.add_argument("--edge_threshold", type=float, default=0.0, help="Adjacency threshold for connected components")
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--max_top_rows", type=int, default=200)
    parser.add_argument("--sensor_order_file", type=str, default="results/sensor_order.npy")
    parser.add_argument("--adjacency_file", type=str, default="results/A_geo.npy")
    return parser.parse_args()


def load_run_config(results_dir: Path) -> dict:
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {results_dir}")
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_split_arrays(results_dir: Path, split_name: str, use_original: bool) -> tuple[np.ndarray, np.ndarray]:
    suffix = "_original" if use_original else ""
    pred_path = results_dir / f"{split_name}_predictions{suffix}.npy"
    true_path = results_dir / f"{split_name}_targets{suffix}.npy"
    if pred_path.exists() and true_path.exists():
        return np.load(pred_path), np.load(true_path)
    return np.load(results_dir / f"{split_name}_predictions.npy"), np.load(results_dir / f"{split_name}_targets.npy")


def reduce_horizon(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.mean(arr, axis=1)
    if mode == "first":
        return arr[:, 0, :, :]
    return arr[:, -1, :, :]


def reconstruct_split_timestamps(results_dir: Path, split_name: str, num_samples: int, horizon_reduce: str) -> list[datetime]:
    config = load_run_config(results_dir)
    data_dir = Path(config["data_dir"])
    sensor_order_file = Path(config["adjacency_file"]).parent / "sensor_order.npy"
    sensor_order = [int(x) for x in np.load(sensor_order_file)]
    sample_rate_hz = float(config["sample_rate_hz"])
    min_step_coverage = float(config["min_step_coverage"])
    tin = int(config["Tin"])
    horizon = int(config["H"])
    split_stride = int(config[f"stride_{split_name}"])

    sensor_start_time: dict[int, pd.Timestamp] = {}
    sensor_lengths: dict[int, int] = {}
    for sensor_id in sensor_order:
        files = list(data_dir.glob(f"**/{sensor_id}-*.parquet"))
        if not files:
            continue
        df = pd.read_parquet(files[0], columns=["ReceivedTime"])
        if df.empty:
            continue
        received_time = pd.to_datetime(df["ReceivedTime"])
        sensor_start_time[sensor_id] = received_time.min()
        sensor_lengths[sensor_id] = len(received_time)

    if not sensor_start_time:
        raise ValueError("Could not reconstruct timestamps: no sensor files found")

    global_start = min(sensor_start_time.values())
    offsets = {
        sensor_id: int(round((sensor_start_time[sensor_id] - global_start).total_seconds() * sample_rate_hz))
        for sensor_id in sensor_start_time
    }
    max_len = max(offsets[sensor_id] + sensor_lengths[sensor_id] for sensor_id in sensor_start_time)

    ordered_sensor_ids = [sensor_id for sensor_id in sensor_order if sensor_id in sensor_start_time]
    observed_mask = np.zeros((max_len, len(ordered_sensor_ids)), dtype=np.float32)
    for sensor_idx, sensor_id in enumerate(ordered_sensor_ids):
        start_idx = max(offsets[sensor_id], 0)
        end_idx = min(start_idx + sensor_lengths[sensor_id], max_len)
        observed_mask[start_idx:end_idx, sensor_idx] = 1.0

    keep_steps = observed_mask.mean(axis=1) >= min_step_coverage
    filtered_indices = np.flatnonzero(keep_steps)
    total_filtered = len(filtered_indices)

    train_end = int(total_filtered * float(config["train_frac"]))
    val_end = train_end + int(total_filtered * float(config["val_frac"]))
    test_end = val_end + int(total_filtered * float(config["test_frac"]))
    pred_end = min(test_end + int(total_filtered * float(config["pred_frac"])), total_filtered)

    split_bounds = {
        "test": (val_end, test_end),
        "pred": (test_end, pred_end if float(config["pred_frac"]) > 0 else total_filtered),
    }
    split_start, split_stop = split_bounds[split_name]
    split_len = split_stop - split_start
    valid_starts = list(range(0, split_len - tin - horizon + 1, split_stride))
    if len(valid_starts) != num_samples:
        raise ValueError(
            f"Timestamp reconstruction mismatch for {split_name}: expected {num_samples}, got {len(valid_starts)}"
        )

    horizon_offset = tin + horizon - 1 if horizon_reduce == "last" else tin
    timestamps: list[datetime] = []
    for start_idx in valid_starts:
        filtered_idx = filtered_indices[split_start + start_idx + horizon_offset]
        timestamps.append(global_start.to_pydatetime() + timedelta(seconds=filtered_idx / sample_rate_hz))
    return timestamps


def connected_components(mask: np.ndarray) -> list[list[int]]:
    n = mask.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(mask[u]).tolist():
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def build_sensor_name_map(results_dir: Path, sensor_ids: np.ndarray) -> dict[int, str]:
    name_map = {int(sid): f"Sensor-{int(sid)}" for sid in sensor_ids.tolist()}
    try:
        cfg = load_run_config(results_dir)
        data_dir = Path(cfg["data_dir"])
        for sid in sensor_ids.tolist():
            matches = sorted(data_dir.glob(f"**/{int(sid)}-*.parquet"))
            if not matches:
                continue
            parts = matches[0].stem.split("-", 2)
            if len(parts) >= 2 and parts[1].strip():
                name_map[int(sid)] = parts[1].strip()
    except Exception:
        pass
    return name_map


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred, true = load_split_arrays(results_dir, args.split_name, args.use_original)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (samples, horizon, sensors, features), got {pred.ndim}D")

    pred_reduced = reduce_horizon(pred, args.horizon_reduce)
    true_reduced = reduce_horizon(true, args.horizon_reduce)
    abs_err = np.abs(pred_reduced[:, :, args.feature_idx] - true_reduced[:, :, args.feature_idx])

    hot_threshold = float(np.quantile(abs_err, min(max(args.color_max_quantile, 0.01), 1.0)))
    active = abs_err >= hot_threshold

    sensor_ids = np.load(Path(args.sensor_order_file)).astype(int)
    if sensor_ids.shape[0] != abs_err.shape[1]:
        raise ValueError(f"Sensor order length mismatch: expected {abs_err.shape[1]}, got {sensor_ids.shape[0]}")

    adjacency = np.load(Path(args.adjacency_file)).astype(float)
    if adjacency.shape[0] != sensor_ids.shape[0] or adjacency.shape[1] != sensor_ids.shape[0]:
        raise ValueError(
            f"Adjacency shape mismatch: expected {(sensor_ids.shape[0], sensor_ids.shape[0])}, got {adjacency.shape}"
        )
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    edge_mask_global = adjacency > float(args.edge_threshold)

    timestamps = reconstruct_split_timestamps(results_dir, args.split_name, abs_err.shape[0], args.horizon_reduce)
    sensor_name_map = build_sensor_name_map(results_dir, sensor_ids)

    rows: list[dict] = []
    cluster_id = 0
    for t in range(active.shape[0]):
        active_nodes = np.flatnonzero(active[t])
        if active_nodes.size < args.min_cluster_size:
            continue

        induced = edge_mask_global[np.ix_(active_nodes, active_nodes)]
        comps = connected_components(induced)
        for comp in comps:
            if len(comp) < args.min_cluster_size:
                continue
            nodes = active_nodes[np.array(comp, dtype=int)]
            cluster_sensor_ids = [int(sensor_ids[i]) for i in nodes.tolist()]
            cluster_sensor_names = [sensor_name_map.get(sid, f"Sensor-{sid}") for sid in cluster_sensor_ids]
            cluster_values = abs_err[t, nodes]
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "time_idx": int(t),
                    "timestamp": timestamps[t].strftime("%Y-%m-%d %H:%M:%S"),
                    "cluster_size": int(len(nodes)),
                    "mean_abs_err": float(np.mean(cluster_values)),
                    "max_abs_err": float(np.max(cluster_values)),
                    "sensor_ids": ";".join(str(x) for x in cluster_sensor_ids),
                    "sensor_names": ";".join(cluster_sensor_names),
                    "sensor_indices": ";".join(str(int(x)) for x in nodes.tolist()),
                }
            )
            cluster_id += 1

    if not rows:
        raise ValueError("No map-hot clusters found; try lowering color_max_quantile or min_cluster_size")

    clusters_df = pd.DataFrame(rows)
    clusters_df = clusters_df.sort_values(["cluster_size", "mean_abs_err", "max_abs_err"], ascending=[False, False, False]).reset_index(drop=True)
    top_df = clusters_df.head(max(1, args.max_top_rows)).copy()

    all_path = out_dir / "map_freq_hot_clusters_all.csv"
    top_path = out_dir / "map_freq_hot_clusters_top.csv"
    clusters_df.to_csv(all_path, index=False)
    top_df.to_csv(top_path, index=False)

    print("Saved map-hot cluster CSVs:")
    print(f"  - {all_path}")
    print(f"  - {top_path}")
    print(f"Rows(all): {len(clusters_df)} | Rows(top): {len(top_df)}")
    print(f"hot_threshold(abs_err @ q={args.color_max_quantile:.3f}): {hot_threshold:.6f}")


if __name__ == "__main__":
    main()
