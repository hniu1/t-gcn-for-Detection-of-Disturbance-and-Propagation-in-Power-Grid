#!/usr/bin/env python3
"""Task 1: find connected sensor clusters with simultaneous high freq and RoCoF deviations."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = "results_masked_cov80_full_scaled"
DEFAULT_OUTPUT_DIR = "downstream_tasks/task1_regional_anomalies/outputs"


def load_split_arrays(results_dir: Path, split_name: str, use_original: bool) -> tuple[np.ndarray, np.ndarray, str]:
    suffix = "_original" if use_original else ""
    pred_path = results_dir / f"{split_name}_predictions{suffix}.npy"
    true_path = results_dir / f"{split_name}_targets{suffix}.npy"
    if pred_path.exists() and true_path.exists():
        return np.load(pred_path), np.load(true_path), "original" if use_original else "scaled"
    return np.load(results_dir / f"{split_name}_predictions.npy"), np.load(results_dir / f"{split_name}_targets.npy"), "scaled"


def reduce_horizon(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.mean(arr, axis=1)
    if mode == "first":
        return arr[:, 0, :, :]
    return arr[:, -1, :, :]


def load_run_config(results_dir: Path) -> dict:
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {results_dir}")
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


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
        files = list(data_dir.glob(f"{sensor_id}-*.parquet"))
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
            f"Timestamp reconstruction mismatch for {split_name}: expected {num_samples} samples, got {len(valid_starts)}"
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
    """Best-effort sensor name lookup from data filenames, then metadata; fallback to generic labels."""
    name_map = {int(sid): f"Sensor-{int(sid)}" for sid in sensor_ids.tolist()}
    try:
        cfg = load_run_config(results_dir)
        data_dir = cfg.get("data_dir")
        if data_dir:
            data_path = Path(data_dir)
            for sid in sensor_ids.tolist():
                matches = sorted(data_path.glob(f"**/{int(sid)}-*.parquet"))
                if not matches:
                    continue
                stem = matches[0].stem
                parts = stem.split("-", 2)
                if len(parts) >= 2 and parts[1].strip():
                    # Example: 1144-UsWaSpokane1144-20240101.parquet -> UsWaSpokane1144
                    name_map[int(sid)] = parts[1].strip()

        meta_file = cfg.get("metadata_file")
        if meta_file:
            meta_path = Path(meta_file)
            if meta_path.exists():
                df = pd.read_excel(meta_path)
                if "FDRID" in df.columns:
                    id_col = "FDRID"
                    if "GridName" in df.columns:
                        name_col = "GridName"
                    elif "Filename" in df.columns:
                        name_col = "Filename"
                    else:
                        name_col = None
                    if name_col is not None:
                        tmp = df[[id_col, name_col]].dropna()
                        for row in tmp.itertuples(index=False):
                            sid = int(getattr(row, id_col))
                            label = str(getattr(row, name_col)).strip()
                            if label and name_map.get(sid, "").startswith("Sensor-"):
                                name_map[sid] = label
    except Exception:
        pass
    return name_map


def build_cluster_plot(
    out_png: Path,
    event: dict,
    pred_freq: np.ndarray,
    true_freq: np.ndarray,
    pred_rocof: np.ndarray,
    true_rocof: np.ndarray,
    sensor_ids_all: np.ndarray,
    sensor_name_map: dict[int, str],
    timestamps: list[datetime],
    window_radius: int,
) -> None:
    t = int(event["time_idx"])
    sensors = [int(x) for x in event["sensor_indices"].split(";") if x != ""]
    if not sensors:
        return

    t0 = max(0, t - window_radius)
    t1 = min(pred_freq.shape[0] - 1, t + window_radius)
    x = np.arange(t0, t1 + 1)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.0), sharex=True)
    cmap = plt.cm.get_cmap("tab10", max(2, len(sensors)))
    sensor_labels = []
    for k, idx in enumerate(sensors):
        sid = int(sensor_ids_all[idx])
        sname = sensor_name_map.get(sid, f"Sensor-{sid}")
        sensor_labels.append(sname)
        color = cmap(k)

        freq_true_line = true_freq[t0:t1 + 1, idx]
        freq_pred_line = pred_freq[t0:t1 + 1, idx]
        rocof_true_line = true_rocof[t0:t1 + 1, idx]
        rocof_pred_line = pred_rocof[t0:t1 + 1, idx]
        freq_dev_line = np.abs(freq_pred_line - freq_true_line)
        rocof_dev_line = np.abs(rocof_pred_line - rocof_true_line)

        axes[0].plot(x, freq_true_line, linewidth=1.8, color=color, label=f"{sid} true")
        axes[0].plot(x, freq_pred_line, linewidth=1.2, linestyle="--", color=color, label=f"{sid} pred")
        axes[1].plot(x, rocof_true_line, linewidth=1.8, color=color, label=f"{sid} true")
        axes[1].plot(x, rocof_pred_line, linewidth=1.2, linestyle="--", color=color, label=f"{sid} pred")
        axes[2].plot(x, freq_dev_line, linewidth=1.6, color=color, label=f"{sid} |freq err|")
        axes[2].plot(x, rocof_dev_line, linewidth=1.1, linestyle=":", color=color, label=f"{sid} |rocof err|")

    axes[0].axvline(t, linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("freq_dev")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].axvline(t, linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("rocof")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)

    axes[2].axvline(t, linestyle=":", linewidth=1.2)
    axes[2].set_ylabel("Deviation")
    axes[2].set_xlabel("Sample index")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8, ncol=2)

    peak_ts = timestamps[t].strftime("%Y-%m-%d %H:%M:%S")
    sensor_text = ", ".join(sensor_labels)
    fig.suptitle(
        f"Cluster {event['cluster_id']} | time_idx={t} | peak_time={peak_ts}\nSensors: {sensor_text}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: connected clusters with simultaneous high freq and RoCoF deviations")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split_name", type=str, default="test", choices=["test", "pred"])
    parser.add_argument("--sensor_order_file", type=str, default="results/sensor_order.npy")
    parser.add_argument("--adjacency_file", type=str, default="results/A_geo.npy")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon_reduce", type=str, default="mean", choices=["mean", "first", "last"])
    parser.add_argument("--use_original", action="store_true", default=True)
    parser.add_argument("--freq_quantile", type=float, default=0.98)
    parser.add_argument("--rocof_quantile", type=float, default=0.98)
    parser.add_argument("--edge_quantile", type=float, default=0.85)
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--max_clusters", type=int, default=30)
    parser.add_argument("--window_radius", type=int, default=20)
    parser.add_argument(
        "--true_rocof_min_abs",
        type=float,
        default=5e-4,
        help="Drop clusters when max |true_rocof| across sensors at event time is <= this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "cluster_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    pred, true, data_variant = load_split_arrays(results_dir, args.split_name, args.use_original)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (samples, horizon, sensors, features), got {pred.ndim}D")

    pred_reduced = reduce_horizon(pred, args.horizon_reduce)
    true_reduced = reduce_horizon(true, args.horizon_reduce)

    # Model outputs include freq_dev and volt_dev. RoCoF is estimated from temporal derivative of freq_dev.
    freq_idx = 0
    pred_freq = pred_reduced[:, :, freq_idx]
    true_freq = true_reduced[:, :, freq_idx]

    pred_rocof = np.zeros_like(pred_freq)
    true_rocof = np.zeros_like(true_freq)
    pred_rocof[1:] = pred_freq[1:] - pred_freq[:-1]
    true_rocof[1:] = true_freq[1:] - true_freq[:-1]

    freq_dev_abs = np.abs(pred_freq - true_freq)
    rocof_dev_abs = np.abs(pred_rocof - true_rocof)

    freq_thr = float(np.quantile(freq_dev_abs, min(max(args.freq_quantile, 0.5), 0.9999)))
    rocof_thr = float(np.quantile(rocof_dev_abs, min(max(args.rocof_quantile, 0.5), 0.9999)))
    active = (freq_dev_abs >= freq_thr) & (rocof_dev_abs >= rocof_thr)

    sensor_ids = np.load(Path(args.sensor_order_file)).astype(int)
    if sensor_ids.shape[0] != pred_freq.shape[1]:
        raise ValueError(f"Sensor order length mismatch: expected {pred_freq.shape[1]}, got {sensor_ids.shape[0]}")

    A = np.load(Path(args.adjacency_file)).astype(float)
    if A.shape[0] != sensor_ids.shape[0] or A.shape[1] != sensor_ids.shape[0]:
        raise ValueError(f"Adjacency shape mismatch: expected {(sensor_ids.shape[0], sensor_ids.shape[0])}, got {A.shape}")
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    pos = A[A > 0]
    edge_thr = float(np.quantile(pos, min(max(args.edge_quantile, 0.01), 0.9999))) if pos.size > 0 else 0.0
    edge_mask_global = A >= edge_thr
    sensor_name_map = build_sensor_name_map(results_dir, sensor_ids)

    timestamps = reconstruct_split_timestamps(results_dir, args.split_name, pred.shape[0], args.horizon_reduce)

    clusters = []
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

            # Drop clusters whose true values have no cross-sensor deviation at this timestamp.
            true_freq_vals = true_freq[t, nodes]
            true_rocof_vals = true_rocof[t, nodes]

            # Drop clusters if true RoCoF is too small for all sensors at this timestamp.
            if float(np.max(np.abs(true_rocof_vals))) <= float(args.true_rocof_min_abs):
                continue

            true_freq_spread = float(np.max(true_freq_vals) - np.min(true_freq_vals))
            true_rocof_spread = float(np.max(true_rocof_vals) - np.min(true_rocof_vals))
            if true_freq_spread <= 1e-12 and true_rocof_spread <= 1e-12:
                continue

            freq_mean = float(np.mean(freq_dev_abs[t, nodes]))
            rocof_mean = float(np.mean(rocof_dev_abs[t, nodes]))
            # Remove clusters if either deviation channel is effectively zero.
            if freq_mean <= 1e-12 or rocof_mean <= 1e-12:
                continue

            magnitude = float(freq_mean + rocof_mean)
            score = float(len(nodes) * magnitude)
            cluster_sensor_ids = [int(sensor_ids[i]) for i in nodes.tolist()]
            cluster_sensor_names = [sensor_name_map.get(sid, f"Sensor-{sid}") for sid in cluster_sensor_ids]
            clusters.append(
                {
                    "cluster_id": int(cluster_id),
                    "time_idx": int(t),
                    "timestamp": timestamps[t].strftime("%Y-%m-%d %H:%M:%S"),
                    "cluster_size": int(len(nodes)),
                    "score": score,
                    "magnitude": magnitude,
                    "mean_freq_dev": freq_mean,
                    "mean_rocof_dev": rocof_mean,
                    "sensor_ids": ";".join(str(x) for x in cluster_sensor_ids),
                    "sensor_names": ";".join(cluster_sensor_names),
                    "sensor_indices": ";".join(str(int(x)) for x in nodes.tolist()),
                }
            )
            cluster_id += 1

    clusters_df = pd.DataFrame(clusters)
    if clusters_df.empty:
        raise ValueError("No clusters detected. Try lower quantiles or lower min_cluster_size.")

    clusters_df = clusters_df.sort_values(["score", "cluster_size", "magnitude"], ascending=[False, False, False]).reset_index(drop=True)
    selected_df = clusters_df.head(args.max_clusters).copy()

    participation: dict[int, int] = {}
    for row in selected_df.itertuples(index=False):
        for sid in [int(x) for x in str(row.sensor_ids).split(";") if x != ""]:
            participation[sid] = participation.get(sid, 0) + 1
    sensor_participation = pd.DataFrame(
        [
            {
                "sensor_id": int(k),
                "sensor_name": sensor_name_map.get(int(k), f"Sensor-{int(k)}"),
                "cluster_count": int(v),
            }
            for k, v in participation.items()
        ]
    ).sort_values("cluster_count", ascending=False)

    for row in selected_df.itertuples(index=False):
        out_png = plot_dir / f"cluster_{int(row.cluster_id)}_t{int(row.time_idx)}.png"
        build_cluster_plot(
            out_png=out_png,
            event=row._asdict(),
            pred_freq=pred_freq,
            true_freq=true_freq,
            pred_rocof=pred_rocof,
            true_rocof=true_rocof,
            sensor_ids_all=sensor_ids,
            sensor_name_map=sensor_name_map,
            timestamps=timestamps,
            window_radius=args.window_radius,
        )

    clusters_df.to_csv(out_dir / "clusters_all.csv", index=False)
    selected_df.to_csv(out_dir / "clusters_top.csv", index=False)
    sensor_participation.to_csv(out_dir / "sensor_participation.csv", index=False)

    threshold_info = {
        "freq_threshold": freq_thr,
        "rocof_threshold": rocof_thr,
        "edge_threshold": edge_thr,
        "true_rocof_min_abs": float(args.true_rocof_min_abs),
        "freq_quantile": float(args.freq_quantile),
        "rocof_quantile": float(args.rocof_quantile),
        "edge_quantile": float(args.edge_quantile),
    }
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as fp:
        json.dump(threshold_info, fp, indent=2)

    summary = {
        "task": "Task 1: connected clusters with simultaneous high freq and RoCoF deviations",
        "results_dir": str(results_dir),
        "split_name": args.split_name,
        "data_variant": data_variant,
        "horizon_reduce": args.horizon_reduce,
        "rocof_note": "RoCoF estimated as first difference of reduced freq_dev over sample time.",
        "num_samples": int(pred.shape[0]),
        "num_sensors": int(sensor_ids.shape[0]),
        "clusters_detected": int(len(clusters_df)),
        "clusters_selected": int(len(selected_df)),
        "top_clusters": selected_df.head(10).to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Saved Task 1 outputs:")
    for name in [
        "clusters_all.csv",
        "clusters_top.csv",
        "sensor_participation.csv",
        "thresholds.json",
        "summary.json",
    ]:
        print(f"  - {out_dir / name}")
    print(f"  - {plot_dir}/*.png")


if __name__ == "__main__":
    main()
