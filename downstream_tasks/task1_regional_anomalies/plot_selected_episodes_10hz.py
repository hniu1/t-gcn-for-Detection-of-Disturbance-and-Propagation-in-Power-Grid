#!/usr/bin/env python3
"""Plot the selected map-cluster episodes using every 10 Hz horizon value."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collapse_map_clusters_and_plot import (
    build_latlon_map,
    build_name_map,
    load_run_config,
    load_split_arrays,
    parse_sensor_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot previously selected cluster episodes at the original 10 Hz horizon resolution."
    )
    parser.add_argument("--results_dir", default="results_masked_cov80_full_scaled")
    parser.add_argument("--split_name", default="pred", choices=["test", "pred"])
    parser.add_argument("--use_original", action="store_true", default=True)
    parser.add_argument("--feature_idx", type=int, default=0)
    parser.add_argument(
        "--episodes_csv",
        default=(
            "downstream_tasks/task1_regional_anomalies/outputs/"
            "map_cluster_episodes_real_time/map_freq_hot_cluster_episodes_selected.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "downstream_tasks/task1_regional_anomalies/outputs/"
            "map_cluster_episodes_real_time_10hz"
        ),
    )
    parser.add_argument(
        "--context_radius",
        type=int,
        default=25,
        help="Context on each side in prediction samples (one sample = one second for the current run).",
    )
    return parser.parse_args()


def reconstruct_horizon_timestamps(
    results_dir: Path,
    split_name: str,
    num_samples: int,
) -> np.ndarray:
    """Return one timestamp for every [sample, horizon] tensor value."""
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
        frame = pd.read_parquet(files[0], columns=["ReceivedTime"])
        if frame.empty:
            continue
        received_time = pd.to_datetime(frame["ReceivedTime"])
        sensor_start_time[sensor_id] = received_time.min()
        sensor_lengths[sensor_id] = len(received_time)

    global_start = min(sensor_start_time.values())
    offsets = {
        sensor_id: int(
            round(
                (sensor_start_time[sensor_id] - global_start).total_seconds()
                * sample_rate_hz
            )
        )
        for sensor_id in sensor_start_time
    }
    max_len = max(
        offsets[sensor_id] + sensor_lengths[sensor_id]
        for sensor_id in sensor_start_time
    )
    ordered_ids = [sid for sid in sensor_order if sid in sensor_start_time]
    observed_mask = np.zeros((max_len, len(ordered_ids)), dtype=np.float32)
    for sensor_idx, sensor_id in enumerate(ordered_ids):
        start_idx = max(offsets[sensor_id], 0)
        end_idx = min(start_idx + sensor_lengths[sensor_id], max_len)
        observed_mask[start_idx:end_idx, sensor_idx] = 1.0

    filtered_indices = np.flatnonzero(
        observed_mask.mean(axis=1) >= min_step_coverage
    )
    total_filtered = len(filtered_indices)
    train_end = int(total_filtered * float(config["train_frac"]))
    val_end = train_end + int(total_filtered * float(config["val_frac"]))
    test_end = val_end + int(total_filtered * float(config["test_frac"]))
    pred_end = min(
        test_end + int(total_filtered * float(config["pred_frac"])),
        total_filtered,
    )
    split_bounds = {
        "test": (val_end, test_end),
        "pred": (
            test_end,
            pred_end if float(config["pred_frac"]) > 0 else total_filtered,
        ),
    }
    split_start, split_stop = split_bounds[split_name]
    split_len = split_stop - split_start
    valid_starts = list(range(0, split_len - tin - horizon + 1, split_stride))
    if len(valid_starts) != num_samples:
        raise ValueError(
            f"Timestamp mismatch: arrays contain {num_samples} samples, "
            f"but reconstruction found {len(valid_starts)}"
        )

    timestamps = np.empty((num_samples, horizon), dtype=object)
    for sample_idx, start_idx in enumerate(valid_starts):
        output_positions = filtered_indices[
            split_start + start_idx + tin : split_start + start_idx + tin + horizon
        ]
        timestamps[sample_idx] = [
            global_start.to_pydatetime()
            + timedelta(seconds=int(position) / sample_rate_hz)
            for position in output_positions
        ]
    return timestamps


def plot_episode_10hz(
    out_png: Path,
    episode: pd.Series,
    pred_freq: np.ndarray,
    true_freq: np.ndarray,
    horizon_timestamps: np.ndarray,
    sensor_order: np.ndarray,
    adjacency: np.ndarray,
    latlon_map: dict[int, tuple[float, float]],
    name_map: dict[int, str],
    context_radius: int,
) -> None:
    sensor_ids = parse_sensor_ids(str(episode.sensor_ids))
    sensor_indices = [int(np.where(sensor_order == sid)[0][0]) for sid in sensor_ids]
    t0 = max(0, int(episode.start_time_idx) - context_radius)
    t1 = min(pred_freq.shape[0] - 1, int(episode.end_time_idx) + context_radius)

    x = horizon_timestamps[t0 : t1 + 1].reshape(-1)
    pred_window = pred_freq[t0 : t1 + 1].reshape(-1, pred_freq.shape[2])
    true_window = true_freq[t0 : t1 + 1].reshape(-1, true_freq.shape[2])

    peak_sample = int(episode.peak_time_idx)
    peak_horizon_err = np.abs(
        pred_freq[peak_sample][:, sensor_indices]
        - true_freq[peak_sample][:, sensor_indices]
    )
    peak_offset = int(np.argmax(np.mean(peak_horizon_err, axis=1)))
    peak_timestamp = horizon_timestamps[peak_sample, peak_offset]
    peak_err = peak_horizon_err[peak_offset]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [1.2, 1.5, 1.3]},
        sharex=False,
    )
    graph_ax, freq_ax, dev_ax = axes

    for i_local, sid_i in enumerate(sensor_ids):
        if sid_i not in latlon_map:
            continue
        lat_i, lon_i = latlon_map[sid_i]
        for j_local, sid_j in enumerate(sensor_ids):
            if j_local <= i_local or sid_j not in latlon_map:
                continue
            weight = float(adjacency[sensor_indices[i_local], sensor_indices[j_local]])
            if weight <= 0:
                continue
            lat_j, lon_j = latlon_map[sid_j]
            graph_ax.plot(
                [lon_i, lon_j],
                [lat_i, lat_j],
                color="lightsteelblue",
                linewidth=0.8 + 1.5 * min(weight, 1.0),
                alpha=0.8,
            )

    vmax = max(float(np.max(peak_err)), 1e-6)
    for sid, err in zip(sensor_ids, peak_err.tolist()):
        if sid not in latlon_map:
            continue
        lat, lon = latlon_map[sid]
        graph_ax.scatter(
            lon,
            lat,
            s=180,
            color=plt.cm.YlOrRd(min(err / vmax, 1.0)),
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        graph_ax.text(lon, lat, f" {sid}", fontsize=8, va="center")
    graph_ax.set_title("Cluster Graph (node color = peak 10 Hz |freq err|)")
    graph_ax.set_xlabel("Longitude")
    graph_ax.set_ylabel("Latitude")
    graph_ax.grid(alpha=0.2)

    cmap = plt.get_cmap("tab10")
    for color_idx, (sid, sensor_idx) in enumerate(zip(sensor_ids, sensor_indices)):
        color = cmap(color_idx % 10)
        label = name_map.get(sid, f"Sensor-{sid}")
        true_line = true_window[:, sensor_idx]
        pred_line = pred_window[:, sensor_idx]
        freq_ax.plot(x, true_line, color=color, linewidth=1.5, label=f"{sid} true")
        freq_ax.plot(
            x,
            pred_line,
            color=color,
            linewidth=1.0,
            linestyle="--",
            label=f"{sid} pred",
        )
        dev_ax.plot(
            x,
            np.abs(pred_line - true_line),
            color=color,
            linewidth=1.3,
            label=f"{sid} |err| {label}",
        )

    episode_start = horizon_timestamps[int(episode.start_time_idx), 0]
    episode_end = horizon_timestamps[int(episode.end_time_idx), -1]
    for axis in (freq_ax, dev_ax):
        axis.axvspan(episode_start, episode_end, color="gold", alpha=0.14)
        axis.axvline(peak_timestamp, linestyle=":", color="black", linewidth=1.2)
        axis.grid(alpha=0.25)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    freq_ax.set_ylabel("freq_dev")
    dev_ax.set_ylabel("|pred-true|")
    dev_ax.set_xlabel(f"Time on {peak_timestamp:%Y-%m-%d} (10 Hz values)")
    freq_ax.legend(fontsize=7, ncol=3)
    dev_ax.legend(fontsize=7, ncol=2)

    sensor_title = ", ".join(
        f"{sid}({name_map.get(sid, f'Sensor-{sid}')})" for sid in sensor_ids
    )
    fig.suptitle(
        f"Episode {int(episode.episode_id)} | 10 Hz | size={int(episode.cluster_size)} | "
        f"peak_mean(1 Hz)={float(episode.peak_mean_abs_err):.4f} | "
        f"window=[{episode_start:%H:%M:%S}, {episode_end:%H:%M:%S}]\n"
        f"Sensors: {sensor_title}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    episodes_csv = Path(args.episodes_csv)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "episode_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_csv)
    if episodes.empty:
        raise ValueError(f"Selected episodes CSV is empty: {episodes_csv}")

    pred, true = load_split_arrays(results_dir, args.split_name, args.use_original)
    if pred.ndim != 4 or true.ndim != 4:
        raise ValueError(
            f"Expected [samples, horizon, sensors, features] arrays; got {pred.shape} and {true.shape}"
        )
    pred_freq = pred[:, :, :, args.feature_idx]
    true_freq = true[:, :, :, args.feature_idx]
    timestamps = reconstruct_horizon_timestamps(
        results_dir, args.split_name, pred.shape[0]
    )

    sensor_order = np.load("results/sensor_order.npy").astype(int)
    adjacency = np.load("results/A_geo.npy").astype(float)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    name_map = build_name_map(results_dir, sensor_order)
    latlon_map = build_latlon_map(results_dir)

    for row in episodes.itertuples(index=False):
        out_png = plot_dir / (
            f"episode_{int(row.episode_id)}_t{int(row.peak_time_idx)}.png"
        )
        plot_episode_10hz(
            out_png=out_png,
            episode=pd.Series(row._asdict()),
            pred_freq=pred_freq,
            true_freq=true_freq,
            horizon_timestamps=timestamps,
            sensor_order=sensor_order,
            adjacency=adjacency,
            latlon_map=latlon_map,
            name_map=name_map,
            context_radius=args.context_radius,
        )

    episodes.to_csv(out_dir / episodes_csv.name, index=False)
    summary = {
        "source_episodes_csv": str(episodes_csv),
        "selected_for_plot": int(len(episodes)),
        "results_dir": str(results_dir),
        "split_name": args.split_name,
        "sample_rate_hz": float(load_run_config(results_dir)["sample_rate_hz"]),
        "horizon_values_plotted": int(pred.shape[1]),
        "context_radius_samples": int(args.context_radius),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved {len(episodes)} 10 Hz episode plots to {plot_dir}")


if __name__ == "__main__":
    main()
