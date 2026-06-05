#!/usr/bin/env python3
"""Collapse repeated map-hot clusters and plot graph + freq/dev traces per episode."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collapse repeated map-hot clusters and plot episode details")
    parser.add_argument("--results_dir", type=str, default="results_masked_cov80_full_scaled")
    parser.add_argument("--split_name", type=str, default="pred", choices=["test", "pred"])
    parser.add_argument("--use_original", action="store_true", default=True)
    parser.add_argument("--horizon_reduce", type=str, default="mean", choices=["mean", "first", "last"])
    parser.add_argument("--feature_idx", type=int, default=0)
    parser.add_argument(
        "--input_clusters_csv",
        type=str,
        default="downstream_tasks/task1_regional_anomalies/outputs/map_freq_hot_clusters_all.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="downstream_tasks/task1_regional_anomalies/outputs/map_cluster_episodes",
    )
    parser.add_argument("--episode_gap", type=int, default=10)
    parser.add_argument("--auto_increase_cutoff", action="store_true", default=True)
    parser.add_argument("--target_max_episodes", type=int, default=800)
    parser.add_argument("--candidate_gaps", type=str, default="10,20,30,60,120")
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--min_peak_mean_abs_err", type=float, default=0.0)
    parser.add_argument("--max_plots", type=int, default=120)
    parser.add_argument("--context_radius", type=int, default=20)
    parser.add_argument(
        "--peak_exclusion_window",
        type=int,
        default=120,
        help="Skip selecting a new episode if its peak is within this many steps of a selected one and they share sensors.",
    )
    parser.add_argument(
        "--window_overlap_mode",
        type=str,
        default="sensor_overlap",
        choices=["sensor_overlap", "global"],
        help="sensor_overlap: suppress only if sensor sets overlap; global: suppress any peak in window.",
    )
    return parser.parse_args()


def load_run_config(results_dir: Path) -> dict:
    with open(results_dir / "config.json", "r", encoding="utf-8") as fp:
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
        raise ValueError(f"Timestamp mismatch: expected {num_samples}, got {len(valid_starts)}")

    horizon_offset = tin + horizon - 1 if horizon_reduce == "last" else tin
    timestamps: list[datetime] = []
    for start_idx in valid_starts:
        filtered_idx = filtered_indices[split_start + start_idx + horizon_offset]
        timestamps.append(global_start.to_pydatetime() + timedelta(seconds=filtered_idx / sample_rate_hz))
    return timestamps


def parse_sensor_ids(sensor_ids_text: str) -> list[int]:
    return [int(x) for x in str(sensor_ids_text).split(";") if str(x) != ""]


def canonical_sensor_set(sensor_ids_text: str) -> str:
    ids = sorted(parse_sensor_ids(sensor_ids_text))
    return ";".join(str(x) for x in ids)


def collapse_to_episodes(df: pd.DataFrame, episode_gap: int) -> pd.DataFrame:
    rows: list[dict] = []
    episode_id = 0
    for sensor_set, group in df.sort_values(["sensor_set", "time_idx"]).groupby("sensor_set", sort=False):
        g = group.sort_values("time_idx").reset_index(drop=True)
        start = 0
        t = g["time_idx"].to_numpy(dtype=int)
        for i in range(1, len(g) + 1):
            boundary = i == len(g) or (t[i] - t[i - 1]) > episode_gap
            if not boundary:
                continue
            seg = g.iloc[start:i]
            peak_idx = int(seg["mean_abs_err"].idxmax())
            peak_row = seg.loc[peak_idx]
            rows.append(
                {
                    "episode_id": episode_id,
                    "sensor_set": sensor_set,
                    "cluster_size": int(peak_row["cluster_size"]),
                    "start_time_idx": int(seg["time_idx"].min()),
                    "end_time_idx": int(seg["time_idx"].max()),
                    "peak_time_idx": int(peak_row["time_idx"]),
                    "n_frames": int(len(seg)),
                    "duration_steps": int(seg["time_idx"].max() - seg["time_idx"].min() + 1),
                    "peak_mean_abs_err": float(seg["mean_abs_err"].max()),
                    "peak_max_abs_err": float(seg["max_abs_err"].max()),
                    "sensor_ids": str(peak_row["sensor_ids"]),
                    "sensor_names": str(peak_row.get("sensor_names", "")),
                    "sensor_indices": str(peak_row["sensor_indices"]),
                    "peak_timestamp": str(peak_row["timestamp"]),
                }
            )
            episode_id += 1
            start = i
    return pd.DataFrame(rows)


def choose_episode_gap(df: pd.DataFrame, initial_gap: int, candidate_gaps: list[int], target_max_episodes: int) -> tuple[int, pd.DataFrame]:
    ordered = sorted(set([g for g in candidate_gaps if g >= initial_gap] + [initial_gap]))
    chosen_gap = ordered[-1]
    chosen_df = collapse_to_episodes(df, chosen_gap)
    for gap in ordered:
        eps = collapse_to_episodes(df, gap)
        chosen_gap, chosen_df = gap, eps
        if len(eps) <= target_max_episodes:
            break
    return chosen_gap, chosen_df


def select_with_peak_window(episodes: pd.DataFrame, max_plots: int, window: int, overlap_mode: str) -> pd.DataFrame:
    if episodes.empty or max_plots <= 0:
        return episodes.head(0).copy()
    selected_rows: list[dict] = []
    selected_peaks: list[int] = []
    selected_sets: list[set[int]] = []

    for row in episodes.itertuples(index=False):
        candidate_peak = int(row.peak_time_idx)
        candidate_set = set(parse_sensor_ids(str(row.sensor_ids)))
        blocked = False
        for prev_peak, prev_set in zip(selected_peaks, selected_sets):
            if abs(candidate_peak - prev_peak) > window:
                continue
            if overlap_mode == "global":
                blocked = True
                break
            if candidate_set.intersection(prev_set):
                blocked = True
                break
        if blocked:
            continue
        selected_rows.append(row._asdict())
        selected_peaks.append(candidate_peak)
        selected_sets.append(candidate_set)
        if len(selected_rows) >= max_plots:
            break
    return pd.DataFrame(selected_rows)


def build_name_map(results_dir: Path, sensor_ids: np.ndarray) -> dict[int, str]:
    cfg = load_run_config(results_dir)
    data_dir = Path(cfg["data_dir"])
    name_map = {int(sid): f"Sensor-{int(sid)}" for sid in sensor_ids.tolist()}
    for sid in sensor_ids.tolist():
        matches = sorted(data_dir.glob(f"**/{int(sid)}-*.parquet"))
        if not matches:
            continue
        parts = matches[0].stem.split("-", 2)
        if len(parts) >= 2 and parts[1].strip():
            name_map[int(sid)] = parts[1].strip()
    return name_map


def build_latlon_map(results_dir: Path) -> dict[int, tuple[float, float]]:
    cfg = load_run_config(results_dir)
    meta_path = Path(cfg["metadata_file"])
    if meta_path.suffix.lower() in {".xls", ".xlsx"}:
        md = pd.read_excel(meta_path)
    else:
        md = pd.read_csv(meta_path)
    if "FDRID" in md.columns:
        id_col = "FDRID"
    elif "SensorID" in md.columns:
        id_col = "SensorID"
    else:
        raise ValueError("Metadata file missing sensor id column")
    out: dict[int, tuple[float, float]] = {}
    for row in md.itertuples(index=False):
        sid = int(getattr(row, id_col))
        lat = float(getattr(row, "Latitude"))
        lon = float(getattr(row, "Longitude"))
        out[sid] = (lat, lon)
    return out


def plot_episode(
    out_png: Path,
    episode: pd.Series,
    pred_freq: np.ndarray,
    true_freq: np.ndarray,
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
    x = np.arange(t0, t1 + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1.2, 1.5, 1.3]}, sharex=False)
    graph_ax, freq_ax, dev_ax = axes

    # Graph panel
    for i_local, sid_i in enumerate(sensor_ids):
        if sid_i not in latlon_map:
            continue
        lat_i, lon_i = latlon_map[sid_i]
        for j_local, sid_j in enumerate(sensor_ids):
            if j_local <= i_local or sid_j not in latlon_map:
                continue
            ai = sensor_indices[i_local]
            aj = sensor_indices[j_local]
            w = float(adjacency[ai, aj])
            if w <= 0:
                continue
            lat_j, lon_j = latlon_map[sid_j]
            graph_ax.plot([lon_i, lon_j], [lat_i, lat_j], color="lightsteelblue", linewidth=0.8 + 1.5 * min(w, 1.0), alpha=0.8)

    peak_t = int(episode.peak_time_idx)
    peak_err = []
    for idx in sensor_indices:
        peak_err.append(float(abs(pred_freq[peak_t, idx] - true_freq[peak_t, idx])))
    peak_err_arr = np.array(peak_err, dtype=float)
    vmax = max(float(np.max(peak_err_arr)), 1e-6)
    for sid, err in zip(sensor_ids, peak_err_arr.tolist()):
        if sid not in latlon_map:
            continue
        lat, lon = latlon_map[sid]
        color = plt.cm.YlOrRd(min(err / vmax, 1.0))
        graph_ax.scatter(lon, lat, s=180, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        graph_ax.text(lon, lat, f" {sid}", fontsize=8, va="center")
    graph_ax.set_title("Cluster Graph (node color = peak |freq err|)")
    graph_ax.set_xlabel("Longitude")
    graph_ax.set_ylabel("Latitude")
    graph_ax.grid(alpha=0.2)

    # Time-series panels
    cmap = plt.get_cmap("tab10")
    for k, (sid, idx) in enumerate(zip(sensor_ids, sensor_indices)):
        color = cmap(k % 10)
        label = name_map.get(sid, f"Sensor-{sid}")
        true_line = true_freq[t0:t1 + 1, idx]
        pred_line = pred_freq[t0:t1 + 1, idx]
        dev_line = np.abs(pred_line - true_line)
        freq_ax.plot(x, true_line, color=color, linewidth=1.8, label=f"{sid} true")
        freq_ax.plot(x, pred_line, color=color, linewidth=1.1, linestyle="--", label=f"{sid} pred")
        dev_ax.plot(x, dev_line, color=color, linewidth=1.6, label=f"{sid} |err| {label}")

    for ax in [freq_ax, dev_ax]:
        ax.axvspan(int(episode.start_time_idx), int(episode.end_time_idx), color="gold", alpha=0.14)
        ax.axvline(int(episode.peak_time_idx), linestyle=":", color="black", linewidth=1.2)
        ax.grid(alpha=0.25)
    freq_ax.set_ylabel("freq_dev")
    dev_ax.set_ylabel("|pred-true|")
    dev_ax.set_xlabel("Sample index")
    freq_ax.legend(fontsize=7, ncol=3)
    dev_ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"Episode {int(episode.episode_id)} | size={int(episode.cluster_size)} | peak_mean={float(episode.peak_mean_abs_err):.4f} | "
        f"window=[{int(episode.start_time_idx)},{int(episode.end_time_idx)}]",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    in_csv = Path(args.input_clusters_csv)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "episode_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input clusters CSV: {in_csv}")

    raw = pd.read_csv(in_csv)
    if raw.empty:
        raise ValueError("Input clusters CSV is empty")
    raw["sensor_set"] = raw["sensor_ids"].map(canonical_sensor_set)

    candidate_gaps = [int(x) for x in args.candidate_gaps.split(",") if x.strip()]
    if args.auto_increase_cutoff:
        used_gap, episodes = choose_episode_gap(raw, args.episode_gap, candidate_gaps, args.target_max_episodes)
    else:
        used_gap, episodes = args.episode_gap, collapse_to_episodes(raw, args.episode_gap)

    episodes = episodes[(episodes["cluster_size"] >= args.min_cluster_size) & (episodes["peak_mean_abs_err"] >= args.min_peak_mean_abs_err)]
    # Rank primarily by error magnitude, with a light size bonus to avoid over-prioritizing large but weaker clusters.
    episodes["rank_score"] = episodes["peak_mean_abs_err"] * (1.0 + 0.15 * np.maximum(episodes["cluster_size"] - 2, 0))
    episodes = episodes.sort_values(["rank_score", "peak_max_abs_err", "n_frames"], ascending=[False, False, False]).reset_index(drop=True)
    episodes["episode_id"] = np.arange(len(episodes), dtype=int)

    episodes.to_csv(out_dir / "map_freq_hot_cluster_episodes_all.csv", index=False)
    top_eps = select_with_peak_window(
        episodes,
        max_plots=args.max_plots,
        window=max(0, args.peak_exclusion_window),
        overlap_mode=args.window_overlap_mode,
    )
    top_eps = top_eps.reset_index(drop=True)
    top_eps["episode_id"] = np.arange(len(top_eps), dtype=int)
    top_eps.to_csv(out_dir / "map_freq_hot_cluster_episodes_selected.csv", index=False)

    pred, true = load_split_arrays(results_dir, args.split_name, args.use_original)
    pred_r = reduce_horizon(pred, args.horizon_reduce)
    true_r = reduce_horizon(true, args.horizon_reduce)
    pred_freq = pred_r[:, :, args.feature_idx]
    true_freq = true_r[:, :, args.feature_idx]

    sensor_order = np.load(Path("results/sensor_order.npy")).astype(int)
    adjacency = np.load(Path("results/A_geo.npy")).astype(float)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    name_map = build_name_map(results_dir, sensor_order)
    latlon_map = build_latlon_map(results_dir)

    for row in top_eps.itertuples(index=False):
        out_png = plot_dir / f"episode_{int(row.episode_id)}_t{int(row.peak_time_idx)}.png"
        plot_episode(
            out_png=out_png,
            episode=pd.Series(row._asdict()),
            pred_freq=pred_freq,
            true_freq=true_freq,
            sensor_order=sensor_order,
            adjacency=adjacency,
            latlon_map=latlon_map,
            name_map=name_map,
            context_radius=args.context_radius,
        )

    summary = {
        "input_rows": int(len(raw)),
        "episodes_after_collapse": int(len(episodes)),
        "selected_for_plot": int(len(top_eps)),
        "used_episode_gap": int(used_gap),
        "peak_exclusion_window": int(args.peak_exclusion_window),
        "window_overlap_mode": str(args.window_overlap_mode),
        "auto_increase_cutoff": bool(args.auto_increase_cutoff),
        "target_max_episodes": int(args.target_max_episodes),
        "min_cluster_size": int(args.min_cluster_size),
        "min_peak_mean_abs_err": float(args.min_peak_mean_abs_err),
        "split_name": args.split_name,
        "results_dir": str(results_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Saved:")
    print(f"  - {out_dir / 'map_freq_hot_cluster_episodes_all.csv'}")
    print(f"  - {out_dir / 'map_freq_hot_cluster_episodes_selected.csv'}")
    print(f"  - {out_dir / 'summary.json'}")
    print(f"  - {plot_dir}/*.png")
    print(f"Episodes: {len(episodes)} | Selected plots: {len(top_eps)} | Used gap: {used_gap}")


if __name__ == "__main__":
    main()
