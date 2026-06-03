#!/usr/bin/env python3
"""Build interactive geographic error maps for the current cov80 run."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from branca.colormap import linear
from folium.plugins import TimestampedGeoJson


DEFAULT_RESULTS_DIR = "results_masked_cov80_full_scaled"
DEFAULT_FEATURE_NAMES = ["freq_dev", "volt_dev"]


def infer_feature_names(num_features: int) -> list[str]:
    return DEFAULT_FEATURE_NAMES[:num_features]


def load_split_arrays(results_dir: Path, split_name: str, use_original: bool) -> tuple[np.ndarray, np.ndarray, str]:
    suffix = "_original" if use_original else ""
    pred_path = results_dir / f"{split_name}_predictions{suffix}.npy"
    true_path = results_dir / f"{split_name}_targets{suffix}.npy"
    if pred_path.exists() and true_path.exists():
        return np.load(pred_path), np.load(true_path), "original" if use_original else "scaled"
    return np.load(results_dir / f"{split_name}_predictions.npy"), np.load(results_dir / f"{split_name}_targets.npy"), "scaled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive error map from current cov80 outputs")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split_name", type=str, default="pred", choices=["test", "pred"])
    parser.add_argument("--use_original", action="store_true", default=True)
    parser.add_argument("--graphs_dir", type=str, default="../results/graphs")
    parser.add_argument("--sensor_order_file", type=str, default="results/sensor_order.npy")
    parser.add_argument("--reordered_csv", type=str, default="test_analysis_cov80/full_window_all_sensors_pred/sensor_order_graph_spectral.csv")
    parser.add_argument("--out_html", type=str, default="results_masked_cov80_full_scaled/graphs/map_error_freq_dev_interactive.html")
    parser.add_argument("--out_html_animated", type=str, default="results_masked_cov80_full_scaled/graphs/map_error_freq_dev_timeplay_interactive.html")
    parser.add_argument("--feature_idx", type=int, default=0)
    parser.add_argument("--horizon_reduce", type=str, default="mean", choices=["mean", "first", "last"])
    parser.add_argument("--highlight_tail_frac", type=float, default=0.2)
    parser.add_argument("--color_max_quantile", type=float, default=0.90)
    parser.add_argument("--color_max_value", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="both", choices=["static", "animated", "both"])
    parser.add_argument("--time_stride", type=int, default=10)
    parser.add_argument("--period_seconds", type=int, default=1)
    parser.add_argument("--frame_duration_ms", type=int, default=350)
    parser.add_argument("--max_speed", type=int, default=240)
    parser.add_argument("--adjacency_file", type=str, default="results/A_geo.npy")
    parser.add_argument("--edge_threshold", type=float, default=0.0)
    parser.add_argument("--no_edges", action="store_true")
    return parser.parse_args()


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
        raise ValueError(f"Timestamp reconstruction mismatch for {split_name}: expected {num_samples} samples, got {len(valid_starts)}")

    if horizon_reduce == "last":
        horizon_offset = tin + horizon - 1
    else:
        horizon_offset = tin

    timestamps: list[datetime] = []
    for start_idx in valid_starts:
        filtered_idx = filtered_indices[split_start + start_idx + horizon_offset]
        timestamps.append(global_start.to_pydatetime() + timedelta(seconds=filtered_idx / sample_rate_hz))
    return timestamps


def draw_graph_edges(m: folium.Map, df: pd.DataFrame, adjacency: np.ndarray, threshold: float) -> int:
    idx_to_latlon = {int(row.original_index): (float(row.lat), float(row.lon)) for row in df.itertuples(index=False)}
    n_edges = 0
    max_weight = float(np.max(adjacency)) if np.size(adjacency) else 0.0
    for i in range(adjacency.shape[0]):
        if i not in idx_to_latlon:
            continue
        lat1, lon1 = idx_to_latlon[i]
        for j in range(i + 1, adjacency.shape[1]):
            if j not in idx_to_latlon:
                continue
            weight = float(adjacency[i, j])
            if weight <= threshold:
                continue
            lat2, lon2 = idx_to_latlon[j]
            norm = weight / max_weight if max_weight > 0 else 0.0
            color = f"rgb({int(255 * norm)},0,{int(255 * (1.0 - norm))})"
            folium.PolyLine([[lat1, lon1], [lat2, lon2]], color=color, weight=max(0.4, 2.0 * norm), opacity=0.28).add_to(m)
            n_edges += 1
    return n_edges


def build_base_dataframe(sensor_ids: np.ndarray, metadata: pd.DataFrame, reordered_df: pd.DataFrame | None, mae_by_sensor: np.ndarray, p95_by_sensor: np.ndarray) -> pd.DataFrame:
    meta_index = metadata.set_index("SensorID")
    rows = []
    for original_index, sensor_id in enumerate(sensor_ids.tolist()):
        if sensor_id not in meta_index.index:
            continue
        meta_row = meta_index.loc[sensor_id]
        reordered_index = None
        if reordered_df is not None and sensor_id in reordered_df.index:
            reordered_index = int(reordered_df.loc[sensor_id, "reordered_index"])
        rows.append({
            "sensor_id": int(sensor_id),
            "original_index": int(original_index),
            "reordered_index": reordered_index,
            "lat": float(meta_row["Latitude"]),
            "lon": float(meta_row["Longitude"]),
            "mae": float(mae_by_sensor[original_index]),
            "p95": float(p95_by_sensor[original_index]),
        })
    return pd.DataFrame(rows)


def compute_highlight_ids(df: pd.DataFrame, tail_frac: float) -> set[int]:
    if tail_frac <= 0 or "reordered_index" not in df or not df["reordered_index"].notna().any():
        return set()
    max_rank = int(df["reordered_index"].max())
    start_rank = int((1.0 - tail_frac) * (max_rank + 1))
    return set(df.loc[df["reordered_index"] >= start_rank, "sensor_id"].astype(int).tolist())


def build_capped_colormap(values: np.ndarray, color_max_quantile: float, color_max_value: float) -> tuple[any, float, float]:
    min_value = float(np.min(values))
    if color_max_value > 0:
        capped_max = float(color_max_value)
    else:
        capped_max = float(np.quantile(values, min(max(color_max_quantile, 0.01), 1.0)))
    max_value = max(capped_max, min_value + 1e-9)
    return linear.YlOrRd_09.scale(min_value, max_value), min_value, max_value


def render_static_map(df: pd.DataFrame, out_html: Path, feature_name: str, horizon_reduce: str, highlight_ids: set[int], highlight_tail_frac: float, adjacency: np.ndarray | None, edge_threshold: float, color_max_quantile: float, color_max_value: float) -> None:
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    m = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap")
    n_edges = draw_graph_edges(m, df, adjacency, edge_threshold) if adjacency is not None else 0
    colormap, _, color_max = build_capped_colormap(df["mae"].to_numpy(), color_max_quantile, color_max_value)
    colormap.caption = f"MAE per sensor ({feature_name}, horizon={horizon_reduce})"
    colormap.add_to(m)
    for _, row in df.iterrows():
        is_highlight = int(row["sensor_id"]) in highlight_ids
        reorder_text = "N/A" if pd.isna(row["reordered_index"]) else str(int(row["reordered_index"]))
        display_mae = min(float(row["mae"]), color_max)
        popup = (
            f"<b>Sensor ID:</b> {int(row['sensor_id'])}<br>"
            f"<b>Original index:</b> {int(row['original_index'])}<br>"
            f"<b>Graph spectral reordered index:</b> {reorder_text}<br>"
            f"<b>{feature_name} MAE:</b> {float(row['mae']):.6f}<br>"
            f"<b>{feature_name} P95 abs error:</b> {float(row['p95']):.6f}"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=7 if is_highlight else 5,
            popup=folium.Popup(popup, max_width=320),
            color="black" if is_highlight else colormap(display_mae),
            fill=True,
            fill_color=colormap(display_mae),
            fill_opacity=0.85,
            weight=1.2 if is_highlight else 0.5,
        ).add_to(m)
    note = (
        f"<div style=\"position: fixed; top: 10px; left: 50px; width: 500px; background-color: white; border:2px solid #666; z-index:9999; font-size:14px; padding:10px;\">"
        f"<b>Static Error Map: {feature_name}</b><br>"
        f"Color = per-sensor MAE. Highlighted markers = rightmost {int(highlight_tail_frac * 100)}% in reordered heatmap axis.<br>"
        f"Edges drawn: {n_edges} (threshold &gt; {edge_threshold})"
        f"</div>"
    )
    m.get_root().html.add_child(folium.Element(note))
    m.save(str(out_html))


def render_animated_map(df: pd.DataFrame, abs_err: np.ndarray, sample_times: list[datetime], out_html: Path, feature_name: str, horizon_reduce: str, highlight_ids: set[int], highlight_tail_frac: float, adjacency: np.ndarray | None, edge_threshold: float, time_stride: int, period_seconds: int, frame_duration_ms: int, max_speed: int, color_max_quantile: float, color_max_value: float) -> int:
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    m = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap")
    n_edges = draw_graph_edges(m, df, adjacency, edge_threshold) if adjacency is not None else 0
    colormap, _, color_max = build_capped_colormap(abs_err, color_max_quantile, color_max_value)
    colormap.caption = f"Instant |pred-actual| ({feature_name}, horizon={horizon_reduce})"
    colormap.add_to(m)
    sensor_rows = list(df.itertuples(index=False))
    features = []
    frame_count = 0
    for t in range(0, abs_err.shape[0], max(1, time_stride)):
        frame_time_dt = sample_times[t]
        frame_time = frame_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
        for row in sensor_rows:
            error_value = float(abs_err[t, int(row.original_index)])
            display_error = min(error_value, color_max)
            is_highlight = int(row.sensor_id) in highlight_ids
            reorder_text = "N/A" if pd.isna(row.reordered_index) else str(int(row.reordered_index))
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row.lon), float(row.lat)]},
                "properties": {
                    "time": frame_time,
                    "popup": f"Sensor {int(row.sensor_id)} | time={frame_time_dt.strftime('%Y-%m-%d %H:%M:%S')}<br>sample={t}<br>abs_error={error_value:.6f}<br>orig_idx={int(row.original_index)} | reorder_idx={reorder_text}",
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": colormap(display_error),
                        "fillOpacity": 0.85,
                        "stroke": True,
                        "weight": 1.0 if is_highlight else 0.4,
                        "color": "black" if is_highlight else colormap(display_error),
                        "radius": 8 if is_highlight else 6,
                    },
                },
            })
        frame_count += 1
    TimestampedGeoJson({"type": "FeatureCollection", "features": features}, period=f"PT{max(1, period_seconds)}S", transition_time=max(50, frame_duration_ms), auto_play=True, loop=True, max_speed=max(1, max_speed), loop_button=True, date_options="YYYY-MM-DD HH:mm:ss", time_slider_drag_update=True).add_to(m)
    note = (
        f"<div style=\"position: fixed; top: 10px; left: 50px; width: 560px; background-color: white; border:2px solid #666; z-index:9999; font-size:14px; padding:10px;\">"
        f"<b>Animated Error Map: {feature_name}</b><br>Points animate by reconstructed timestamps (stride={time_stride}). Autoplay enabled.<br>"
        f"Highlighted markers = rightmost {int(highlight_tail_frac * 100)}% in reordered heatmap axis.<br>Edges drawn: {n_edges} (threshold &gt; {edge_threshold})"
        f"</div>"
    )
    m.get_root().html.add_child(folium.Element(note))
    m.save(str(out_html))
    return frame_count


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    graphs_dir = Path(args.graphs_dir)
    out_html = Path(args.out_html)
    out_html_animated = Path(args.out_html_animated)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html_animated.parent.mkdir(parents=True, exist_ok=True)
    pred, true, _ = load_split_arrays(results_dir, args.split_name, args.use_original)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (samples, horizon, sensors, features), got {pred.ndim}D")
    feature_names = infer_feature_names(pred.shape[-1])
    feature_idx = int(np.clip(args.feature_idx, 0, pred.shape[-1] - 1))
    feature_name = feature_names[feature_idx]
    pred_reduced = reduce_horizon(pred, args.horizon_reduce)
    true_reduced = reduce_horizon(true, args.horizon_reduce)
    abs_err = np.abs(pred_reduced[:, :, feature_idx] - true_reduced[:, :, feature_idx])
    sample_times = reconstruct_split_timestamps(results_dir, args.split_name, abs_err.shape[0], args.horizon_reduce)
    mae_by_sensor = np.mean(abs_err, axis=0)
    p95_by_sensor = np.quantile(abs_err, 0.95, axis=0)
    sensor_ids = np.load(Path(args.sensor_order_file)).astype(int)
    metadata = pd.read_csv(graphs_dir / "sensor_metadata.csv")
    metadata["SensorID"] = metadata["SensorID"].astype(int)
    reordered_df = None
    reordered_path = Path(args.reordered_csv)
    if reordered_path.exists():
        reordered_df = pd.read_csv(reordered_path)
        reordered_df["sensor_id"] = reordered_df["sensor_id"].astype(int)
        reordered_df["reordered_index"] = reordered_df["reordered_index"].astype(int)
        reordered_df = reordered_df.set_index("sensor_id")
    df = build_base_dataframe(sensor_ids, metadata, reordered_df, mae_by_sensor, p95_by_sensor)
    highlight_ids = compute_highlight_ids(df, args.highlight_tail_frac)
    adjacency = None if args.no_edges else np.load(Path(args.adjacency_file))
    if args.mode in ("static", "both"):
        render_static_map(df, out_html, feature_name, args.horizon_reduce, highlight_ids, args.highlight_tail_frac, adjacency, args.edge_threshold, args.color_max_quantile, args.color_max_value)
        print(f"Saved static map: {out_html}")
    if args.mode in ("animated", "both"):
        frames = render_animated_map(df, abs_err, sample_times, out_html_animated, feature_name, args.horizon_reduce, highlight_ids, args.highlight_tail_frac, adjacency, args.edge_threshold, args.time_stride, args.period_seconds, args.frame_duration_ms, args.max_speed, args.color_max_quantile, args.color_max_value)
        print(f"Saved animated map: {out_html_animated}")
        print(f"Animation frames: {frames}")
    print(f"Mapped sensors: {len(df)}")


if __name__ == "__main__":
    main()