# Downstream Task 1: Connected Sensor Clusters with Joint Freq/RoCoF Deviations

This task finds connected sensor clusters on the test split where both frequency-deviation error and RoCoF error are high at the same time.

## Method

1. Load test predictions and targets from a results directory.
2. Reduce forecast horizon to one value per sample (`mean`, `first`, or `last`).
3. Reduce to one horizon value per sample (`mean`, `first`, or `last`).
4. Compute sensor-level absolute deviation for `freq_dev`.
5. Estimate sensor-level RoCoF from temporal difference of `freq_dev` and compute absolute deviation.
6. Mark sensors active where both deviations exceed high quantile thresholds.
7. Build graph components among active sensors using strong adjacency edges.
8. Save same-time connected components as clusters and rank by size-weighted deviation magnitude.
9. Plot true, predicted, and deviation traces for each selected cluster.

## Default Inputs

- Results: `results_masked_cov80_full_scaled`
- Sensor order: `results/sensor_order.npy`
- Adjacency: `results/A_geo.npy`
- Split: `test`

## Run

```bash
python downstream_tasks/task1_regional_anomalies/task1_detect_regional_anomalies.py
```

Example with explicit cluster settings:

```bash
python downstream_tasks/task1_regional_anomalies/task1_detect_regional_anomalies.py \
	--freq_quantile 0.98 \
	--rocof_quantile 0.98 \
	--edge_quantile 0.85 \
	--min_cluster_size 3 \
	--max_clusters 30
```

## Outputs

All outputs are saved under:


Files:


---

## Alternative: Interactive Map-Style Hot Clusters

A complementary approach extracts clusters based on the interactive error map criterion: high frequency-deviation error (no RoCoF requirement), connected by geographic adjacency. This is useful for understanding where the model performs poorly on the most salient feature.

### Map-Hot Cluster Pipeline

**Step 1: Extract map-hot clusters**

```bash
python downstream_tasks/task1_regional_anomalies/generate_map_freq_hot_clusters.py \
	--results_dir results_masked_cov80_full_scaled \
	--split_name pred \
	--color_max_quantile 0.90 \
	--edge_threshold 0.0 \
	--min_cluster_size 2 \
	--max_rows 200
```

Outputs:
- `downstream_tasks/task1_regional_anomalies/outputs/map_freq_hot_clusters_all.csv` (all detected clusters)
- `downstream_tasks/task1_regional_anomalies/outputs/map_freq_hot_clusters_top.csv` (top 200 by cluster size + error)

**Step 2: Collapse and plot episodes**

```bash
python downstream_tasks/task1_regional_anomalies/collapse_map_clusters_and_plot.py \
	--split_name pred \
	--episode_gap 120 \
	--auto_increase_cutoff \
	--target_max_episodes 800 \
	--min_cluster_size 3 \
	--min_peak_mean_abs_err 0.02 \
	--max_plots 150 \
	--context_radius 25 \
	--peak_exclusion_window 180 \
	--window_overlap_mode sensor_overlap \
	--out_dir downstream_tasks/task1_regional_anomalies/outputs/map_cluster_episodes_current
```

Outputs:
- `downstream_tasks/task1_regional_anomalies/outputs/map_cluster_episodes_current/map_freq_hot_cluster_episodes_all.csv` (collapsed episodes)
- `downstream_tasks/task1_regional_anomalies/outputs/map_cluster_episodes_current/map_freq_hot_cluster_episodes_selected.csv` (top selected)
- `downstream_tasks/task1_regional_anomalies/outputs/map_cluster_episodes_current/episode_plots/*.png` (combined graph + time-series plots)
- `downstream_tasks/task1_regional_anomalies/outputs/map_cluster_episodes_current/summary.json`

### Key Differences from Task 1

| Aspect | Task 1 (Joint Freq/RoCoF) | Map-Hot Clusters |
|--------|---------------------------|------------------|
| Selection Criterion | High `freq_dev` AND high `RoCoF` error | High `freq_dev` error only |
| Split | `test` | `pred` (or `test`) |
| Cluster Count | ~30 (tight filter) | ~18k raw, ~300 episodes (loose filter) |
| Ranking | Size-weighted joint magnitude | Peak mean absolute error |
| De-duplication | None (unique timestamps) | Collapses same-sensor sets by time gap (default 120 steps) |
| Peak Suppression | None | Window-based (suppress nearby peaks with sensor overlap) |

The map-hot pipeline is particularly useful for:
- Identifying where frequency prediction is worst (most visible on interactive map).
- Finding recurring regional patterns (same sensor clusters appearing over time).
- Reducing 18k raw events to ~35 focused episodes for detailed visual inspection.

