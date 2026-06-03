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

- `downstream_tasks/task1_regional_anomalies/outputs`

Files:

- `clusters_all.csv`
- `clusters_top.csv`
- `sensor_participation.csv`
- `thresholds.json`
- `summary.json`
- `cluster_plots/*.png` (pred/true/deviation traces for each selected cluster)
