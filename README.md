# RADAR: Rapid Automated Detection And Recognition of disturbances in power distribution networks

Multi-Feature T-GCN for PMU Spatiotemporal Forecasting

This directory contains a modular T-GCN model for multi-feature spatiotemporal forecasting on PMU data.

## Architecture

The model combines:
- **5 Dynamic Input Channels**: Δf (freq deviation), RoCoF, Δθ (angle dynamics), ΔV (voltage residual), observed-mask
- **Static Features**: GridName embeddings
- **Graph Convolution**: Applied across spatial network at each timestep
- **Temporal GRU**: Recurrent layer to capture inter-node temporal dynamics over Tin history
- **Prediction**: Multi-step forecast (H=10 steps) of 2 target features (Δf, ΔV)

## Components

### `data_processing.py`
Handles data loading and feature engineering:
- `PMUDataProcessor`: Main class for loading sensor data and computing features
- `prepare_data()`: Convenience function to load all sensors and return feature dict + grid map

**Key Features**:
- Loads PMU time-series from parquet files and matches to adjacency sensor ordering
- Computes 4 dynamic features with proper handling of angle unwrapping
- Loads metadata + city-inferred locations for GridName mapping
- Aligns sensors by start-time offset (sample-rate based)
- Supports coverage filtering with `min_step_coverage`
- Supports robust missing-value imputation and train-only feature scaling

### `dataset.py`
PyTorch Dataset implementation:
- `PMUForecastDataset`: Sliding window dataset with (Tin, H) window parameters
- `create_datasets()`: Time-based train/val/test/pred split

**Usage**:
```python
train_ds, val_ds, test_ds, pred_ds = create_datasets(
  freq_dev, rocof, angle_delta, volt_dev, observed_mask, grid_embeddings,
  Tin=100, H=10,
  train_frac=0.8, val_frac=0.05, test_frac=0.05, pred_frac=0.10,
  stride_train=10, stride_val=10, stride_test=10, stride_pred=10
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
```

### `model.py`
T-GCN model architecture:
- `GraphConvLayer`: Single graph convolution operation
- `TemporalGRULayer`: GRU-based temporal module capturing inter-node dynamics
- `MultiFeatureTGCN`: Full model combining graph convolutions + temporal GRU
- `normalize_adjacency()`: Normalization function for adjacency matrix

**Key Design Choices**:
- GRU for temporal modeling (matches original T-GCN architecture)
- Uses pre-computed geographic adjacency from `A_geo.npy`
- Static embeddings concatenated at each timestep
- GRU treats nodes as separate sequences to capture shared temporal patterns
- Output: (B, H, N, F_out) predictions via linear projection from GRU hidden state

### `train.py`
Main training script:
- Loads adjacency matrix and PMU data
- Creates train/val/test/pred datasets
- Trains model with early stopping
- Evaluates on all splits and saves scaled + original-unit arrays/metrics

## Cov80 Workflow (Current Main Run)

`cov80` means timesteps are retained only when at least 80% of sensors are observed:

- `min_step_coverage = 0.8`
- keep timestep if observed ratio >= 0.8

Current run snapshot (`results_masked_cov80_full_scaled/config.json`):

- Kept timesteps: 819,582 / 864,064 (94.85%)
- Sensors: 107
- Tin/H: 100/10
- Strides: train/val/test/pred = 10/10/10/10
- Split fractions: 0.8/0.05/0.05/0.10

Pair counts for this run:

- train: 65,556
- val: 4,087
- test: 4,087
- pred: 8,185
- total across splits: 81,915

### Missing-Signal Synthetic Value Policy (Deterministic Imputation)

After cov80 filtering, missing entries inside retained timesteps are filled per sensor and per feature channel using:

1. Linear interpolation for short gaps (`short_gap_steps = 10`)
2. Forward/backward fill for medium gaps (`medium_gap_steps = 300`)
3. Final fallback to train-only median for that sensor-feature column (or 0.0 if unavailable)

Notes:

- Fallback uses training timeline statistics only (prevents leakage from val/test/pred).
- The observed-mask channel is kept in model input, so the network can distinguish originally observed vs imputed entries.

## Quick Start

### 1. Prepare Data
Ensure you have:
- `../data/FDRLocation.xlsx` - Sensor metadata
- `../data/2024-06-01/*.parquet` - 126 sensor parquet files
- `../results/A_geo.npy` - Pre-computed geographic adjacency
- `../results/inferred_locations.csv` - Inferred locations for missing sensors

### 2. Run Training
```bash
python train.py \
  --data_dir /lustre/orion/proj-shared/cli138/7hn/FNET/data/2024-06-01 \
  --metadata_file /lustre/orion/proj-shared/cli138/7hn/FNET/data/FDRLocation.xlsx \
  --adjacency_file /lustre/orion/proj-shared/cli138/7hn/FNET/t-gcn-for-Detection-of-Disturbance-and-Propagation-in-Power-Grid/results/A_geo.npy \
  --inferred_locs_file /lustre/orion/proj-shared/cli138/7hn/FNET/t-gcn-for-Detection-of-Disturbance-and-Propagation-in-Power-Grid/results/inferred_locations.csv \
  --output_dir /lustre/orion/proj-shared/cli138/7hn/FNET/t-gcn-for-Detection-of-Disturbance-and-Propagation-in-Power-Grid/results_masked_cov80_full_scaled \
  --Tin 100 \
  --H 10 \
  --hidden_dim 64 \
  --num_layers 2 \
  --epochs 50 \
  --batch_size 128 \
  --stride_train 10 \
  --stride_val 10 \
  --stride_test 10 \
  --stride_pred 10 \
  --train_frac 0.8 \
  --val_frac 0.05 \
  --test_frac 0.05 \
  --pred_frac 0.10 \
  --min_step_coverage 0.8 \
  --short_gap_steps 10 \
  --medium_gap_steps 300 \
  --feature_scaling robust \
  --scaled_clip_value 10.0 \
  --lr 1e-3
```

### 3. Output Files
Training produces in the configured output directory (for cov80 run: `results_masked_cov80_full_scaled/`):
- `best_model.pt` - Best model weights
- `test_predictions.npy`, `test_targets.npy` - Scaled-domain test arrays
- `test_predictions_original.npy`, `test_targets_original.npy` - Original-unit test arrays
- `pred_predictions.npy`, `pred_targets.npy` - Scaled-domain pred arrays
- `pred_predictions_original.npy`, `pred_targets_original.npy` - Original-unit pred arrays
- `test_target_mask.npy`, `pred_target_mask.npy` - Target masks
- `history.npy` - Training history (losses)
- `config.json` - Full configuration + split metrics in scaled and original units

## Data Shapes

| Variable | Shape | Description |
|----------|-------|-------------|
| `x_dyn` | (B, Tin, N, 5) | Input: Δf, RoCoF, Δθ, ΔV, observed-mask |
| `x_grid` | (B, N, 4) | Static: GridName embeddings |
| `A` | (N, N) | Adjacency matrix (geographic k-NN) |
| `y_pred` | (B, H, N, 2) | Output: Δf, ΔV predictions |

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `Tin` | 100 | Input history (10 sec at 10 Hz) |
| `H` | 10 | Forecast horizon (1 sec) |
| `hidden_dim` | 64 | GCN/temporal conv hidden dimension |
| `num_layers` | 2 | Number of GCN layers |
| `batch_size` | 32 | Parser default; cov80 run used 128 |
| `lr` | 1e-3 | Learning rate (Adam) |
| `epochs` | 50 | Max training epochs |
| `min_step_coverage` | 0.8 | Keep timestep only if observed ratio >= threshold |
| `short_gap_steps` | 10 | Interpolation gap limit |
| `medium_gap_steps` | 300 | ffill/bfill gap limit |
| `feature_scaling` | robust | Train-only scaling mode |
| `scaled_clip_value` | 10.0 | Clip scaled features to +/- value |

## Extensions

Future improvements:
- [ ] Add attention mechanism to learn dynamic edges
- [ ] Support graph learning (learn adjacency from data)
- [ ] Add uncertainty quantification (prediction intervals)
- [ ] Implement rolling evaluation (sliding window test)
- [ ] Multi-horizon training with horizon-specific losses
- [ ] Incorporate external features (weather, load, etc.)

