# RADAR: Rapid Automated Detection And Recognition of disturbances in power distribution networks

Multi-Feature T-GCN for PMU Spatiotemporal Forecasting

This directory contains a modular T-GCN model for multi-feature spatiotemporal forecasting on PMU data.

## Architecture

The model combines:
- **4 Dynamic Features**: Δf (freq deviation), RoCoF (rate of change), Δθ (angle dynamics), ΔV (voltage residual)
- **Static Features**: GridName embeddings
- **Graph Convolution**: Applied across spatial network at each timestep
- **Temporal GRU**: Recurrent layer to capture inter-node temporal dynamics over Tin history
- **Prediction**: Multi-step forecast (H=10 steps) of 3 target features (Δf, Δθ, ΔV)

## Components

### `data_processing.py`
Handles data loading and feature engineering:
- `PMUDataProcessor`: Main class for loading sensor data and computing features
- `prepare_data()`: Convenience function to load all sensors and return feature dict + grid map

**Key Features**:
- Loads PMU time-series from parquet files (126 sensors)
- Computes 4 dynamic features with proper handling of angle unwrapping
- Loads metadata + city-inferred locations for GridName mapping
- Aligns sensors to common time range

### `dataset.py`
PyTorch Dataset implementation:
- `PMUForecastDataset`: Sliding window dataset with (Tin, H) window parameters
- `create_datasets()`: Time-based train/val/test split (80/10/10)

**Usage**:
```python
train_ds, val_ds, test_ds = create_datasets(
    freq_dev, rocof, angle_delta, volt_dev, grid_embeddings,
    Tin=100, H=10, stride_train=1, stride_val=5, stride_test=5
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
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
- Creates train/val/test datasets
- Trains model with early stopping
- Evaluates on test set and saves results

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
  --data_dir ../data/2024-06-01 \
  --metadata_file ../data/FDRLocation.xlsx \
  --adjacency_file ../results/A_geo.npy \
  --inferred_locs_file ../results/inferred_locations.csv \
  --Tin 100 \
  --H 10 \
  --hidden_dim 64 \
  --num_layers 2 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3
```

### 3. Output Files
Training produces in `./results/`:
- `best_model.pt` - Best model weights
- `test_predictions.npy` - Predictions on test set (B, H, N, F_out)
- `test_targets.npy` - Ground truth targets
- `history.npy` - Training history (losses)
- `config.json` - Training configuration + final metrics

## Data Shapes

| Variable | Shape | Description |
|----------|-------|-------------|
| `x_dyn` | (B, Tin, N, 4) | Input: Δf, RoCoF, Δθ, ΔV |
| `x_grid` | (B, N, 4) | Static: GridName embeddings |
| `A` | (N, N) | Adjacency matrix (geographic k-NN) |
| `y_pred` | (B, H, N, 3) | Output: Δf, Δθ, ΔV predictions |

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `Tin` | 100 | Input history (10 sec at 10 Hz) |
| `H` | 10 | Forecast horizon (1 sec) |
| `hidden_dim` | 64 | GCN/temporal conv hidden dimension |
| `num_layers` | 2 | Number of GCN layers |
| `batch_size` | 32 | Batch size |
| `lr` | 1e-3 | Learning rate (Adam) |
| `epochs` | 50 | Max training epochs |

## Extensions

Future improvements:
- [ ] Add attention mechanism to learn dynamic edges
- [ ] Support graph learning (learn adjacency from data)
- [ ] Add uncertainty quantification (prediction intervals)
- [ ] Implement rolling evaluation (sliding window test)
- [ ] Multi-horizon training with horizon-specific losses
- [ ] Incorporate external features (weather, load, etc.)

