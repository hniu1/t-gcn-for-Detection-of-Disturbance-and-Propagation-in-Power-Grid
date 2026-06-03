"""
Training script for multi-feature T-GCN model.

Loads PMU data, creates datasets, trains the T-GCN model, and evaluates.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json

from data_processing import PMUDataProcessor
from dataset import create_datasets
from model import MultiFeatureTGCN, normalize_adjacency
from typing import Dict


TARGET_FEATURE_NAMES = ('freq_dev', 'volt_dev')


def should_log_progress(batch_idx: int, total_batches: int, percent_step: int) -> bool:
    """Return True when batch progress crosses the next logging threshold."""
    if total_batches <= 0:
        return False
    if batch_idx == 1 or batch_idx == total_batches:
        return True
    if percent_step <= 0:
        return False
    prev_pct = ((batch_idx - 1) * 100) // total_batches
    curr_pct = (batch_idx * 100) // total_batches
    return (curr_pct // percent_step) > (prev_pct // percent_step)


def masked_mse_loss(y_pred: torch.Tensor, y_target: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE so imputed targets do not dominate optimization."""
    sq_err = (y_pred - y_target) ** 2
    weighted = sq_err * y_mask
    denom = torch.clamp(y_mask.sum(), min=1.0)
    return weighted.sum() / denom


def train_epoch(model, train_loader, optimizer, device, A_norm, epoch_idx, total_epochs, progress_pct_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_batches = len(train_loader)
    epoch_start = time.time()
    
    for batch_idx, (x_dyn, x_grid, y_target, y_mask) in enumerate(train_loader, start=1):
        x_dyn = x_dyn.to(device)
        x_grid = x_grid.to(device)
        y_target = y_target.to(device)
        y_mask = y_mask.to(device)
        A_norm = A_norm.to(device)
        
        # Forward pass
        y_pred = model(x_dyn, x_grid, A_norm)  # (B, H, N, F_out)
        
        # Compute masked loss (ignore imputed target regions)
        loss = masked_mse_loss(y_pred, y_target, y_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1

        if should_log_progress(batch_idx, total_batches, progress_pct_step):
            elapsed = time.time() - epoch_start
            pct = (100.0 * batch_idx / total_batches) if total_batches > 0 else 100.0
            avg_loss_so_far = total_loss / batch_count
            batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0.0
            remaining_batches = total_batches - batch_idx
            eta_sec = (remaining_batches / batches_per_sec) if batches_per_sec > 0 else 0.0
            print(
                f"Epoch {epoch_idx}/{total_epochs} train {pct:5.1f}% "
                f"({batch_idx}/{total_batches} batches) | "
                f"Avg Loss: {avg_loss_so_far:.6f} | ETA: {eta_sec/60:.1f} min"
            )
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


def evaluate(model, data_loader, device, A_norm, phase_name=None, epoch_idx=None, total_epochs=None, progress_pct_step=25):
    """Evaluate on validation or test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_masks = []
    total_batches = len(data_loader)
    phase_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (x_dyn, x_grid, y_target, y_mask) in enumerate(data_loader, start=1):
            x_dyn = x_dyn.to(device)
            x_grid = x_grid.to(device)
            y_target = y_target.to(device)
            y_mask = y_mask.to(device)
            A_norm = A_norm.to(device)
            
            y_pred = model(x_dyn, x_grid, A_norm)
            loss = masked_mse_loss(y_pred, y_target, y_mask)
            
            total_loss += loss.item()
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
            all_masks.append(y_mask.cpu().numpy())

            if phase_name and should_log_progress(batch_idx, total_batches, progress_pct_step):
                elapsed = time.time() - phase_start
                pct = (100.0 * batch_idx / total_batches) if total_batches > 0 else 100.0
                batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0.0
                remaining_batches = total_batches - batch_idx
                eta_sec = (remaining_batches / batches_per_sec) if batches_per_sec > 0 else 0.0
                epoch_prefix = ""
                if epoch_idx is not None and total_epochs is not None:
                    epoch_prefix = f"Epoch {epoch_idx}/{total_epochs} "
                print(
                    f"{epoch_prefix}{phase_name} {pct:5.1f}% "
                    f"({batch_idx}/{total_batches} batches) | ETA: {eta_sec/60:.1f} min"
                )
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    # Concatenate predictions and targets
    preds = np.concatenate(all_preds, axis=0)  # (num_samples, H, N, F_out)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    return avg_loss, preds, targets, masks


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: (num_samples, H, N, F_out) predictions
        targets: (num_samples, H, N, F_out) ground truth
    
    Returns:
        Dict with keys: mse, mae, rmse, r2, mape (all per-element means)
    """
    # Flatten to observed-only vectors
    valid = masks.flatten() > 0.5
    pred_flat = predictions.flatten()[valid]
    target_flat = targets.flatten()[valid]
    if pred_flat.size == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan'), 'mape': float('nan')}
    
    # MSE
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    # MAE
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # R² (coefficient of determination)
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # MAPE (Mean Absolute Percentage Error)
    # Only compute where target != 0 to avoid division by zero
    nonzero_mask = np.abs(target_flat) > 1e-8
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((target_flat[nonzero_mask] - pred_flat[nonzero_mask]) / target_flat[nonzero_mask])) * 100
    else:
        mape = 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }


def robust_impute_feature_matrix(
    feature_matrix: np.ndarray,
    observed_mask: np.ndarray,
    train_end: int,
    short_gap_steps: int,
    medium_gap_steps: int,
) -> np.ndarray:
    """Impute missing values per sensor using interpolation, fill, then train median fallback."""
    out = feature_matrix.copy()
    T, N = out.shape
    for j in range(N):
        col = pd.Series(out[:, j], dtype=float)
        # Train-only fallback avoids target leakage.
        train_obs = observed_mask[:train_end, j] > 0.5
        train_vals = out[:train_end, j][train_obs]
        fallback = float(np.nanmedian(train_vals)) if train_vals.size > 0 else 0.0

        col = col.interpolate(method='linear', limit=short_gap_steps, limit_direction='both')
        col = col.ffill(limit=medium_gap_steps).bfill(limit=medium_gap_steps)
        col = col.fillna(fallback)
        out[:, j] = col.to_numpy(dtype=np.float32)
    return out


def compute_feature_scaler(
    feature_matrix: np.ndarray,
    observed_mask: np.ndarray,
    train_end: int,
    method: str,
) -> Dict[str, float]:
    """Compute train-only scaling stats from observed training values."""
    train_values = feature_matrix[:train_end][observed_mask[:train_end] > 0.5]
    if train_values.size == 0:
        return {'center': 0.0, 'scale': 1.0}

    if method == 'standard':
        center = float(np.mean(train_values))
        scale = float(np.std(train_values))
    elif method == 'robust':
        center = float(np.median(train_values))
        q25, q75 = np.percentile(train_values, [25, 75])
        scale = float(q75 - q25)
    else:
        center = 0.0
        scale = 1.0

    if not np.isfinite(scale) or scale < 1e-6:
        fallback_scale = float(np.std(train_values))
        scale = fallback_scale if np.isfinite(fallback_scale) and fallback_scale >= 1e-6 else 1.0

    return {'center': center, 'scale': scale}


def apply_feature_scaler(
    feature_matrix: np.ndarray,
    scaler: Dict[str, float],
    clip_value: float | None,
) -> np.ndarray:
    """Apply centering/scaling and optional clipping."""
    scaled = (feature_matrix - scaler['center']) / scaler['scale']
    if clip_value is not None and clip_value > 0:
        scaled = np.clip(scaled, -clip_value, clip_value)
    return scaled.astype(np.float32, copy=False)


def inverse_transform_outputs(
    output_array: np.ndarray,
    feature_scalers: Dict[str, Dict[str, float]],
    feature_names: tuple[str, ...] = TARGET_FEATURE_NAMES,
) -> np.ndarray:
    """Map scaled outputs back to original physical units."""
    restored = output_array.astype(np.float32, copy=True)
    for feature_idx, feature_name in enumerate(feature_names):
        scaler = feature_scalers.get(feature_name)
        if not scaler:
            continue
        restored[..., feature_idx] = restored[..., feature_idx] * scaler['scale'] + scaler['center']
    return restored


def main():
    parser = argparse.ArgumentParser(description='Train multi-feature T-GCN on PMU data')
    parser.add_argument('--data_dir', type=str, default='./data/2024-06-01', help='Path to parquet data')
    parser.add_argument('--metadata_file', type=str, default='./data/FDRLocation.xlsx', help='Metadata file')
    parser.add_argument('--adjacency_file', type=str, default='./results/A_geo.npy', help='Adjacency matrix file (A_geo.npy for 108 metadata sensors)')
    parser.add_argument('--inferred_locs_file', type=str, default='./results/inferred_locations.csv', help='Inferred locations file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    parser.add_argument('--Tin', type=int, default=100, help='Input history length (timesteps)')
    parser.add_argument('--H', type=int, default=10, help='Forecast horizon (steps)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--stride_train', type=int, default=1, help='Train window stride')
    parser.add_argument('--stride_val', type=int, default=1, help='Validation window stride')
    parser.add_argument('--stride_test', type=int, default=1, help='Test window stride')
    parser.add_argument('--stride_pred', type=int, default=1, help='Prediction holdout window stride')
    parser.add_argument('--train_frac', type=float, default=0.8, help='Fraction of timeline for training')
    parser.add_argument('--val_frac', type=float, default=0.05, help='Fraction of timeline for validation')
    parser.add_argument('--test_frac', type=float, default=0.05, help='Fraction of timeline for testing')
    parser.add_argument('--pred_frac', type=float, default=0.10, help='Fraction of timeline reserved for final prediction holdout')
    parser.add_argument('--min_step_coverage', type=float, default=0.8, help='Keep timestep only if observed node ratio >= this value')
    parser.add_argument('--short_gap_steps', type=int, default=10, help='Max gap for interpolation fill')
    parser.add_argument('--medium_gap_steps', type=int, default=300, help='Max gap for ffill/bfill')
    parser.add_argument('--sample_rate_hz', type=float, default=10.0, help='Sample rate used to align sensors by start-time offset')
    parser.add_argument('--feature_scaling', type=str, default='robust', choices=['none', 'standard', 'robust'], help='Train-only scaling applied to continuous features')
    parser.add_argument('--scaled_clip_value', type=float, default=10.0, help='Clip scaled continuous features to +/- this value; <=0 disables clipping')
    parser.add_argument('--progress_pct_step', type=int, default=5, help='Log training progress every N percent of an epoch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()

    if args.device.startswith('cuda'):
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for CUDA run (compatibility mode).")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Multi-Feature T-GCN Training")
    print("=" * 60)
    
    # Load adjacency matrix
    print("Loading adjacency matrix from:", args.adjacency_file)
    A_np = np.load(args.adjacency_file)
    A = torch.from_numpy(A_np).float()
    # Try to load saved sensor ordering that matches A_geo.npy
    sensor_order_path = Path(args.adjacency_file).parent / 'sensor_order.npy'
    sensor_order_list = None
    if sensor_order_path.exists():
        sensor_order = np.load(sensor_order_path)
        sensor_order_list = [int(x) for x in sensor_order]
        print(f"Loaded sensor order mapping ({len(sensor_order_list)} entries) from: {sensor_order_path}")

    A_norm = normalize_adjacency(A).to(args.device)
    N = A_np.shape[0]
    print(f"Adjacency matrix shape: {A_np.shape}")
    
    # Identify active sensors from parquet files (matching geo graph methodology)
    print("\nIdentifying active sensors from parquet files...")
    data_path = Path(args.data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))
    
    def extract_sensor_id(fname):
        """Extract sensor ID from parquet filename (e.g., '1047-...' -> 1047)"""
        stem = Path(fname).stem
        try:
            return int(stem.split('-')[0])
        except (ValueError, IndexError):
            return None
    
    active_ids = sorted(set(
        sid for sid in (extract_sensor_id(f.name) for f in parquet_files)
        if sid is not None
    ))
    print(f"Found {len(active_ids)} active sensors in data files")
    print(f"Example IDs: {active_ids[:5]}")
    
    # Load metadata and match with active sensors
    print("\nLoading metadata and matching with active sensors...")
    metadata = pd.read_excel(Path(args.metadata_file))
    print(f"Total sensors in metadata: {len(metadata)}")
    
    # Filter to sensors that have both data files AND metadata
    # If we have a saved sensor order mapping (from A_geo), use it to preserve adjacency ordering
    if sensor_order_list is not None:
        # Keep only sensors present in metadata and active_ids, in adjacency order
        metadata_sensor_ids = [int(sid) for sid in sensor_order_list if (sid in active_ids and sid in set(metadata['FDRID'].values))]
        print(f"Using adjacency sensor order; matched sensors: {len(metadata_sensor_ids)}")
    else:
        metadata_matched = metadata[metadata['FDRID'].isin(active_ids)].copy()
        metadata_matched = metadata_matched.reset_index(drop=True)
        metadata_sensor_ids = sorted(metadata_matched['FDRID'].unique())
    
    print(f"Sensors with both data files and metadata: {len(metadata_sensor_ids)}")
    
    # Verify we have the right number
    if len(metadata_sensor_ids) != N:
        print(f"ERROR: Expected {N} matched sensors (from adjacency), got {len(metadata_sensor_ids)}")
        print(f"Adjacency matrix may not match this data. Check A_geo.npy matches this date's data.")
        return
    
    # Load and align data on a full timeline with missingness masks.
    print("\nLoading PMU data and building full-timeline tensors...")
    proc = PMUDataProcessor(Path(args.data_dir), Path(args.metadata_file), Path(args.inferred_locs_file))
    proc.load_metadata()

    sensor_payload = {}
    sensor_start_time = {}
    for sid in metadata_sensor_ids:
        try:
            df = proc.load_sensor_data(int(sid))
            if df is None or len(df) == 0:
                continue
            feats = proc.compute_features(df)
            t_start = pd.to_datetime(df['ReceivedTime']).min()
            sensor_payload[int(sid)] = feats
            sensor_start_time[int(sid)] = t_start
        except Exception as exc:
            print(f"Skipping sensor {sid}: {exc}")

    sensor_ids = [int(sid) for sid in metadata_sensor_ids if int(sid) in sensor_payload]
    if len(sensor_ids) == 0:
        print("ERROR: No sensors could be loaded. Check data paths.")
        return

    global_start = min(sensor_start_time[sid] for sid in sensor_ids)
    offsets = {
        sid: int(round((sensor_start_time[sid] - global_start).total_seconds() * args.sample_rate_hz))
        for sid in sensor_ids
    }
    max_len = max(offsets[sid] + len(sensor_payload[sid]['freq_dev']) for sid in sensor_ids)

    N_loaded = len(sensor_ids)
    freq_dev = np.full((max_len, N_loaded), np.nan, dtype=np.float32)
    rocof = np.full((max_len, N_loaded), np.nan, dtype=np.float32)
    angle_delta = np.full((max_len, N_loaded), np.nan, dtype=np.float32)
    volt_dev = np.full((max_len, N_loaded), np.nan, dtype=np.float32)
    observed_mask = np.zeros((max_len, N_loaded), dtype=np.float32)

    for i, sid in enumerate(sensor_ids):
        payload = sensor_payload[sid]
        t0 = max(offsets[sid], 0)
        L = len(payload['freq_dev'])
        t1 = min(t0 + L, max_len)
        sl = slice(t0, t1)
        take = t1 - t0
        freq_dev[sl, i] = payload['freq_dev'][:take]
        rocof[sl, i] = payload['rocof'][:take]
        angle_delta[sl, i] = payload['angle_delta'][:take]
        volt_dev[sl, i] = payload['volt_dev'][:take]
        observed_mask[sl, i] = 1.0

    # Keep only timesteps with sufficient node coverage.
    step_coverage = observed_mask.mean(axis=1)
    keep_steps = step_coverage >= args.min_step_coverage
    if not np.any(keep_steps):
        print("ERROR: No timesteps satisfy min_step_coverage. Lower the threshold.")
        return

    kept_ratio = float(np.mean(keep_steps))
    print(f"Kept {int(keep_steps.sum())}/{len(keep_steps)} timesteps ({kept_ratio:.2%}) with coverage >= {args.min_step_coverage:.2f}")

    freq_dev = freq_dev[keep_steps]
    rocof = rocof[keep_steps]
    angle_delta = angle_delta[keep_steps]
    volt_dev = volt_dev[keep_steps]
    observed_mask = observed_mask[keep_steps]

    # Robust imputation uses train-only medians for final fallback.
    T_filtered = freq_dev.shape[0]
    train_end_tmp = int(T_filtered * args.train_frac)
    freq_dev = robust_impute_feature_matrix(freq_dev, observed_mask, train_end_tmp, args.short_gap_steps, args.medium_gap_steps)
    rocof = robust_impute_feature_matrix(rocof, observed_mask, train_end_tmp, args.short_gap_steps, args.medium_gap_steps)
    angle_delta = robust_impute_feature_matrix(angle_delta, observed_mask, train_end_tmp, args.short_gap_steps, args.medium_gap_steps)
    volt_dev = robust_impute_feature_matrix(volt_dev, observed_mask, train_end_tmp, args.short_gap_steps, args.medium_gap_steps)

    feature_scalers = {}
    clip_value = args.scaled_clip_value if args.scaled_clip_value > 0 else None
    if args.feature_scaling != 'none':
        for feature_name, feature_matrix in [
            ('freq_dev', freq_dev),
            ('rocof', rocof),
            ('angle_delta', angle_delta),
            ('volt_dev', volt_dev),
        ]:
            scaler = compute_feature_scaler(feature_matrix, observed_mask, train_end_tmp, args.feature_scaling)
            feature_scalers[feature_name] = scaler
            print(
                f"Scaling {feature_name}: method={args.feature_scaling} "
                f"center={scaler['center']:.6f} scale={scaler['scale']:.6f}"
            )

        freq_dev = apply_feature_scaler(freq_dev, feature_scalers['freq_dev'], clip_value)
        rocof = apply_feature_scaler(rocof, feature_scalers['rocof'], clip_value)
        angle_delta = apply_feature_scaler(angle_delta, feature_scalers['angle_delta'], clip_value)
        volt_dev = apply_feature_scaler(volt_dev, feature_scalers['volt_dev'], clip_value)
    else:
        print("Feature scaling disabled; using raw engineered feature values.")

    print(f"Feature arrays shape after filtering: {freq_dev.shape} (T={freq_dev.shape[0]}, N={N_loaded})")
    
    # Handle mismatch: properly index adjacency using sensor_order if available
    if N_loaded != N:
        if N_loaded < N:
            print(f"WARNING: Got {N_loaded} sensors but adjacency matrix is {N}x{N}")
            if sensor_order_list is not None:
                # Build index mapping from sensor ID -> adjacency index
                idx_map = {int(sid): idx for idx, sid in enumerate(sensor_order_list)}
                good_indices = [idx_map[sid] for sid in sensor_ids if sid in idx_map]
                if len(good_indices) != N_loaded:
                    print(f"WARNING: Number of good indices ({len(good_indices)}) != N_loaded ({N_loaded})")
                print(f"Using adjacency indices: keeping {len(good_indices)} sensors at indices: {good_indices[:10]}{'...' if len(good_indices)>10 else ''}")
                # Slice original numpy adjacency and re-normalize
                A_np = A_np[np.ix_(good_indices, good_indices)]
                A = torch.from_numpy(A_np).float()
                A_norm = normalize_adjacency(A).to(args.device)
                N = A_np.shape[0]
                print(f"Adjacency resized to: {A_np.shape}")
            else:
                print(f"Using top-left {N_loaded}x{N_loaded} submatrix of adjacency")
                A_norm = A_norm[:N_loaded, :N_loaded]
                N = N_loaded
        else:
            print(f"ERROR: Got {N_loaded} sensors but adjacency matrix is only {N}x{N}")
            return
    
    # Create grid embeddings: one embedding per unique GridName, mapped to sensors
    # grid_names is per loaded sensor in the same order as sensor_ids
    grid_names = [proc.get_grid_name(int(sid)) for sid in sensor_ids]
    unique_grids = sorted(set(grid_names))
    grid_idx_map = {g: i for i, g in enumerate(unique_grids)}
    np.random.seed(args.seed)
    grid_emb_by_grid = np.random.randn(len(unique_grids), 4).astype(np.float32)
    # Map per-sensor grid embedding
    grid_embeddings = np.zeros((N, 4), dtype=np.float32)
    for i, g in enumerate(grid_names[:N]):
        grid_embeddings[i] = grid_emb_by_grid[grid_idx_map[g]]
    
    # Create datasets
    print(f"\nCreating datasets (Tin={args.Tin}, H={args.H})...")
    train_ds, val_ds, test_ds, pred_ds = create_datasets(
        freq_dev, rocof, angle_delta, volt_dev, observed_mask, grid_embeddings,
        Tin=args.Tin,
        H=args.H,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        pred_frac=args.pred_frac,
        stride_train=args.stride_train,
        stride_val=args.stride_val,
        stride_test=args.stride_test,
        stride_pred=args.stride_pred,
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Prediction samples: {len(pred_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred_loader = DataLoader(pred_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\nCreating model (N={N}, hidden_dim={args.hidden_dim}, layers={args.num_layers})...")
    model = MultiFeatureTGCN(
        N=N,
        F_dyn=5,      # Δf, RoCoF, Δθ, ΔV, observed-mask
        F_static=4,   # Grid embedding
        F_out=2,      # Δf, ΔV (outputs)
        Tin=args.Tin,
        H=args.H,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(args.device)
    
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'pred_loss': [],
        'epochs': args.epochs
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs} started")
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            A_norm,
            epoch + 1,
            args.epochs,
            args.progress_pct_step,
        )
        
        # Validate
        val_loss, _, _, _ = evaluate(
            model,
            val_loader,
            args.device,
            A_norm,
            phase_name='val',
            epoch_idx=epoch + 1,
            total_epochs=args.epochs,
            progress_pct_step=max(args.progress_pct_step * 2, 10),
        )
        
        # Test (for monitoring)
        test_loss, _, _, _ = evaluate(
            model,
            test_loader,
            args.device,
            A_norm,
            phase_name='test',
            epoch_idx=epoch + 1,
            total_epochs=args.epochs,
            progress_pct_step=max(args.progress_pct_step * 2, 10),
        )

        pred_loss, _, _, _ = evaluate(
            model,
            pred_loader,
            args.device,
            A_norm,
            phase_name='pred',
            epoch_idx=epoch + 1,
            total_epochs=args.epochs,
            progress_pct_step=max(args.progress_pct_step * 2, 10),
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['pred_loss'].append(pred_loss)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} complete | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Test Loss: {test_loss:.6f} | Pred Loss: {pred_loss:.6f}"
        )
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {elapsed_time:.1f}s")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_train_loss, train_preds, train_targets, train_masks = evaluate(model, train_loader, args.device, A_norm, phase_name='final-train', progress_pct_step=25)
    final_val_loss, val_preds, val_targets, val_masks = evaluate(model, val_loader, args.device, A_norm, phase_name='final-val', progress_pct_step=25)
    final_test_loss, test_preds, test_targets, test_masks = evaluate(model, test_loader, args.device, A_norm, phase_name='final-test', progress_pct_step=25)
    final_pred_loss, pred_preds, pred_targets, pred_masks = evaluate(model, pred_loader, args.device, A_norm, phase_name='final-pred', progress_pct_step=25)
    
    # Compute metrics in both scaled space and original units.
    train_metrics_scaled = compute_metrics(train_preds, train_targets, train_masks)
    val_metrics_scaled = compute_metrics(val_preds, val_targets, val_masks)
    test_metrics_scaled = compute_metrics(test_preds, test_targets, test_masks)
    pred_metrics_scaled = compute_metrics(pred_preds, pred_targets, pred_masks)

    train_preds_original = inverse_transform_outputs(train_preds, feature_scalers)
    train_targets_original = inverse_transform_outputs(train_targets, feature_scalers)
    val_preds_original = inverse_transform_outputs(val_preds, feature_scalers)
    val_targets_original = inverse_transform_outputs(val_targets, feature_scalers)
    test_preds_original = inverse_transform_outputs(test_preds, feature_scalers)
    test_targets_original = inverse_transform_outputs(test_targets, feature_scalers)
    pred_preds_original = inverse_transform_outputs(pred_preds, feature_scalers)
    pred_targets_original = inverse_transform_outputs(pred_targets, feature_scalers)

    train_metrics_original = compute_metrics(train_preds_original, train_targets_original, train_masks)
    val_metrics_original = compute_metrics(val_preds_original, val_targets_original, val_masks)
    test_metrics_original = compute_metrics(test_preds_original, test_targets_original, test_masks)
    pred_metrics_original = compute_metrics(pred_preds_original, pred_targets_original, pred_masks)
    
    # Print losses
    print(f"\n--- Loss (MSE) ---")
    print(f"Train Loss: {final_train_loss:.6f}")
    print(f"Val Loss:   {final_val_loss:.6f}")
    print(f"Test Loss:  {final_test_loss:.6f}")
    print(f"Pred Loss:  {final_pred_loss:.6f}")
    
    # Print detailed metrics for each set
    print(f"\n--- Train Metrics (scaled space) ---")
    print(f"MSE:  {train_metrics_scaled['mse']:.6f}")
    print(f"MAE:  {train_metrics_scaled['mae']:.6f}")
    print(f"RMSE: {train_metrics_scaled['rmse']:.6f}")
    print(f"R²:   {train_metrics_scaled['r2']:.4f}")
    print(f"MAPE: {train_metrics_scaled['mape']:.2f}%")

    print(f"\n--- Val Metrics (scaled space) ---")
    print(f"MSE:  {val_metrics_scaled['mse']:.6f}")
    print(f"MAE:  {val_metrics_scaled['mae']:.6f}")
    print(f"RMSE: {val_metrics_scaled['rmse']:.6f}")
    print(f"R²:   {val_metrics_scaled['r2']:.4f}")
    print(f"MAPE: {val_metrics_scaled['mape']:.2f}%")

    print(f"\n--- Test Metrics (scaled space) ---")
    print(f"MSE:  {test_metrics_scaled['mse']:.6f}")
    print(f"MAE:  {test_metrics_scaled['mae']:.6f}")
    print(f"RMSE: {test_metrics_scaled['rmse']:.6f}")
    print(f"R²:   {test_metrics_scaled['r2']:.4f}")
    print(f"MAPE: {test_metrics_scaled['mape']:.2f}%")

    print(f"\n--- Prediction Metrics (scaled space) ---")
    print(f"MSE:  {pred_metrics_scaled['mse']:.6f}")
    print(f"MAE:  {pred_metrics_scaled['mae']:.6f}")
    print(f"RMSE: {pred_metrics_scaled['rmse']:.6f}")
    print(f"R²:   {pred_metrics_scaled['r2']:.4f}")
    print(f"MAPE: {pred_metrics_scaled['mape']:.2f}%")

    print(f"\n--- Train Metrics (original units) ---")
    print(f"MSE:  {train_metrics_original['mse']:.6f}")
    print(f"MAE:  {train_metrics_original['mae']:.6f}")
    print(f"RMSE: {train_metrics_original['rmse']:.6f}")
    print(f"R²:   {train_metrics_original['r2']:.4f}")
    print(f"MAPE: {train_metrics_original['mape']:.2f}%")

    print(f"\n--- Val Metrics (original units) ---")
    print(f"MSE:  {val_metrics_original['mse']:.6f}")
    print(f"MAE:  {val_metrics_original['mae']:.6f}")
    print(f"RMSE: {val_metrics_original['rmse']:.6f}")
    print(f"R²:   {val_metrics_original['r2']:.4f}")
    print(f"MAPE: {val_metrics_original['mape']:.2f}%")

    print(f"\n--- Test Metrics (original units) ---")
    print(f"MSE:  {test_metrics_original['mse']:.6f}")
    print(f"MAE:  {test_metrics_original['mae']:.6f}")
    print(f"RMSE: {test_metrics_original['rmse']:.6f}")
    print(f"R²:   {test_metrics_original['r2']:.4f}")
    print(f"MAPE: {test_metrics_original['mape']:.2f}%")

    print(f"\n--- Prediction Metrics (original units) ---")
    print(f"MSE:  {pred_metrics_original['mse']:.6f}")
    print(f"MAE:  {pred_metrics_original['mae']:.6f}")
    print(f"RMSE: {pred_metrics_original['rmse']:.6f}")
    print(f"R²:   {pred_metrics_original['r2']:.4f}")
    print(f"MAPE: {pred_metrics_original['mape']:.2f}%")
    
    # Save results
    np.save(output_dir / 'test_predictions.npy', test_preds)
    np.save(output_dir / 'test_targets.npy', test_targets)
    np.save(output_dir / 'test_predictions_original.npy', test_preds_original)
    np.save(output_dir / 'test_targets_original.npy', test_targets_original)
    np.save(output_dir / 'test_target_mask.npy', test_masks)
    np.save(output_dir / 'pred_predictions.npy', pred_preds)
    np.save(output_dir / 'pred_targets.npy', pred_targets)
    np.save(output_dir / 'pred_predictions_original.npy', pred_preds_original)
    np.save(output_dir / 'pred_targets_original.npy', pred_targets_original)
    np.save(output_dir / 'pred_target_mask.npy', pred_masks)
    np.save(output_dir / 'history.npy', history)
    
    # Save config with metrics
    config = vars(args)
    config['final_train_loss'] = float(final_train_loss)
    config['final_val_loss'] = float(final_val_loss)
    config['final_test_loss'] = float(final_test_loss)
    config['final_pred_loss'] = float(final_pred_loss)
    config['elapsed_time'] = elapsed_time
    config['num_params'] = int(sum(p.numel() for p in model.parameters()))
    config['kept_timestep_ratio'] = float(kept_ratio)
    config['kept_timesteps'] = int(keep_steps.sum())
    config['total_aligned_timesteps'] = int(len(keep_steps))
    config['feature_scalers'] = feature_scalers
    
    # Add detailed metrics
    config['train_metrics_scaled'] = train_metrics_scaled
    config['val_metrics_scaled'] = val_metrics_scaled
    config['test_metrics_scaled'] = test_metrics_scaled
    config['pred_metrics_scaled'] = pred_metrics_scaled
    config['train_metrics_original'] = train_metrics_original
    config['val_metrics_original'] = val_metrics_original
    config['test_metrics_original'] = test_metrics_original
    config['pred_metrics_original'] = pred_metrics_original
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - best_model.pt")
    print(f"  - test_predictions.npy")
    print(f"  - test_targets.npy")
    print(f"  - test_predictions_original.npy")
    print(f"  - test_targets_original.npy")
    print(f"  - test_target_mask.npy")
    print(f"  - pred_predictions.npy")
    print(f"  - pred_targets.npy")
    print(f"  - pred_predictions_original.npy")
    print(f"  - pred_targets_original.npy")
    print(f"  - pred_target_mask.npy")
    print(f"  - history.npy")
    print(f"  - config.json")


if __name__ == '__main__':
    main()
