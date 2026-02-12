"""
Training script for multi-feature T-GCN model.

Loads PMU data, creates datasets, trains the T-GCN model, and evaluates.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json

from data_processing import PMUDataProcessor, prepare_data
from dataset import PMUForecastDataset, create_datasets
from model import MultiFeatureTGCN, normalize_adjacency
from typing import Dict


def train_epoch(model, train_loader, optimizer, criterion, device, A_norm):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for x_dyn, x_grid, y_target in train_loader:
        x_dyn = x_dyn.to(device)
        x_grid = x_grid.to(device)
        y_target = y_target.to(device)
        A_norm = A_norm.to(device)
        
        # Forward pass
        y_pred = model(x_dyn, x_grid, A_norm)  # (B, H, N, F_out)
        
        # Compute loss (average over batch, horizon, nodes, features)
        loss = criterion(y_pred, y_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


def evaluate(model, data_loader, criterion, device, A_norm):
    """Evaluate on validation or test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_dyn, x_grid, y_target in data_loader:
            x_dyn = x_dyn.to(device)
            x_grid = x_grid.to(device)
            y_target = y_target.to(device)
            A_norm = A_norm.to(device)
            
            y_pred = model(x_dyn, x_grid, A_norm)
            loss = criterion(y_pred, y_target)
            
            total_loss += loss.item()
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    # Concatenate predictions and targets
    preds = np.concatenate(all_preds, axis=0)  # (num_samples, H, N, F_out)
    targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, preds, targets


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: (num_samples, H, N, F_out) predictions
        targets: (num_samples, H, N, F_out) ground truth
    
    Returns:
        Dict with keys: mse, mae, rmse, r2, mape (all per-element means)
    """
    # Flatten to (num_elements,)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
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
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
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
    
    # Load and prepare data
    print("\nLoading and preparing PMU data...")
    features_dict, sensor_ids, grid_map, grid_names = prepare_data(
        Path(args.data_dir),
        Path(args.metadata_file),
        Path(args.inferred_locs_file),
        sensor_ids=metadata_sensor_ids  # Use matched metadata sensors
    )
    
    print(f"Loaded {len(features_dict)} sensors")
    print(f"Grid map has {len(grid_map)} unique grids")
    
    if len(features_dict) == 0:
        print("ERROR: No sensors could be loaded. Check data paths.")
        return
    
    # Extract features
    freq_dev = []
    rocof = []
    angle_delta = []
    volt_dev = []
    
    for sid in sensor_ids:
        if sid in features_dict:
            feats = features_dict[sid]
            freq_dev.append(feats['freq_dev'])
            rocof.append(feats['rocof'])
            angle_delta.append(feats['angle_delta'])
            volt_dev.append(feats['volt_dev'])
    
    if len(freq_dev) == 0:
        print("ERROR: No valid features extracted from sensors.")
        return
    
    # Align to common length
    min_len = min(len(f) for f in freq_dev)
    freq_dev = np.array([f[:min_len] for f in freq_dev]).T  # (T, N_loaded)
    rocof = np.array([r[:min_len] for r in rocof]).T
    angle_delta = np.array([a[:min_len] for a in angle_delta]).T
    volt_dev = np.array([v[:min_len] for v in volt_dev]).T
    
    N_loaded = freq_dev.shape[1]
    print(f"Feature arrays shape: {freq_dev.shape} (T={freq_dev.shape[0]}, N={N_loaded})")
    
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
    # grid_names is a list of grid name per sensor in the same order as sensor_ids
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
    train_ds, val_ds, test_ds = create_datasets(
        freq_dev, rocof, angle_delta, volt_dev, grid_embeddings,
        Tin=args.Tin,
        H=args.H,
        train_frac=0.8,
        val_frac=0.1,
        stride_train=1,
        stride_val=5,  # Subsample validation
        stride_test=5   # Subsample test
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\nCreating model (N={N}, hidden_dim={args.hidden_dim}, layers={args.num_layers})...")
    model = MultiFeatureTGCN(
        N=N,
        F_dyn=4,      # Δf, RoCoF, Δθ, ΔV
        F_static=4,   # Grid embedding
        F_out=3,      # Δf, Δθ, ΔV (outputs)
        Tin=args.Tin,
        H=args.H,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(args.device)
    
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'epochs': args.epochs
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, A_norm)
        
        # Validate
        val_loss, _, _ = evaluate(model, val_loader, criterion, args.device, A_norm)
        
        # Test (for monitoring)
        test_loss, _, _ = evaluate(model, test_loader, criterion, args.device, A_norm)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {elapsed_time:.1f}s")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_train_loss, train_preds, train_targets = evaluate(model, train_loader, criterion, args.device, A_norm)
    final_val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, args.device, A_norm)
    final_test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, args.device, A_norm)
    
    # Compute detailed metrics
    train_metrics = compute_metrics(train_preds, train_targets)
    val_metrics = compute_metrics(val_preds, val_targets)
    test_metrics = compute_metrics(test_preds, test_targets)
    
    # Print losses
    print(f"\n--- Loss (MSE) ---")
    print(f"Train Loss: {final_train_loss:.6f}")
    print(f"Val Loss:   {final_val_loss:.6f}")
    print(f"Test Loss:  {final_test_loss:.6f}")
    
    # Print detailed metrics for each set
    print(f"\n--- Train Metrics ---")
    print(f"MSE:  {train_metrics['mse']:.6f}")
    print(f"MAE:  {train_metrics['mae']:.6f}")
    print(f"RMSE: {train_metrics['rmse']:.6f}")
    print(f"R²:   {train_metrics['r2']:.4f}")
    print(f"MAPE: {train_metrics['mape']:.2f}%")
    
    print(f"\n--- Val Metrics ---")
    print(f"MSE:  {val_metrics['mse']:.6f}")
    print(f"MAE:  {val_metrics['mae']:.6f}")
    print(f"RMSE: {val_metrics['rmse']:.6f}")
    print(f"R²:   {val_metrics['r2']:.4f}")
    print(f"MAPE: {val_metrics['mape']:.2f}%")
    
    print(f"\n--- Test Metrics ---")
    print(f"MSE:  {test_metrics['mse']:.6f}")
    print(f"MAE:  {test_metrics['mae']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"R²:   {test_metrics['r2']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.2f}%")
    
    # Save results
    np.save(output_dir / 'test_predictions.npy', test_preds)
    np.save(output_dir / 'test_targets.npy', test_targets)
    np.save(output_dir / 'history.npy', history)
    
    # Save config with metrics
    config = vars(args)
    config['final_train_loss'] = float(final_train_loss)
    config['final_val_loss'] = float(final_val_loss)
    config['final_test_loss'] = float(final_test_loss)
    config['elapsed_time'] = elapsed_time
    config['num_params'] = int(sum(p.numel() for p in model.parameters()))
    
    # Add detailed metrics
    config['train_metrics'] = train_metrics
    config['val_metrics'] = val_metrics
    config['test_metrics'] = test_metrics
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - best_model.pt")
    print(f"  - test_predictions.npy")
    print(f"  - test_targets.npy")
    print(f"  - history.npy")
    print(f"  - config.json")


if __name__ == '__main__':
    main()
