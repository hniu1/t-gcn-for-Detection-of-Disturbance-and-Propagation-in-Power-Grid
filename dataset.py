"""
PyTorch Dataset for multi-feature T-GCN training.

Implements sliding window sampling:
- Input window (context): Tin timesteps of (N, F_dyn) where F_dyn = 4
- Output window (target): H timesteps of (N, F_out) where F_out = 3

Also handles GridName embedding concatenation (4-dim static features).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional


class PMUForecastDataset(Dataset):
    """
    Sliding window dataset for PMU forecasting.
    
    Creates samples of form:
    - Input: (Tin, N, F_dyn) where F_dyn=4 (Δf, RoCoF, Δθ, ΔV)
    - Grid embeddings: (N, 4) static features
    - Target: (H, N, F_out) where F_out=3 (Δf, Δθ, ΔV)
    """

    def __init__(self, 
                 freq_dev: np.ndarray,          # (T, N)
                 rocof: np.ndarray,             # (T, N)
                 angle_delta: np.ndarray,       # (T, N)
                 volt_dev: np.ndarray,          # (T, N)
                 grid_embeddings: np.ndarray,   # (N, 4)
                 Tin: int = 100,
                 H: int = 10,
                 stride: int = 1):
        """
        Args:
            freq_dev: Frequency deviation (T, N)
            rocof: Rate of change of frequency (T, N)
            angle_delta: Angle dynamics (T, N)
            volt_dev: Voltage residual (T, N)
            grid_embeddings: Static embeddings (N, 4)
            Tin: History length (input window size)
            H: Forecast horizon (output window size)
            stride: Sampling stride (1 for all windows, >1 for sparse sampling)
        """
        assert len(freq_dev) == len(rocof) == len(angle_delta) == len(volt_dev)
        assert freq_dev.shape[1] == rocof.shape[1] == angle_delta.shape[1] == volt_dev.shape[1]
        assert grid_embeddings.shape[0] == freq_dev.shape[1]
        
        self.T, self.N = freq_dev.shape
        self.Tin = Tin
        self.H = H
        self.stride = stride
        
        # Stack features: (T, N, 4)
        self.X = np.stack([freq_dev, rocof, angle_delta, volt_dev], axis=2)
        
        # Target is subset of features: Δf, Δθ, ΔV (not RoCoF)
        # Indices: [0] Δf, [2] Δθ, [3] ΔV
        self.y = np.stack([freq_dev, angle_delta, volt_dev], axis=2)
        
        # Static embeddings
        self.grid_emb = grid_embeddings
        
        # Valid window start indices
        self.valid_starts = []
        for t in range(0, self.T - self.Tin - self.H + 1, self.stride):
            self.valid_starts.append(t)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_input: (Tin, N, 4) input features
            x_grid: (N, 4) grid embeddings
            y_target: (H, N, 3) target features
        """
        t_start = self.valid_starts[idx]
        
        # Input window
        x_input = self.X[t_start:t_start + self.Tin].copy()  # (Tin, N, 4)
        
        # Grid embeddings (same for all samples)
        x_grid = self.grid_emb.copy()  # (N, 4)
        
        # Target window
        y_target = self.y[t_start + self.Tin:t_start + self.Tin + self.H].copy()  # (H, N, 3)
        
        return (
            torch.from_numpy(x_input).float(),
            torch.from_numpy(x_grid).float(),
            torch.from_numpy(y_target).float()
        )


def create_datasets(
    freq_dev: np.ndarray,
    rocof: np.ndarray,
    angle_delta: np.ndarray,
    volt_dev: np.ndarray,
    grid_embeddings: np.ndarray,
    Tin: int = 100,
    H: int = 10,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    stride_train: int = 1,
    stride_val: int = 1,
    stride_test: int = 1
) -> Tuple[PMUForecastDataset, PMUForecastDataset, PMUForecastDataset]:
    """
    Create train/val/test splits based on time.
    
    Args:
        train_frac: Fraction for training (0.8 = 80%)
        val_frac: Fraction for validation (0.1 = 10%)
        Test fraction is implicitly 1 - train_frac - val_frac
        stride_* : Sampling stride for each split (useful to reduce val/test size)
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    T = len(freq_dev)
    
    # Split by time
    train_end = int(T * train_frac)
    val_end = train_end + int(T * val_frac)
    
    def slice_arrays(start, end, arrays):
        return tuple(arr[start:end] for arr in arrays)
    
    freq_train, rocof_train, angle_train, volt_train = \
        slice_arrays(0, train_end, [freq_dev, rocof, angle_delta, volt_dev])
    
    freq_val, rocof_val, angle_val, volt_val = \
        slice_arrays(train_end, val_end, [freq_dev, rocof, angle_delta, volt_dev])
    
    freq_test, rocof_test, angle_test, volt_test = \
        slice_arrays(val_end, T, [freq_dev, rocof, angle_delta, volt_dev])
    
    train_ds = PMUForecastDataset(
        freq_train, rocof_train, angle_train, volt_train, grid_embeddings,
        Tin=Tin, H=H, stride=stride_train
    )
    
    val_ds = PMUForecastDataset(
        freq_val, rocof_val, angle_val, volt_val, grid_embeddings,
        Tin=Tin, H=H, stride=stride_val
    )
    
    test_ds = PMUForecastDataset(
        freq_test, rocof_test, angle_test, volt_test, grid_embeddings,
        Tin=Tin, H=H, stride=stride_test
    )
    
    return train_ds, val_ds, test_ds
