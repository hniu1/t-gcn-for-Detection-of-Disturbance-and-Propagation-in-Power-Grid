"""
Data processing for multi-feature T-GCN training.

Loads sensor time-series data and computes dynamic features:
- Frequency deviation: Δf = f - 60
- RoCoF: df/dt (first difference)
- Angle dynamics: Δθ (unwrapped angle difference)
- Voltage residual: ΔV = V - per-sensor baseline

Also loads metadata (GridName) for static embeddings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List


class PMUDataProcessor:
    """Preprocess PMU sensor data for multi-feature forecasting."""

    def __init__(self, data_dir: Path, metadata_file: Path, inferred_locs_file: Path = None):
        """
        Args:
            data_dir: Path to parquet files (e.g., data/2024-06-01)
            metadata_file: Path to FDRLocation.xlsx
            inferred_locs_file: Optional path to inferred_locations.csv for missing sensors
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = Path(metadata_file)
        self.inferred_locs_file = Path(inferred_locs_file) if inferred_locs_file else None
        
        self.metadata = None
        self.inferred_locs = None
        self.sensor_ids = None
        self.grid_names = None
        self.grid_name_to_idx = None

    def load_metadata(self):
        """Load sensor metadata from Excel."""
        try:
            self.metadata = pd.read_excel(self.metadata_file)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata = pd.DataFrame()
        
        if self.inferred_locs_file and self.inferred_locs_file.exists():
            try:
                self.inferred_locs = pd.read_csv(self.inferred_locs_file)
            except Exception as e:
                print(f"Error loading inferred locations: {e}")
                self.inferred_locs = None

    def get_grid_name(self, sensor_id: int) -> str:
        """Get GridName for a sensor (from metadata or inferred)."""
        if self.metadata is not None and not self.metadata.empty:
            row = self.metadata[self.metadata['FDRID'].astype(int) == sensor_id]
            if not row.empty:
                return row.iloc[0].get('GridName', f'sensor_{sensor_id}')
        
        if self.inferred_locs is not None and not self.inferred_locs.empty:
            row = self.inferred_locs[self.inferred_locs['FDRID'].astype(int) == sensor_id]
            if not row.empty:
                return row.iloc[0].get('Filename', f'sensor_{sensor_id}')
        
        return f'sensor_{sensor_id}'

    def build_grid_map(self, sensor_ids: List[int]):
        """Create mapping from unique GridNames to indices."""
        grid_names = []
        for sid in sensor_ids:
            gn = self.get_grid_name(sid)
            grid_names.append(gn)
        
        unique_grids = sorted(set(grid_names))
        self.grid_name_to_idx = {gn: i for i, gn in enumerate(unique_grids)}
        self.grid_names = grid_names
        return self.grid_name_to_idx, grid_names

    def load_sensor_data(self, sensor_id: int, columns: List[str] = None) -> pd.DataFrame:
        """Load time-series for a single sensor."""
        if columns is None:
            columns = ['Frequency', 'VoltageAngle', 'VoltageMagnitude', 'ReceivedTime']
        
        fname = self.data_dir / f'{sensor_id}-*.parquet'
        files = list(self.data_dir.glob(f'{sensor_id}-*.parquet'))
        
        if not files:
            raise FileNotFoundError(f'No parquet file found for sensor {sensor_id}')
        
        try:
            df = pd.read_parquet(files[0], columns=columns)
        except Exception as e:
            print(f"Error reading {files[0]}: {e}")
            return None
        
        # Ensure datetime and sort
        df['ReceivedTime'] = pd.to_datetime(df['ReceivedTime'])
        df = df.sort_values('ReceivedTime').reset_index(drop=True)
        
        return df

    def compute_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute dynamic features from raw sensor data.
        
        Returns dict with keys: 'freq_dev', 'rocof', 'angle_delta', 'volt_dev'
        Each is a 1D numpy array (T,).
        """
        freq = df['Frequency'].values.astype(float)
        angle = df['VoltageAngle'].values.astype(float)
        volt = df['VoltageMagnitude'].values.astype(float)
        
        T = len(freq)
        
        # Frequency deviation
        freq_dev = freq - 60.0
        
        # RoCoF (Rate of Change of Frequency): df/dt
        rocof = np.zeros(T)
        rocof[1:] = np.diff(freq_dev)
        rocof[0] = rocof[1]  # Replicate first value
        
        # Angle dynamics: unwrap and compute difference
        angle_unwrap = np.unwrap(angle)
        angle_delta = np.zeros(T)
        angle_delta[1:] = np.diff(angle_unwrap)
        angle_delta[0] = angle_delta[1]
        
        # Voltage residual: V - per-sensor baseline (use mean)
        volt_baseline = np.nanmean(volt)
        volt_dev = volt - volt_baseline
        
        return {
            'freq_dev': freq_dev,
            'rocof': rocof,
            'angle_delta': angle_delta,
            'volt_dev': volt_dev
        }

    def prepare_tensor(self, sensor_ids: List[int], features_dict: Dict[int, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stack features across sensors into 2D arrays.
        
        Args:
            sensor_ids: List of sensor IDs (defines ordering)
            features_dict: {sensor_id: {feature_name: array}}
        
        Returns:
            (freq_dev, rocof, angle_delta, volt_dev) each shape (T, N)
        """
        # Get min length (in case sensors have different time ranges)
        min_len = min(len(features_dict[sid]['freq_dev']) for sid in sensor_ids)
        
        N = len(sensor_ids)
        
        freq_dev = np.zeros((min_len, N))
        rocof = np.zeros((min_len, N))
        angle_delta = np.zeros((min_len, N))
        volt_dev = np.zeros((min_len, N))
        
        for i, sid in enumerate(sensor_ids):
            feats = features_dict[sid]
            freq_dev[:, i] = feats['freq_dev'][:min_len]
            rocof[:, i] = feats['rocof'][:min_len]
            angle_delta[:, i] = feats['angle_delta'][:min_len]
            volt_dev[:, i] = feats['volt_dev'][:min_len]
        
        return freq_dev, rocof, angle_delta, volt_dev


def prepare_data(data_dir: Path, metadata_file: Path, inferred_locs_file: Path = None, 
                 sensor_ids: List[int] = None) -> Tuple[Dict, np.ndarray, Dict, List[str]]:
    """
    Convenience function to load and prepare all sensor data.
    
    Returns:
        (features_dict, sensor_ids_arr, grid_map, grid_names)
        - features_dict: {sid: {feature_name: array}}
        - sensor_ids_arr: ordered array of sensor IDs
        - grid_map: {grid_name_str: index}
        - grid_names: list of grid name (per sensor in sensor_ids order)
    """
    proc = PMUDataProcessor(data_dir, metadata_file, inferred_locs_file)
    proc.load_metadata()
    
    # Find available sensors if not specified
    if sensor_ids is None:
        parquet_files = list(Path(data_dir).glob('*.parquet'))
        sensor_ids = sorted(set(int(f.stem.split('-')[0]) for f in parquet_files))
    
    # Build grid name map (grid_names preserves order per sensor_ids)
    grid_map, grid_names = proc.build_grid_map(sensor_ids)
    
    # Load features for each sensor
    features_dict = {}
    valid_ids = []
    
    for sid in sensor_ids:
        try:
            df = proc.load_sensor_data(sid)
            if df is not None and len(df) > 0:
                feats = proc.compute_features(df)
                features_dict[sid] = feats
                valid_ids.append(sid)
        except Exception as e:
            print(f"Skipping sensor {sid}: {e}")
    
    return features_dict, np.array(valid_ids), grid_map, grid_names
