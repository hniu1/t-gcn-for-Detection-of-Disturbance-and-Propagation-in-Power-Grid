"""
Diagnostic script to verify adjacency matrix ordering and identify sensor 797's position.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # Load adjacency matrix
    A_geo = np.load('../results/A_geo.npy')
    print(f"A_geo.npy shape: {A_geo.shape}")
    
    # Load metadata to get sensor ordering
    metadata = pd.read_excel('../data/FDRLocation.xlsx')
    print(f"Total sensors in metadata: {len(metadata)}")
    
    # Identify active sensors from parquet files
    data_path = Path('../data/2024-06-01')
    parquet_files = sorted(data_path.glob("*.parquet"))
    
    def extract_sensor_id(fname):
        """Extract sensor ID from parquet filename"""
        stem = Path(fname).stem
        try:
            return int(stem.split('-')[0])
        except (ValueError, IndexError):
            return None
    
    active_ids = sorted(set(
        sid for sid in (extract_sensor_id(f.name) for f in parquet_files)
        if sid is not None
    ))
    print(f"\nActive sensors in data: {len(active_ids)}")
    print(f"Example active IDs: {active_ids[:5]}")
    
    # Filter metadata to active sensors (PRESERVING ORIGINAL ORDER FROM XLSX)
    metadata_matched = metadata[metadata['FDRID'].isin(active_ids)]
    metadata_matched = metadata_matched.reset_index(drop=True)
    sensor_ids_in_order = metadata_matched['FDRID'].values
    
    print(f"\nSensors in A_geo.npy (in xlsx order):")
    print(f"  Count: {len(sensor_ids_in_order)}")
    print(f"  First 5: {sensor_ids_in_order[:5]}")
    print(f"  Last 5: {sensor_ids_in_order[-5:]}")
    
    # Find sensor 797's position
    if 797 in sensor_ids_in_order:
        idx_797 = np.where(sensor_ids_in_order == 797)[0][0]
        print(f"\n✓ Sensor 797 is at index {idx_797} in A_geo.npy")
    else:
        print(f"\n✗ Sensor 797 NOT in A_geo.npy!")
        print(f"  Available sensors: {sensor_ids_in_order}")
        return
    
    # Now simulate what train.py does
    print("\n" + "="*60)
    print("Simulating train.py behavior:")
    print("="*60)
    
    # Try loading data for each sensor
    loaded_count = 0
    failed_sensors = []
    loaded_sensor_ids = []
    
    for i, sid in enumerate(sensor_ids_in_order):
        # Find parquet file for this sensor
        files = list(data_path.glob(f"{sid}-*.parquet"))
        if not files:
            failed_sensors.append((sid, "No parquet file"))
            continue
        
        # Try to load
        try:
            df = pd.read_parquet(files[0])
            if len(df) == 0:
                failed_sensors.append((sid, "Empty file"))
            else:
                loaded_count += 1
                loaded_sensor_ids.append(sid)
        except Exception as e:
            failed_sensors.append((sid, str(e)))
    
    print(f"\nLoaded successfully: {loaded_count}")
    print(f"Failed: {len(failed_sensors)}")
    
    if failed_sensors:
        print("\nFailed sensors:")
        for sid, reason in failed_sensors:
            idx = np.where(sensor_ids_in_order == sid)[0][0] if sid in sensor_ids_in_order else -1
            print(f"  Sensor {sid} (index {idx}): {reason}")
    
    N_loaded = loaded_count
    N_original = len(sensor_ids_in_order)
    
    print(f"\n" + "="*60)
    print("ADJACENCY MATRIX HANDLING:")
    print("="*60)
    print(f"Original A_geo.npy: {N_original}×{N_original}")
    print(f"Can actually load: {N_loaded} sensors")
    print(f"Mismatch: {N_original - N_loaded} sensor(s)")
    
    # Current approach
    print(f"\nCurrent train.py approach: A_norm[:N_loaded, :N_loaded]")
    print(f"  → Takes top-left {N_loaded}×{N_loaded} submatrix")
    print(f"  → DROPS sensors at indices {N_loaded} to {N_original-1}")
    
    if 797 in failed_sensors:
        idx_797 = np.where(sensor_ids_in_order == 797)[0][0]
        print(f"\n⚠️  Sensor 797 is at index {idx_797}")
        if idx_797 < N_loaded:
            print(f"    ERROR: Sensor 797 would be INCLUDED in the {N_loaded}×{N_loaded} submatrix")
            print(f"    This will cause data loading error in training!")
        else:
            print(f"    ✓ Sensor 797 would be dropped by submatrix (good)")
    
    # Recommended fix
    print(f"\n" + "="*60)
    print("RECOMMENDED FIX:")
    print("="*60)
    print(f"\n1. Properly exclude failed sensors from adjacency:")
    failed_indices = []
    for sid, _ in failed_sensors:
        if sid in sensor_ids_in_order:
            idx = np.where(sensor_ids_in_order == sid)[0][0]
            failed_indices.append(idx)
    
    good_indices = [i for i in range(N_original) if i not in failed_indices]
    print(f"   Keep indices: {good_indices}")
    print(f"   Drop indices: {failed_indices}")
    print(f"   Result: {len(good_indices)}×{len(good_indices)} adjacency (with proper sensor mapping)")
    
    print(f"\n2. OR rebuild A_geo.npy from scratch:")
    print(f"   Run: cd .. && python build_geo_graph.py")
    print(f"   This will create new A_{len(good_indices)}.npy")


if __name__ == '__main__':
    main()
