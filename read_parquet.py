import pandas as pd
import os
import glob

# Directory containing the parquet files
dir_path = '2024-06-01'

# Get all parquet files in the directory
parquet_files = glob.glob(os.path.join(dir_path, '*.parquet'))

# Loop through each file and read it
for file_path in parquet_files:
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"Successfully read {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        print("-" * 50)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print("Finished reading all parquet files.")