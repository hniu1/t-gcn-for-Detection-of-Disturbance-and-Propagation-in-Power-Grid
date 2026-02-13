# PMU Graph Construction Pipeline

**Pre-GNN stage**: Build physically meaningful graphs from PMU sensor data before training Graph Neural Networks for disturbance propagation prediction.

## Overview

This pipeline constructs three types of graphs from distributed PMU sensors:

1. **Geographic Graph (A_geo)**: Based on spatial proximity (k-nearest neighbors)
2. **Correlation Graph (A_corr)**: Based on temporal correlation of measurements
3. **Hybrid Graph (A_hybrid)**: Weighted combination of geographic and correlation information

## Workflow

```
Input:
├── Sensor Metadata (FDRLocation.xlsx)
│   └── FDRID, Latitude, Longitude
└── Time Series Data (data/2024-06-01/*.parquet)
    └── Frequency, VoltageAngle, VoltageMagnitude per sensor

Processing:
├── Load and match sensors with metadata
├── Build A_geo: k-NN based on lat/lon distances
├── Build A_corr: Pearson correlation between time series
├── Build A_hybrid: α·A_geo + (1-α)·A_corr
└── Sparsify and normalize

Output:
├── A_geo.npy, A_corr.npy, A_hybrid.npy
├── sensor_metadata.csv
├── PNG visualizations (static maps)
└── Interactive Folium maps (HTML)
```

## File Structure

```
FNET/
├── graph_utils.py                    # Utility functions for graph construction
├── build_pmu_graph.py               # Main pipeline script
├── build_interactive_maps.py         # Interactive visualization
├── data/
│   ├── 2024-06-01/                  # Parquet files (one per sensor)
│   └── FDRLocation.xlsx             # Sensor metadata
└── results/
    └── graphs/                       # Output directory
        ├── A_geo.npy
        ├── A_corr.npy
        ├── A_hybrid.npy
        ├── sensor_metadata.csv
        ├── geographic_distribution.png
        ├── graph_geographic.png
        ├── graph_correlation.png
        ├── graph_hybrid.png
        ├── map_geographic_interactive.html
        ├── map_correlation_interactive.html
        └── map_hybrid_interactive.html
```

## Usage

### Step 1: Build Graphs and Generate Visualizations

```bash
python build_pmu_graph.py
```

This script:
- Loads sensor metadata from `data/FDRLocation.xlsx`
- Loads time series from `data/2024-06-01/*.parquet`
- Constructs A_geo, A_corr, and A_hybrid matrices
- Generates static visualizations (PNG)
- Saves matrices and metadata to `results/graphs/`

**Parameters** (modify in `main()` function):
- `variable`: Which time series variable to use ('Frequency', 'VoltageAngle', 'VoltageMagnitude')
- `k_geo`: Number of neighbors for geographic graph (default: 5)
- `corr_threshold`: Correlation threshold (default: 0.6)
- `alpha`: Weight for hybrid graph (default: 0.5)
- `sparsify_k`: Keep top-k edges per node (default: 6)

### Step 2: Create Interactive Maps

```bash
python build_interactive_maps.py
```

Generates interactive Folium maps (HTML) for each graph type.

## Key Components

### `graph_utils.py`

#### `build_knn_adjacency(coordinates, k=5, normalize=True)`
Builds geographic adjacency matrix using k-nearest neighbors.
- Input: Sensor coordinates (N, 2)
- Output: (N, N) symmetric matrix with weights = 1/(1+distance)

#### `build_correlation_adjacency(X, threshold=0.6, remove_self_loops=True)`
Builds correlation-based adjacency matrix from time series.
- Input: Time series (T, N)
- Output: (N, N) matrix with weights = |Pearson correlation|
- Weak correlations (|r| < threshold) zeroed out

#### `build_hybrid_adjacency(A_geo, A_corr, alpha=0.5, normalize=True)`
Combines geographic and correlation graphs.
- Formula: A_hybrid = α·A_geo + (1-α)·A_corr
- Allows flexible blending: α=1 (pure geography), α=0 (pure correlation)

#### `sparsify_adjacency(A, method='threshold', threshold=0.1, k=None)`
Sparsifies dense matrices to keep only strong edges.
- Methods:
  - `'threshold'`: Keep edges > threshold
  - `'knn'`: Keep top-k edges per node (more stable for GNNs)

#### `compute_graph_statistics(A)`
Computes basic graph metrics:
- Number of nodes, edges, density
- Degree distribution (min, max, mean)
- Sparsity

### `build_pmu_graph.py`

#### `PMUGraphBuilder` Class
Main orchestrator for the pipeline.

Key methods:
- `load_metadata()`: Load sensor locations
- `load_timeseries(variable)`: Load and align time series
- `build_geographic_graph(k)`: Build A_geo
- `build_correlation_graph(threshold)`: Build A_corr
- `build_hybrid_graph(alpha)`: Build A_hybrid
- `save_matrices()`: Save to disk
- `visualize_graph(A, title)`: Static matplotlib visualization
- `run_pipeline()`: Execute all steps

## Graph Properties

### Geographic Graph (A_geo)
- **Construction**: k-NN on lat/lon coordinates
- **Weights**: Inverse distance decay
- **Use case**: Captures spatial structure of the grid
- **Edge threshold**: 0.2 (in visualization)

### Correlation Graph (A_corr)
- **Construction**: Pearson correlation of frequency/voltage time series
- **Weights**: |correlation| coefficient
- **Use case**: Captures temporal dependencies and signal coherence
- **Edge threshold**: 0.3 (in visualization)
- **Note**: Weak correlations |r| < 0.6 removed

### Hybrid Graph (A_hybrid)
- **Construction**: α·A_geo + (1-α)·A_corr
- **Alpha values**:
  - α=0.7: Emphasis on geography
  - α=0.5: Balanced
  - α=0.3: Emphasis on correlations
- **Use case**: Combines spatial and temporal information
- **Sparsification**: Keep top-6 edges per node (for GNN stability)

## Visualization

### Static Maps (PNG)
- `geographic_distribution.png`: Sensor locations only
- `graph_geographic.png`: A_geo with edges
- `graph_correlation.png`: A_corr with edges
- `graph_hybrid.png`: A_hybrid with edges

**Edge representation**:
- Color: Blue gradient (darker = stronger)
- Width: Proportional to edge weight
- Opacity: Proportional to edge weight

### Interactive Maps (HTML)
- Zoom, pan, hover for details
- Marker colors indicate node degree:
  - Gray: isolated
  - Blue: low connectivity
  - Orange: medium connectivity
  - Red: high connectivity

## Sanity Checks

After running the pipeline, verify:

1. **Matrix shapes**: All (N, N) where N = number of sensors
2. **Symmetry**: A_geo and A_corr should be symmetric (if not, fix sparsification)
3. **Values**: All weights in [0, 1]
4. **Connectivity**: 
   - A_geo should be well-connected (k-NN ensures this)
   - A_corr depends on data (may be sparse)
   - A_hybrid should balance both
5. **Visualization**: Edges should follow geographic patterns and temporal correlations

## Loading Matrices for GNN Training

```python
import numpy as np

# Load adjacency matrix
A = np.load('results/graphs/A_hybrid.npy')

# Load sensor metadata
import pandas as pd
metadata = pd.read_csv('results/graphs/sensor_metadata.csv')

# Use A as input to GNN
# A.shape = (N, N) where N = number of sensors
# A is sparse and normalized
```

## Customization

### Modify Graph Parameters

Edit `build_pmu_graph.py` main function:

```python
builder.run_pipeline(
    variable='Frequency',      # Change variable
    k_geo=6,                   # More neighbors
    corr_threshold=0.5,        # Lower threshold = more edges
    alpha=0.7,                 # More geographic weight
    sparsify_k=8              # Keep more edges
)
```

### Different Time Series Variables

Replace `variable` in `load_timeseries()`:
- `'Frequency'`: Grid frequency
- `'VoltageAngle'`: Phase angle
- `'VoltageMagnitude'`: Magnitude

### Multiple Graphs from Different Variables

Create multiple matrices:
```python
# Build graphs for different variables
for var in ['Frequency', 'VoltageAngle', 'VoltageMagnitude']:
    builder.load_timeseries(variable=var)
    builder.build_correlation_graph(threshold=0.6)
    A_corr_var = builder.A_corr.copy()
    # Save A_corr_var
```

## Troubleshooting

### Issue: "No matching time series data found"
- **Cause**: Sensor IDs in parquet filenames don't match metadata
- **Fix**: Extract sensor IDs from filenames, match with metadata FDRID column

### Issue: Sparse correlation graph
- **Cause**: Time series are not correlated (threshold too high)
- **Fix**: Lower `corr_threshold` (e.g., 0.4 or 0.5)

### Issue: Too many edges in visualization
- **Cause**: Edge threshold too low
- **Fix**: Increase `edge_threshold` in `visualize_graph()`

### Issue: Memory error with large datasets
- **Cause**: Correlation matrix computation (O(N²))
- **Fix**: Subsample sensors or use sparse matrix operations

## References

- **Haversine distance**: https://en.wikipedia.org/wiki/Haversine_formula
- **Pearson correlation**: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
- **k-NN graphs**: https://en.wikipedia.org/wiki/K-nearest_neighbors_graph
- **T-GCN**: https://arxiv.org/abs/1811.05320

## Notes

- All matrices are symmetric and normalized to [0, 1]
- Self-loops are removed (diagonal = 0)
- For very large networks, consider sparse matrix formats (scipy.sparse)
- Edge thresholds in visualization are tunable parameters
- Experiment with α and correlation thresholds for your specific use case
