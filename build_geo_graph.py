"""
Build Geographic Sensor Graph (Stage 1)
========================================

Construct a k-NN graph based on geographic proximity of active sensors.

Tasks:
1. Load sensor metadata
2. Identify 126 active sensors by matching with data files
3. Build k-NN geographic adjacency matrix using haversine distance
4. Convert distances to weights
5. Visualize and save results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import warnings

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not installed. Install with: pip install cartopy")

from matplotlib import cm
from matplotlib.colors import Normalize

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("data")
METADATA_FILE = DATA_DIR / "FDRLocation.xlsx"
TIMESERIES_DIR = DATA_DIR / "2024-06-01"
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# k-NN parameters
K_NEIGHBORS = 5
DISTANCE_SIGMA = 100.0  # km, for weight conversion
WEIGHT_THRESHOLD = 0.0  # for visualization (show all edges)
EARTH_RADIUS_KM = 6371.0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute haversine distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude/Longitude of point 1 (degrees)
        lat2, lon2: Latitude/Longitude of point 2 (degrees)
    
    Returns:
        Distance in kilometers
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def extract_sensor_id(filename):
    """
    Extract sensor ID from filename.
    
    Format: <sensor_id>-<location>-YYYYMMDD.parquet
    E.g., 1047-UsNmLasCruces1047-20240101.parquet -> 1047
    """
    stem = Path(filename).stem
    sensor_id = stem.split('-')[0]
    try:
        return int(sensor_id)
    except ValueError:
        return None


def identify_active_sensors():
    """
    Identify active sensor IDs by matching with data files.
    
    Returns:
        Set of active sensor IDs
    """
    active_ids = set()
    
    parquet_files = list(TIMESERIES_DIR.glob("*.parquet"))
    
    for f in parquet_files:
        sid = extract_sensor_id(f.name)
        if sid is None:
            continue

        # Verify the parquet file actually contains data (non-empty)
        try:
            df = pd.read_parquet(f)
            if df is not None and len(df) > 0:
                active_ids.add(sid)
            else:
                print(f"     Skipping sensor {sid}: parquet file empty: {f}")
        except Exception as e:
            print(f"     Skipping sensor {sid}: failed to read parquet {f}: {e}")
    
    return active_ids


def load_active_sensor_metadata():
    """
    Load metadata and filter to active sensors only.
    
    Returns:
        DataFrame with columns [FDRID, GridName, Latitude, Longitude]
        FDRID matches the sensor ID from parquet files
    """
    print("\n" + "="*70)
    print("STAGE 1: BUILD GEOGRAPHIC SENSOR GRAPH")
    print("="*70)
    
    # Load metadata
    print("\n[1/7] Loading metadata...")
    df_meta = pd.read_excel(METADATA_FILE)
    print(f"     Total sensors in metadata: {len(df_meta)}")
    
    # Identify active sensors
    print("\n[2/7] Identifying active sensors...")
    active_ids = identify_active_sensors()
    print(f"     Active sensors found in data files: {len(active_ids)}")
    print(f"     Example IDs: {sorted(list(active_ids))[:5]}")
    
    # Filter to active only
    print("\n[3/7] Filtering metadata to active sensors...")
    df_active = df_meta[df_meta['FDRID'].isin(active_ids)].reset_index(drop=True)
    print(f"     Sensors with valid metadata: {len(df_active)}")
    
    # Report any unmatched sensors
    unmatched = active_ids - set(df_meta['FDRID'])
    if unmatched:
        print(f"     WARNING: {len(unmatched)} sensor(s) in data files not found in metadata")
        print(f"     Unmatched IDs: {sorted(list(unmatched))[:10]}")
        if len(unmatched) > 10:
            print(f"     ... and {len(unmatched)-10} more")
    
    if len(df_active) == 0:
        raise ValueError("No active sensors matched metadata!")
    
    return df_active, active_ids


# ============================================================================
# BUILD GEOGRAPHIC GRAPH
# ============================================================================

def build_knn_graph(df_sensors, k=K_NEIGHBORS, sigma=DISTANCE_SIGMA):
    """
    Build k-nearest neighbor graph based on geographic proximity.
    
    Args:
        df_sensors: DataFrame with Latitude, Longitude columns
        k: Number of nearest neighbors
        sigma: Bandwidth parameter for weight conversion (km)
    
    Returns:
        A: Adjacency matrix (n x n, symmetric)
    """
    print("\n[4/7] Building k-NN graph (k={})...".format(k))
    
    n_sensors = len(df_sensors)
    
    # Extract coordinates
    coords = df_sensors[['Latitude', 'Longitude']].values
    
    # Use sklearn's NearestNeighbors with haversine metric
    # Convert degrees to radians for haversine
    coords_rad = np.radians(coords)
    
    nbrs = NearestNeighbors(n_neighbors=k+1,  # +1 to include self
                             algorithm='ball_tree',
                             metric='haversine',
                             radius=None)
    nbrs.fit(coords_rad)
    
    # Find neighbors (distances in radians on unit sphere)
    distances_rad, indices = nbrs.kneighbors(coords_rad)
    
    # Convert radians to kilometers
    distances_km = distances_rad * EARTH_RADIUS_KM
    
    # Build adjacency matrix
    A = np.zeros((n_sensors, n_sensors))
    
    for i in range(n_sensors):
        # Skip first neighbor (self)
        neighbors = indices[i, 1:k+1]
        dists = distances_km[i, 1:k+1]
        
        # Convert distances to weights: w = exp(-d/sigma)
        weights = np.exp(-dists / sigma)
        
        # Set adjacency
        A[i, neighbors] = weights
    
    # Make symmetric
    print("     Making adjacency matrix symmetric...")
    A = np.maximum(A, A.T)
    
    # Remove self-loops (diagonal should be 0)
    np.fill_diagonal(A, 0)
    
    print(f"     Graph shape: {A.shape}")
    print(f"     Non-zero entries: {np.count_nonzero(A)}")
    print(f"     Min weight: {A[A > 0].min():.4f}")
    print(f"     Max weight: {A[A > 0].max():.4f}")
    print(f"     Mean weight: {A[A > 0].mean():.4f}")
    
    return A


def compute_graph_stats(A):
    """
    Compute basic graph statistics.
    
    Args:
        A: Adjacency matrix
    
    Returns:
        Dictionary with statistics
    """
    n_nodes = A.shape[0]
    n_edges = np.count_nonzero(A) // 2  # Divide by 2 for undirected
    degrees = np.count_nonzero(A, axis=1)
    avg_degree = degrees.mean()
    
    # Check connectivity using BFS/simple approach
    # A graph is fully connected if all nodes are reachable from node 0
    def is_connected(adj_matrix):
        """Check if graph is connected (undirected)"""
        n = adj_matrix.shape[0]
        visited = [False] * n
        
        # BFS from node 0
        queue = [0]
        visited[0] = True
        
        while queue:
            u = queue.pop(0)
            # Find neighbors with non-zero weights
            neighbors = np.where(adj_matrix[u] > 0)[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        return all(visited)
    
    is_fully_connected = is_connected(A)
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_degree': avg_degree,
        'min_degree': degrees.min(),
        'max_degree': degrees.max(),
        'is_connected': is_fully_connected,
        'degrees': degrees
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_graph_analysis(df_sensors, A):
    """
    Create comprehensive graph analysis plots:
    - Degree distribution
    - Edge weight distribution
    - Adjacency matrix heatmap (subset)
    - Graph statistics summary
    """
    print("\n[6b/7] Creating analysis plots...")
    
    stats = compute_graph_stats(A)
    degrees = stats['degrees']
    edge_weights = A[A > 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Graph Analysis - Geographic k-NN Network', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    # 1. Degree Distribution
    ax = axes[0, 0]
    ax.hist(degrees, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(degrees.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {degrees.mean():.2f}')
    ax.axvline(np.median(degrees), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(degrees):.0f}')
    ax.set_xlabel('Node Degree', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Edge Weight Distribution
    ax = axes[0, 1]
    ax.hist(edge_weights, bins=40, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(edge_weights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {edge_weights.mean():.3f}')
    ax.axvline(np.median(edge_weights), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(edge_weights):.3f}')
    ax.set_xlabel('Edge Weight', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Adjacency Matrix Heatmap (all sensors)
    ax = axes[1, 0]
    im = ax.imshow(A, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Sensor Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sensor Index', fontsize=11, fontweight='bold')
    ax.set_title(f'Adjacency Matrix (all {A.shape[0]} sensors)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Weight', fontsize=10)
    
    # 4. Summary Statistics Text
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute additional metrics
    density = (2 * stats['n_edges']) / (stats['n_nodes'] * (stats['n_nodes'] - 1))
    clustering_proxy = "N/A"  # Could compute clustering coefficient
    
    summary_text = f"""
GRAPH STATISTICS SUMMARY
{'='*40}

Network Size:
  • Nodes (sensors): {stats['n_nodes']}
  • Total edges: {stats['n_edges']}
  • Expected max edges: {stats['n_nodes']*(stats['n_nodes']-1)//2}
  
Connectivity:
  • Fully connected: {'✓ YES' if stats['is_connected'] else '✗ NO'}
  • Network density: {density:.4f}
  
Degree Statistics:
  • Min degree: {stats['min_degree']}
  • Max degree: {stats['max_degree']}
  • Mean degree: {stats['avg_degree']:.2f}
  • Median degree: {np.median(degrees):.0f}
  • Std dev degree: {np.std(degrees):.2f}
  
Edge Weight Statistics:
  • Min weight: {edge_weights.min():.4f}
  • Max weight: {edge_weights.max():.4f}
  • Mean weight: {edge_weights.mean():.4f}
  • Median weight: {np.median(edge_weights):.4f}
  
Parameters:
  • k (neighbors): {K_NEIGHBORS}
  • σ (bandwidth): {DISTANCE_SIGMA} km
  • Weight threshold (viz): {WEIGHT_THRESHOLD}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "graph_analysis.png", dpi=150, bbox_inches='tight')
    print(f"     [✓] Saved: results/graph_analysis.png")
    plt.close()


def plot_geographic_graph(df_sensors, A, threshold=WEIGHT_THRESHOLD):
    """
    Visualize geographic sensor graph with US map as background.
    
    Args:
        df_sensors: DataFrame with Latitude, Longitude
        A: Adjacency matrix
        threshold: Only plot edges with weight > threshold
    """
    print("\n[6/7] Visualizing graph...")
    
    if HAS_CARTOPY:
        # Create figure with cartopy projection
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='#f0e5d8', alpha=0.8, zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#b8d1e8', alpha=0.8, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5, zorder=1)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray', alpha=0.3, zorder=1)
        
        # Set extent to continental US + Hawaii/Puerto Rico
        ax.set_extent([-125, -60, 20, 50], crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--', 
                         xlocs=range(-120, -60, 10), ylocs=range(20, 55, 10), zorder=2)
        gl.top_labels = False
        gl.right_labels = False
        
    else:
        # Fallback: simple matplotlib plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Set US bounds
        ax.set_xlim(-125, -60)
        ax.set_ylim(20, 50)
        ax.set_facecolor('#b8d1e8')
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    lats = df_sensors['Latitude'].values
    lons = df_sensors['Longitude'].values
    
    # Draw edges with color/thickness based on weight (before nodes)
    edge_count = 0
    
    # Get weight normalization for color mapping
    weights_all = A[A > threshold]
    if len(weights_all) > 0:
        norm = Normalize(vmin=weights_all.min(), vmax=weights_all.max())
        cmap = cm.get_cmap('plasma')  # Use plasma colormap (low=purple, high=yellow)
    else:
        norm = None
        cmap = None
    
    for i in range(len(df_sensors)):
        for j in range(i+1, len(df_sensors)):
            weight = A[i, j]
            if weight > threshold:
                # Much more dramatic variations
                alpha = min(weight * 2.5, 1.0)  # Even higher opacity
                linewidth = 0.5 + weight * 5.0  # More dramatic: 0.5-5.5 range
                
                # Get color from colormap based on weight
                if cmap is not None:
                    color = cmap(norm(weight))
                else:
                    color = '#0066FF'
                
                ax.plot([lons[i], lons[j]], [lats[i], lats[j]],
                       color=color, alpha=alpha, linewidth=linewidth, 
                       zorder=3, transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
                edge_count += 1
    
    # Draw nodes (on top of edges)
    scatter = ax.scatter(lons, lats, c='red', s=120, edgecolors='darkred', 
                        linewidth=1.5, alpha=0.95, zorder=5, 
                        label='Sensors',
                        transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
    
    # No sensor ID labels - clean visualization
    
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    # Use actual sensor count in title
    n_sensors = len(df_sensors)
    ax.set_title(f'Geographic Sensor Graph with k-NN Edges ({n_sensors} Active Sensors with Metadata)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11)
    
    # Add info text box - BOTTOM RIGHT
    stats = compute_graph_stats(A)
    edge_weights = A[A > 0]
    info_text = (f"Nodes: {stats['n_nodes']}\n"
                f"Total edges: {stats['n_edges']}\n"
                f"Edges shown: {edge_count}\n"
                f"Avg degree: {stats['avg_degree']:.2f}\n"
                f"Fully connected: {'✓' if stats['is_connected'] else '✗'}\n"
                f"Avg edge weight: {edge_weights.mean():.3f}\n"
                f"k={K_NEIGHBORS}, σ={DISTANCE_SIGMA}km")
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95, pad=0.8))
    
    # Add colorbar for edge weights
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, fraction=0.046, aspect=30)
    cbar.set_label('Edge Weight', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "geo_graph.png", dpi=150, bbox_inches='tight')
    print(f"     [✓] Saved: results/geo_graph.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute complete geographic graph construction"""
    
    # Load and filter metadata
    df_sensors, active_ids = load_active_sensor_metadata()
    
    # Build k-NN graph
    A_geo = build_knn_graph(df_sensors, k=K_NEIGHBORS, sigma=DISTANCE_SIGMA)
    
    # Compute statistics
    print("\n[5/7] Computing graph statistics...")
    stats = compute_graph_stats(A_geo)
    
    # Save adjacency matrix
    print("\n[7/7] Saving results...")
    np.save(RESULTS_DIR / "A_geo.npy", A_geo)
    print(f"     [✓] Saved: results/A_geo.npy")
    # Save sensor ordering (matches rows/cols of A_geo.npy)
    sensor_order = df_sensors['FDRID'].values
    np.save(RESULTS_DIR / "sensor_order.npy", sensor_order)
    (RESULTS_DIR / "sensor_order.csv").write_text('\n'.join(map(str, sensor_order)))
    print(f"     [✓] Saved: results/sensor_order.npy and sensor_order.csv")
    
    # Visualize geographic graph
    plot_geographic_graph(df_sensors, A_geo, threshold=WEIGHT_THRESHOLD)
    
    # Visualize graph analysis
    plot_graph_analysis(df_sensors, A_geo)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nGraph Construction Results:")
    print(f"  Number of nodes (sensors):     {stats['n_nodes']}")
    print(f"  Number of edges (k={K_NEIGHBORS}):           {stats['n_edges']}")
    print(f"  Average degree:                {stats['avg_degree']:.2f}")
    print(f"  Min degree:                    {stats['min_degree']}")
    print(f"  Max degree:                    {stats['max_degree']}")
    print(f"  Graph fully connected:         {'Yes' if stats['is_connected'] else 'No'}")
    
    print(f"\nParameters:")
    print(f"  k (nearest neighbors):         {K_NEIGHBORS}")
    print(f"  sigma (weight bandwidth):      {DISTANCE_SIGMA} km")
    print(f"  weight threshold (viz):        {WEIGHT_THRESHOLD}")
    
    print(f"\nOutputs:")
    print(f"  [✓] results/A_geo.npy          (Adjacency matrix, shape {A_geo.shape})")
    print(f"  [✓] results/geo_graph.png      (Geographic visualization with map)")
    print(f"  [✓] results/graph_analysis.png (Degree/weight distributions + stats)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
