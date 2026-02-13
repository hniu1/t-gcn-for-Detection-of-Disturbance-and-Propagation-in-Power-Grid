"""
Pre-GNN Graph Construction Pipeline for PMU Sensor Data.

This script:
1. Loads sensor metadata (location, ID)
2. Loads multivariate time series data
3. Constructs geographic and correlation-based adjacency matrices
4. Creates a hybrid graph
5. Visualizes and saves the results
"""

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import warnings

from graph_utils import (
    build_knn_adjacency,
    build_correlation_adjacency,
    build_hybrid_adjacency,
    sparsify_adjacency,
    compute_graph_statistics
)

warnings.filterwarnings('ignore')


class PMUGraphBuilder:
    """Main class for constructing graphs from PMU sensor data."""
    
    def __init__(self, metadata_path, timeseries_dir, results_dir='results/graphs'):
        """
        Initialize the graph builder.
        
        Parameters:
        -----------
        metadata_path : str
            Path to FDRLocation.xlsx or CSV with sensor metadata
        timeseries_dir : str
            Directory containing parquet files with time series data
        results_dir : str
            Directory to save outputs
        """
        self.metadata_path = metadata_path
        self.timeseries_dir = timeseries_dir
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Data storage
        self.metadata = None
        self.sensor_ids = None
        self.coordinates = None
        self.timeseries_data = None
        self.X = None  # Time series array (T, N)
        
        # Graph matrices
        self.A_geo = None
        self.A_corr = None
        self.A_hybrid = None
        
        print("PMUGraphBuilder initialized.")
        print(f"Results will be saved to: {results_dir}")
    
    def load_metadata(self):
        """Load sensor metadata from Excel or CSV."""
        print("\n[1/6] Loading sensor metadata...")
        
        if self.metadata_path.endswith('.xlsx'):
            self.metadata = pd.read_excel(self.metadata_path)
        else:
            self.metadata = pd.read_csv(self.metadata_path)
        
        # Check required columns
        required_cols = ['FDRID', 'GridName', 'Latitude', 'Longitude']
        if not all(col in self.metadata.columns for col in required_cols):
            # Try case-insensitive matching
            cols_lower = {col.lower(): col for col in self.metadata.columns}
            col_mapping = {}
            for req in required_cols:
                if req.lower() in cols_lower:
                    col_mapping[req] = cols_lower[req.lower()]
            if len(col_mapping) == len(required_cols):
                self.metadata.rename(columns={v: k for k, v in col_mapping.items()}, inplace=True)
        
        # Extract sensor IDs and coordinates
        self.sensor_ids = self.metadata['FDRID'].astype(str).values
        self.coordinates = self.metadata[['Latitude', 'Longitude']].values
        
        print(f"  Loaded {len(self.sensor_ids)} sensors.")
        print(f"  Sensor ID examples: {self.sensor_ids[:5]}")
        print(f"  Coordinate range:")
        print(f"    Latitude:  [{self.coordinates[:, 0].min():.2f}, {self.coordinates[:, 0].max():.2f}]")
        print(f"    Longitude: [{self.coordinates[:, 1].min():.2f}, {self.coordinates[:, 1].max():.2f}]")
        
        return self
    
    def load_timeseries(self, variable='Frequency', max_sensors=None):
        """
        Load time series data from parquet files.
        
        Parameters:
        -----------
        variable : str
            Which variable to extract ('Frequency', 'VoltageAngle', 'VoltageMagnitude')
        max_sensors : int
            Maximum number of sensors to load (for debugging)
        
        Returns:
        --------
        self
        """
        print(f"\n[2/6] Loading time series data ({variable})...")
        
        parquet_files = sorted(glob.glob(os.path.join(self.timeseries_dir, '*.parquet')))
        
        if max_sensors:
            parquet_files = parquet_files[:max_sensors]
        
        timeseries_dict = {}
        
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path, engine='pyarrow')
                sensor_id = os.path.basename(file_path).split('-')[0]
                
                # Extract the variable and flatten
                if variable in df.columns:
                    timeseries_dict[sensor_id] = df[variable].values
            except Exception as e:
                print(f"  Warning: Could not load {file_path}: {e}")
        
        # Match with metadata sensor IDs
        matched_data = []
        matched_ids = []
        
        for sid in self.sensor_ids:
            if str(sid) in timeseries_dict:
                matched_data.append(timeseries_dict[str(sid)])
                matched_ids.append(sid)
        
        if len(matched_data) == 0:
            raise ValueError("No matching time series data found for sensors in metadata!")
        
        # Create time series array: (n_timestamps, n_sensors)
        # Pad sequences to same length
        max_len = max(len(x) for x in matched_data)
        X = np.full((max_len, len(matched_data)), np.nan)
        
        for i, ts in enumerate(matched_data):
            X[:len(ts), i] = ts
        
        # Use forward fill for missing values
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = np.nanmean(X[:, i])
        
        self.X = X
        self.sensor_ids = matched_ids
        self.coordinates = self.metadata[self.metadata['FDRID'].astype(str).isin(matched_ids)][['Latitude', 'Longitude']].values
        
        print(f"  Loaded time series for {len(matched_ids)} sensors.")
        print(f"  Shape: {X.shape}")
        print(f"  Variable range: [{np.nanmin(X):.4f}, {np.nanmax(X):.4f}]")
        
        return self
    
    def build_geographic_graph(self, k=5):
        """
        Build geographic adjacency matrix using k-NN.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        
        Returns:
        --------
        self
        """
        print(f"\n[3/6] Building geographic adjacency matrix (k={k})...")
        
        self.A_geo = build_knn_adjacency(self.coordinates, k=k, normalize=True)
        
        stats = compute_graph_statistics(self.A_geo)
        print(f"  Geographic graph stats:")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
            else:
                print(f"    {key}: {val}")
        
        return self
    
    def build_correlation_graph(self, threshold=0.6):
        """
        Build correlation-based adjacency matrix.
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold
        
        Returns:
        --------
        self
        """
        print(f"\n[4/6] Building correlation adjacency matrix (threshold={threshold})...")
        
        self.A_corr = build_correlation_adjacency(self.X, threshold=threshold, remove_self_loops=True)
        
        stats = compute_graph_statistics(self.A_corr)
        print(f"  Correlation graph stats:")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
            else:
                print(f"    {key}: {val}")
        
        return self
    
    def build_hybrid_graph(self, alpha=0.5, sparsify_method='knn', sparsify_k=6):
        """
        Build hybrid graph combining geographic and correlation information.
        
        Parameters:
        -----------
        alpha : float
            Weight: A_hybrid = alpha * A_geo + (1-alpha) * A_corr
        sparsify_method : str
            How to sparsify: 'threshold' or 'knn'
        sparsify_k : int
            For knn sparsification
        
        Returns:
        --------
        self
        """
        print(f"\n[5/6] Building hybrid adjacency matrix (alpha={alpha})...")
        
        self.A_hybrid = build_hybrid_adjacency(self.A_geo, self.A_corr, alpha=alpha, normalize=True)
        
        # Sparsify
        self.A_hybrid = sparsify_adjacency(self.A_hybrid, method=sparsify_method, k=sparsify_k)
        
        stats = compute_graph_statistics(self.A_hybrid)
        print(f"  Hybrid graph stats:")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
            else:
                print(f"    {key}: {val}")
        
        return self
    
    def save_matrices(self):
        """Save adjacency matrices to disk."""
        print(f"\n[6/6] Saving adjacency matrices...")
        
        if self.A_geo is not None:
            np.save(os.path.join(self.results_dir, 'A_geo.npy'), self.A_geo)
            print(f"  Saved: A_geo.npy")
        
        if self.A_corr is not None:
            np.save(os.path.join(self.results_dir, 'A_corr.npy'), self.A_corr)
            print(f"  Saved: A_corr.npy")
        
        if self.A_hybrid is not None:
            np.save(os.path.join(self.results_dir, 'A_hybrid.npy'), self.A_hybrid)
            print(f"  Saved: A_hybrid.npy")
        
        # Save sensor metadata
        metadata_out = pd.DataFrame({
            'SensorID': self.sensor_ids,
            'Latitude': self.coordinates[:, 0],
            'Longitude': self.coordinates[:, 1]
        })
        metadata_out.to_csv(os.path.join(self.results_dir, 'sensor_metadata.csv'), index=False)
        print(f"  Saved: sensor_metadata.csv")
    
    def visualize_geographic(self, figsize=(14, 10)):
        """
        Visualize sensor locations on a map.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax
        """
        print(f"\n  Visualizing geographic distribution...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot sensors as points
        ax.scatter(self.coordinates[:, 1], self.coordinates[:, 0], 
                  c='red', s=50, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('PMU Sensor Geographic Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_graph(self, A, title='Graph Visualization', figsize=(14, 10), 
                       edge_threshold=0.1, edge_width_scale=2):
        """
        Visualize graph with edges on geographic map.
        
        Parameters:
        -----------
        A : ndarray
            Adjacency matrix
        title : str
            Figure title
        figsize : tuple
            Figure size
        edge_threshold : float
            Only draw edges above this weight
        edge_width_scale : float
            Scale factor for edge widths
        
        Returns:
        --------
        fig, ax
        """
        print(f"  Visualizing {title}...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot edges
        n_edges_drawn = 0
        for i in range(A.shape[0]):
            for j in range(i+1, A.shape[1]):
                if A[i, j] > edge_threshold:
                    # Draw edge
                    lat1, lon1 = self.coordinates[i]
                    lat2, lon2 = self.coordinates[j]
                    
                    alpha = min(A[i, j], 1.0)
                    width = A[i, j] * edge_width_scale
                    
                    ax.plot([lon1, lon2], [lat1, lat2], 'b-', 
                           alpha=alpha*0.5, linewidth=width, zorder=1)
                    n_edges_drawn += 1
        
        # Plot sensors
        ax.scatter(self.coordinates[:, 1], self.coordinates[:, 0], 
                  c='red', s=100, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3,
                  label='Sensors')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{title}\n({n_edges_drawn} edges visualized)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig, ax
    
    def run_pipeline(self, variable='Frequency', k_geo=5, corr_threshold=0.6, 
                    alpha=0.5, sparsify_k=6):
        """
        Run the complete pipeline.
        
        Parameters:
        -----------
        variable : str
            Time series variable to use
        k_geo : int
            Number of neighbors for geographic graph
        corr_threshold : float
            Correlation threshold
        alpha : float
            Hybrid graph weight
        sparsify_k : int
            Sparsification k for knn method
        
        Returns:
        --------
        self
        """
        print("\n" + "="*60)
        print("PMU GRAPH CONSTRUCTION PIPELINE")
        print("="*60)
        
        self.load_metadata()
        self.load_timeseries(variable=variable)
        self.build_geographic_graph(k=k_geo)
        self.build_correlation_graph(threshold=corr_threshold)
        self.build_hybrid_graph(alpha=alpha, sparsify_k=sparsify_k)
        self.save_matrices()
        
        print("\n" + "="*60)
        print("Pipeline complete!")
        print("="*60)
        
        return self


def main():
    """Main execution."""
    
    # Paths
    metadata_path = 'data/FDRLocation.xlsx'
    timeseries_dir = 'data/2024-06-01'
    results_dir = 'results/graphs'
    
    # Initialize builder
    builder = PMUGraphBuilder(metadata_path, timeseries_dir, results_dir)
    
    # Run pipeline
    builder.run_pipeline(
        variable='Frequency',
        k_geo=5,
        corr_threshold=0.6,
        alpha=0.5,
        sparsify_k=6
    )
    
    # Visualize geographic distribution
    fig1, ax1 = builder.visualize_geographic()
    fig1.savefig(os.path.join(results_dir, 'geographic_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: geographic_distribution.png")
    
    # Visualize each graph
    fig2, ax2 = builder.visualize_graph(builder.A_geo, title='Geographic Graph (A_geo)', 
                                        edge_threshold=0.2)
    fig2.savefig(os.path.join(results_dir, 'graph_geographic.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: graph_geographic.png")
    
    fig3, ax3 = builder.visualize_graph(builder.A_corr, title='Correlation Graph (A_corr)', 
                                        edge_threshold=0.3)
    fig3.savefig(os.path.join(results_dir, 'graph_correlation.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: graph_correlation.png")
    
    fig4, ax4 = builder.visualize_graph(builder.A_hybrid, title='Hybrid Graph (A_hybrid)', 
                                        edge_threshold=0.2)
    fig4.savefig(os.path.join(results_dir, 'graph_hybrid.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: graph_hybrid.png")
    
    plt.close('all')
    
    print(f"\nAll outputs saved to: {results_dir}")


if __name__ == '__main__':
    main()
