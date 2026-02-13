"""
Example script: Loading and using the pre-computed graphs.

Demonstrates how to:
1. Load adjacency matrices
2. Extract graph statistics
3. Prepare for GNN training (PyTorch Geometric format)
4. Basic graph analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os


def load_graphs(results_dir='results/graphs'):
    """Load all pre-computed adjacency matrices."""
    
    print("Loading pre-computed graphs...")
    
    A_geo = np.load(os.path.join(results_dir, 'A_geo.npy'))
    A_corr = np.load(os.path.join(results_dir, 'A_corr.npy'))
    A_hybrid = np.load(os.path.join(results_dir, 'A_hybrid.npy'))
    metadata = pd.read_csv(os.path.join(results_dir, 'sensor_metadata.csv'))
    
    print(f"  A_geo: {A_geo.shape}, edges={np.count_nonzero(np.triu(A_geo, k=1))}")
    print(f"  A_corr: {A_corr.shape}, edges={np.count_nonzero(np.triu(A_corr, k=1))}")
    print(f"  A_hybrid: {A_hybrid.shape}, edges={np.count_nonzero(np.triu(A_hybrid, k=1))}")
    print(f"  Sensors: {len(metadata)}")
    
    return A_geo, A_corr, A_hybrid, metadata


def adjacency_to_coo(A):
    """Convert dense adjacency matrix to COO format (edges list)."""
    
    i, j = np.nonzero(A)
    weights = A[i, j]
    
    return i, j, weights


def adjacency_to_pytorch_geometric(A):
    """Convert adjacency matrix to PyTorch Geometric format.
    
    Returns:
    --------
    edge_index : ndarray, shape (2, n_edges)
        Edge list in COO format (both directions)
    edge_weight : ndarray, shape (n_edges,)
        Edge weights (both directions)
    """
    
    import torch
    
    i, j = np.nonzero(A)
    weights = A[i, j]
    
    edge_index = torch.tensor([i, j], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    return edge_index, edge_weight


def compute_node_degrees(A):
    """Compute degree of each node."""
    
    degrees = np.sum(A > 0, axis=1)
    return degrees


def get_node_neighbors(A, node_id, threshold=0.0):
    """Get neighbors of a specific node.
    
    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    node_id : int
        Node index (0 to n_sensors-1)
    threshold : float
        Only return neighbors with edge weight > threshold
    
    Returns:
    --------
    neighbors : list of (neighbor_id, weight) tuples
    """
    
    neighbors = []
    for j in range(A.shape[1]):
        if A[node_id, j] > threshold:
            neighbors.append((j, A[node_id, j]))
    
    # Sort by weight (descending)
    neighbors.sort(key=lambda x: x[1], reverse=True)
    
    return neighbors


def plot_degree_distribution(A_dict, title='Degree Distribution'):
    """Plot degree distribution for multiple graphs.
    
    Parameters:
    -----------
    A_dict : dict
        Dictionary {graph_name: adjacency_matrix}
    title : str
        Plot title
    """
    
    fig, axes = plt.subplots(1, len(A_dict), figsize=(15, 4))
    
    if len(A_dict) == 1:
        axes = [axes]
    
    for ax, (name, A) in zip(axes, A_dict.items()):
        degrees = np.sum(A > 0, axis=1)
        
        ax.hist(degrees, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\n(mean={degrees.mean():.2f}, max={degrees.max()})')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_node_summary(A, metadata, node_id):
    """Print detailed information about a specific node."""
    
    sensor_id = metadata.iloc[node_id]['SensorID']
    lat = metadata.iloc[node_id]['Latitude']
    lon = metadata.iloc[node_id]['Longitude']
    
    degree = np.sum(A > 0)
    neighbors = get_node_neighbors(A, node_id)
    
    print(f"\nNode {node_id} (Sensor {sensor_id}):")
    print(f"  Location: ({lat:.2f}, {lon:.2f})")
    print(f"  Degree: {degree}")
    print(f"  Top 5 Neighbors:")
    
    for neighbor_id, weight in neighbors[:5]:
        neighbor_sensor = metadata.iloc[neighbor_id]['SensorID']
        print(f"    - Node {neighbor_id} (Sensor {neighbor_sensor}): weight={weight:.4f}")


def example_workflow():
    """Complete example workflow."""
    
    print("="*70)
    print("PRE-GNN GRAPH LOADING AND USAGE EXAMPLE")
    print("="*70)
    
    # Step 1: Load graphs
    print("\n[1] Loading graphs...")
    A_geo, A_corr, A_hybrid, metadata = load_graphs()
    
    # Step 2: Basic statistics
    print("\n[2] Computing statistics...")
    
    for name, A in [('A_geo', A_geo), ('A_corr', A_corr), ('A_hybrid', A_hybrid)]:
        degrees = compute_node_degrees(A)
        n_edges = np.count_nonzero(np.triu(A, k=1))
        density = 2 * n_edges / (A.shape[0] * (A.shape[0] - 1))
        
        print(f"\n{name}:")
        print(f"  Nodes: {A.shape[0]}")
        print(f"  Edges: {n_edges}")
        print(f"  Density: {density:.4f}")
        print(f"  Avg degree: {degrees.mean():.2f}")
        print(f"  Min/Max degree: {degrees.min()}/{degrees.max()}")
    
    # Step 3: Convert to PyTorch format (example)
    print("\n[3] Converting to PyTorch Geometric format...")
    
    try:
        import torch
        edge_index, edge_weight = adjacency_to_pytorch_geometric(A_hybrid)
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Edge weight shape: {edge_weight.shape}")
        print(f"  Example edges:")
        for i in range(min(5, edge_index.shape[1])):
            print(f"    ({edge_index[0, i].item()}, {edge_index[1, i].item()}): {edge_weight[i].item():.4f}")
    except ImportError:
        print("  PyTorch not installed, skipping...")
    
    # Step 4: Node neighborhood analysis
    print("\n[4] Analyzing node neighborhoods (A_hybrid)...")
    
    # Find highly connected node
    degrees = compute_node_degrees(A_hybrid)
    top_node = np.argmax(degrees)
    
    print_node_summary(A_hybrid, metadata, top_node)
    
    # Find another node
    if len(metadata) > 50:
        print_node_summary(A_hybrid, metadata, 50)
    
    # Step 5: Visualizations
    print("\n[5] Generating visualizations...")
    
    A_dict = {
        'A_geo': A_geo,
        'A_corr': A_corr,
        'A_hybrid': A_hybrid
    }
    
    fig = plot_degree_distribution(A_dict)
    output_path = 'results/graphs/degree_distribution_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    # Step 6: Save sparse matrix version
    print("\n[6] Saving sparse matrix versions...")
    
    for name, A in A_dict.items():
        sparse_A = csr_matrix(A)
        output_path = f'results/graphs/{name}_sparse.npz'
        from scipy.sparse import save_npz
        save_npz(output_path, sparse_A)
        print(f"  Saved: {output_path} (format: CSR sparse, {sparse_A.nnz} non-zeros)")
    
    # Step 7: Example: Ready for GNN
    print("\n[7] Ready for GNN training!")
    print("  Code example:")
    print("""
    from torch_geometric.data import Data
    import torch
    
    # Load data
    A = np.load('results/graphs/A_hybrid.npy')
    edge_index, edge_weight = adjacency_to_pytorch_geometric(A)
    
    # Create node features (e.g., 10-dimensional feature vector)
    x = torch.randn(108, 10)
    
    # Create PyTorch Geometric graph data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    # Use in GNN model
    # output = gnn_model(data)
    """)
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == '__main__':
    example_workflow()
