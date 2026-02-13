"""
Utility script for loading, analyzing, and validating PMU graphs.
"""

import numpy as np
import pandas as pd
import os


class GraphAnalyzer:
    """Analyze and validate PMU adjacency matrices."""
    
    def __init__(self, results_dir='results/graphs'):
        """
        Initialize analyzer and load graphs.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing saved matrices and metadata
        """
        self.results_dir = results_dir
        self.metadata = None
        self.A_geo = None
        self.A_corr = None
        self.A_hybrid = None
        
        self._load_all()
    
    def _load_all(self):
        """Load all matrices and metadata."""
        
        # Load metadata
        metadata_path = os.path.join(self.results_dir, 'sensor_metadata.csv')
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            print(f"Loaded metadata: {len(self.metadata)} sensors")
        
        # Load matrices
        matrices = {'A_geo': 'A_geo.npy', 'A_corr': 'A_corr.npy', 'A_hybrid': 'A_hybrid.npy'}
        for name, filename in matrices.items():
            path = os.path.join(self.results_dir, filename)
            if os.path.exists(path):
                setattr(self, name, np.load(path))
                print(f"Loaded {name}: {getattr(self, name).shape}")
    
    def analyze_matrix(self, A, name='Graph'):
        """
        Comprehensive analysis of an adjacency matrix.
        
        Parameters:
        -----------
        A : ndarray
            Adjacency matrix
        name : str
            Name for reporting
        
        Returns:
        --------
        stats : dict
        """
        n_nodes = A.shape[0]
        
        # Basic properties
        n_edges = np.count_nonzero(np.triu(A, k=1))
        density = 2 * n_edges / (n_nodes * (n_nodes - 1))
        
        # Degree statistics
        degrees = np.sum(A > 0, axis=1)
        
        # Weight statistics
        weights = A[A > 0]
        
        # Connectivity
        is_symmetric = np.allclose(A, A.T)
        has_self_loops = np.diag(A).sum() > 0
        
        stats = {
            'name': name,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': degrees.mean(),
            'min_degree': degrees.min(),
            'max_degree': degrees.max(),
            'std_degree': degrees.std(),
            'sparsity': 1 - (np.count_nonzero(A) / A.size),
            'avg_weight': weights.mean(),
            'min_weight': weights.min(),
            'max_weight': weights.max(),
            'std_weight': weights.std(),
            'symmetric': is_symmetric,
            'self_loops': has_self_loops,
            'values_in_01': np.all((A >= 0) & (A <= 1))
        }
        
        return stats
    
    def print_analysis(self):
        """Print analysis of all matrices."""
        
        print("\n" + "="*70)
        print("GRAPH ANALYSIS REPORT")
        print("="*70)
        
        matrices = [
            (self.A_geo, 'A_geo (Geographic)'),
            (self.A_corr, 'A_corr (Correlation)'),
            (self.A_hybrid, 'A_hybrid (Hybrid)')
        ]
        
        for A, name in matrices:
            if A is not None:
                stats = self.analyze_matrix(A, name)
                self._print_stats(stats)
    
    def _print_stats(self, stats):
        """Pretty print statistics."""
        print(f"\n{stats['name']}:")
        print(f"  Nodes: {stats['n_nodes']}")
        print(f"  Edges: {stats['n_edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Degree - Avg: {stats['avg_degree']:.2f}, Min: {stats['min_degree']}, Max: {stats['max_degree']}, Std: {stats['std_degree']:.2f}")
        print(f"  Sparsity: {stats['sparsity']:.4f}")
        print(f"  Weights - Avg: {stats['avg_weight']:.4f}, Range: [{stats['min_weight']:.4f}, {stats['max_weight']:.4f}]")
        print(f"  Symmetric: {stats['symmetric']}")
        print(f"  Self-loops: {stats['self_loops']}")
        print(f"  Values in [0,1]: {stats['values_in_01']}")
    
    def sanity_check(self):
        """Run sanity checks on matrices."""
        
        print("\n" + "="*70)
        print("SANITY CHECKS")
        print("="*70)
        
        checks_passed = 0
        checks_total = 0
        
        matrices = [
            (self.A_geo, 'A_geo'),
            (self.A_corr, 'A_corr'),
            (self.A_hybrid, 'A_hybrid')
        ]
        
        for A, name in matrices:
            if A is not None:
                print(f"\n{name}:")
                
                # Check 1: Shape
                checks_total += 1
                if A.shape[0] == A.shape[1]:
                    print(f"  ✓ Square matrix: {A.shape}")
                    checks_passed += 1
                else:
                    print(f"  ✗ Not square: {A.shape}")
                
                # Check 2: Symmetry
                checks_total += 1
                if np.allclose(A, A.T):
                    print(f"  ✓ Symmetric")
                    checks_passed += 1
                else:
                    print(f"  ✗ Not symmetric (max diff: {np.max(np.abs(A - A.T)):.6f})")
                
                # Check 3: No self-loops
                checks_total += 1
                if np.diag(A).sum() == 0:
                    print(f"  ✓ No self-loops")
                    checks_passed += 1
                else:
                    print(f"  ✗ Has self-loops (sum: {np.diag(A).sum():.6f})")
                
                # Check 4: Values in [0, 1]
                checks_total += 1
                if np.all((A >= 0) & (A <= 1)):
                    print(f"  ✓ All values in [0, 1]")
                    checks_passed += 1
                else:
                    print(f"  ✗ Values out of [0, 1]: min={A.min():.6f}, max={A.max():.6f}")
                
                # Check 5: Connectivity
                checks_total += 1
                degrees = np.sum(A > 0, axis=1)
                if np.all(degrees > 0):
                    print(f"  ✓ All nodes connected (min degree: {degrees.min()})")
                    checks_passed += 1
                else:
                    isolated = np.sum(degrees == 0)
                    print(f"  ⚠ {isolated} isolated nodes")
        
        print(f"\n{'='*70}")
        print(f"Checks passed: {checks_passed}/{checks_total}")
        print(f"{'='*70}")
    
    def degree_distribution(self):
        """Analyze degree distributions."""
        
        print("\n" + "="*70)
        print("DEGREE DISTRIBUTION")
        print("="*70)
        
        matrices = [
            (self.A_geo, 'A_geo'),
            (self.A_corr, 'A_corr'),
            (self.A_hybrid, 'A_hybrid')
        ]
        
        for A, name in matrices:
            if A is not None:
                degrees = np.sum(A > 0, axis=1)
                
                print(f"\n{name}:")
                print(f"  Mean degree: {degrees.mean():.2f}")
                print(f"  Median degree: {np.median(degrees):.2f}")
                print(f"  Degree range: [{degrees.min()}, {degrees.max()}]")
                print(f"  Highly connected (degree > mean): {np.sum(degrees > degrees.mean())} nodes")
    
    def export_gnn_format(self, matrix_name='A_hybrid'):
        """
        Export matrix in a format suitable for GNN training.
        
        Parameters:
        -----------
        matrix_name : str
            Which matrix to export ('A_geo', 'A_corr', 'A_hybrid')
        
        Returns:
        --------
        A : ndarray
            Adjacency matrix
        edge_index : ndarray, shape (2, n_edges)
            COO format edge list (compatible with PyTorch Geometric)
        edge_weight : ndarray, shape (n_edges,)
            Edge weights
        """
        A = getattr(self, matrix_name)
        
        if A is None:
            raise ValueError(f"{matrix_name} not loaded")
        
        # Get edges and weights
        i, j = np.nonzero(A)
        weights = A[i, j]
        
        # Undirected: only keep upper triangle
        mask = i < j
        i, j = i[mask], j[mask]
        weights = weights[mask]
        
        # Create bidirectional edge index
        edge_index = np.array([
            np.concatenate([i, j]),
            np.concatenate([j, i])
        ])
        
        edge_weight = np.concatenate([weights, weights])
        
        print(f"\nExported {matrix_name}:")
        print(f"  Nodes: {A.shape[0]}")
        print(f"  Edges (undirected): {len(weights)}")
        print(f"  Edges (directed): {edge_index.shape[1]}")
        
        return A, edge_index, edge_weight


def main():
    """Main execution."""
    
    analyzer = GraphAnalyzer('results/graphs')
    
    # Run analysis
    analyzer.print_analysis()
    analyzer.sanity_check()
    analyzer.degree_distribution()
    
    # Export for GNN
    print("\n" + "="*70)
    print("EXPORTING FOR GNN TRAINING")
    print("="*70)
    
    A, edge_index, edge_weight = analyzer.export_gnn_format('A_hybrid')
    
    # Optionally save in PyTorch format
    print("\nTo use in PyTorch/PyTorch Geometric:")
    print("```python")
    print("import torch")
    print("from torch_geometric.data import Data")
    print("")
    print("edge_index = torch.tensor(edge_index, dtype=torch.long)")
    print("edge_weight = torch.tensor(edge_weight, dtype=torch.float)")
    print("x = torch.randn(num_sensors, num_features)  # Node features")
    print("")
    print("data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)")
    print("```")


if __name__ == '__main__':
    main()
