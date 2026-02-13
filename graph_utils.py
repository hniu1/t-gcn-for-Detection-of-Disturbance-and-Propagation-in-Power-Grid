"""
Utility functions for pre-GNN graph construction from PMU sensor data.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance between two points in kilometers.
    
    Parameters:
    -----------
    lat1, lon1 : float or array
        Latitude and longitude of point 1 (in degrees)
    lat2, lon2 : float or array
        Latitude and longitude of point 2 (in degrees)
    
    Returns:
    --------
    distance : float or array
        Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth's radius in kilometers
    
    return c * r


def build_knn_adjacency(coordinates, k=5, normalize=True):
    """
    Build k-nearest neighbors adjacency matrix from coordinates.
    
    Parameters:
    -----------
    coordinates : ndarray, shape (n_sensors, 2)
        Array of [latitude, longitude] for each sensor
    k : int
        Number of nearest neighbors
    normalize : bool
        Whether to normalize weights to [0, 1]
    
    Returns:
    --------
    A : ndarray, shape (n_sensors, n_sensors)
        Sparse adjacency matrix with weights as inverse distance (symmetric)
    """
    n_sensors = coordinates.shape[0]
    
    # Compute pairwise distances
    distances = squareform(pdist(coordinates, metric=lambda u, v: 
                                 haversine_distance(u[0], u[1], v[0], v[1])))
    
    # Initialize adjacency matrix
    A = np.zeros((n_sensors, n_sensors))
    
    # Find k nearest neighbors for each sensor (excluding self)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coordinates)
    _, indices = nbrs.kneighbors(coordinates)
    
    # Build adjacency matrix with inverse distance weights
    for i in range(n_sensors):
        for j in indices[i]:
            if i != j:
                # Weight: inverse of distance (closer neighbors = higher weight)
                dist = distances[i, j]
                weight = 1.0 / (1.0 + dist)  # Avoid division by zero
                A[i, j] = weight
    
    # Ensure symmetry: A[i,j] = A[j,i] = max(A[i,j], A[j,i])
    A = np.maximum(A, A.T)
    
    # Normalize to [0, 1]
    if normalize and A.max() > 0:
        A = A / A.max()
    
    return A


def build_correlation_adjacency(X, threshold=0.6, remove_self_loops=True):
    """
    Build correlation-based adjacency matrix from time series data.
    
    Parameters:
    -----------
    X : ndarray, shape (n_timestamps, n_sensors)
        Multivariate time series data
    threshold : float
        Absolute correlation threshold (edges below this are removed)
    remove_self_loops : bool
        Whether to zero out diagonal
    
    Returns:
    --------
    A : ndarray, shape (n_sensors, n_sensors)
        Correlation-based adjacency matrix (symmetric)
    """
    # Compute Pearson correlation matrix
    corr_matrix = np.corrcoef(X.T)  # Shape: (n_sensors, n_sensors)
    
    # Zero out weak correlations
    A = np.abs(corr_matrix) > threshold
    A = A.astype(float)
    
    # Apply correlation weights (use absolute correlation as weights)
    A = A * np.abs(corr_matrix)
    
    # Ensure symmetry (correlation matrix should already be symmetric)
    A = np.maximum(A, A.T)
    
    # Remove self-loops
    if remove_self_loops:
        np.fill_diagonal(A, 0)
    
    return A


def build_hybrid_adjacency(A_geo, A_corr, alpha=0.5, normalize=True):
    """
    Combine geographic and correlation adjacency matrices.
    
    Parameters:
    -----------
    A_geo : ndarray, shape (n_sensors, n_sensors)
        Geographic adjacency matrix
    A_corr : ndarray, shape (n_sensors, n_sensors)
        Correlation adjacency matrix
    alpha : float
        Weight parameter: A_hybrid = alpha * A_geo + (1 - alpha) * A_corr
    normalize : bool
        Whether to normalize to [0, 1]
    
    Returns:
    --------
    A : ndarray, shape (n_sensors, n_sensors)
        Hybrid adjacency matrix (symmetric)
    """
    A = alpha * A_geo + (1 - alpha) * A_corr
    
    # Ensure symmetry
    A = np.maximum(A, A.T)
    
    if normalize and A.max() > 0:
        A = A / A.max()
    
    return A


def sparsify_adjacency(A, method='threshold', threshold=0.1, k=None):
    """
    Sparsify adjacency matrix by keeping only strong connections.
    
    Parameters:
    -----------
    A : ndarray, shape (n_sensors, n_sensors)
        Dense adjacency matrix
    method : str
        'threshold' - keep edges above threshold
        'knn' - keep top-k edges per node
    threshold : float
        For threshold method
    k : int
        For knn method
    
    Returns:
    --------
    A_sparse : ndarray, shape (n_sensors, n_sensors)
        Sparsified adjacency matrix (symmetric)
    """
    A_sparse = A.copy()
    
    if method == 'threshold':
        A_sparse[A_sparse < threshold] = 0
    elif method == 'knn':
        if k is None:
            k = 5
        for i in range(A.shape[0]):
            # Keep only top-k edges for each node (excluding self)
            top_indices = np.argsort(A_sparse[i])[-k-1:]
            top_indices = top_indices[top_indices != i]
            mask = np.ones(A.shape[0], dtype=bool)
            mask[top_indices] = False
            A_sparse[i, mask] = 0
    
    # Ensure symmetry
    A_sparse = np.maximum(A_sparse, A_sparse.T)
    
    return A_sparse


def compute_graph_statistics(A):
    """
    Compute basic statistics about the graph.
    
    Parameters:
    -----------
    A : ndarray, shape (n_sensors, n_sensors)
        Adjacency matrix
    
    Returns:
    --------
    stats : dict
        Dictionary with statistics
    """
    n_nodes = A.shape[0]
    n_edges = np.count_nonzero(np.triu(A, k=1))  # Count upper triangle only (symmetric)
    density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    
    # Degree distribution
    degrees = np.sum(A > 0, axis=1)
    
    stats = {
        'n_nodes': n_nodes,
        'n_edges': int(n_edges),
        'density': density,
        'avg_degree': degrees.mean(),
        'min_degree': degrees.min(),
        'max_degree': degrees.max(),
        'sparsity': 1 - (np.count_nonzero(A) / A.size)
    }
    
    return stats
