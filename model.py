"""
Multi-feature T-GCN model for spatiotemporal forecasting.

Architecture:
- Input: (B, Tin, N, F_dyn) + (B, N, F_static) for dynamic + static features
- Graph: (N, N) adjacency matrix with edge weights
- Process: Stack graphs over time into (B, Tin, N, N) with shared edges
- Output: (B, H, N, F_out)

Design:
- Keep model simple (no attention, no graph learning, as per user request)
- Use pre-computed geographic adjacency
- Concatenate static embeddings at each timestep
- Maintain core T-GCN architecture (graph conv → temporal conv)
"""

import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):
    """
    Single graph convolutional layer.
    
    Output: Y = AX W where:
    - A: normalized adjacency (N, N)
    - X: input features (batch, N, F_in)
    - W: learned weights (F_in, F_out)
    """
    
    def __init__(self, F_in: int, F_out: int):
        super().__init__()
        self.F_in = F_in
        self.F_out = F_out
        self.weight = nn.Parameter(torch.FloatTensor(F_in, F_out))
        self.bias = nn.Parameter(torch.FloatTensor(F_out))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A: Normalized adjacency (N, N)
            X: Input features (batch, N, F_in) or (N, F_in)
        
        Returns:
            Y: Output features same shape as X
        """
        # Y = A @ X @ W + b
        out = torch.matmul(X, self.weight) + self.bias  # (..., N, F_out)
        out = torch.matmul(A, out)  # (N, N) @ (..., N, F_out) -> (..., N, F_out)
        return out


class TemporalGRULayer(nn.Module):
    """
    Temporal modeling using GRU over time.
    
    Treats nodes as separate batch items to capture inter-node temporal dependencies.
    Returns final hidden state which summarizes the Tin timesteps of history.
    """
    
    def __init__(self, F_in: int, F_out: int, dropout: float = 0.1):
        super().__init__()
        self.F_in = F_in
        self.F_out = F_out
        self.gru = nn.GRU(
            input_size=F_in,
            hidden_size=F_out,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # No dropout for single layer
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, Tin, N, F_in) - batch of sequences
        
        Returns:
            h_final: (B, N, F_out) - final hidden state per node
        """
        B, T, N, F_in = X.shape
        assert F_in == self.F_in
        
        # Reshape to (B*N, T, F_in): treat each node as a separate sequence
        X_reshaped = X.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
        
        # GRU: (B*N, T, F_in) → output (B*N, T, F_out), h (1, B*N, F_out)
        _, h = self.gru(X_reshaped)
        
        # Extract final hidden state: (1, B*N, F_out) → (B*N, F_out)
        h_final = h[0]
        
        # Reshape back to (B, N, F_out)
        h_final = h_final.reshape(B, N, self.F_out)
        
        return h_final


class MultiFeatureTGCN(nn.Module):
    """
    T-GCN variant for multi-feature spatiotemporal forecasting.
    
    Architecture:
    - GCN applies at each timestep to capture spatial dependencies
    - GRU aggregates temporal information across Tin steps
    - Linear head projects to multi-step forecasts
    
    Input: 
    - x_dyn: (B, Tin, N, F_dyn) dynamic features
    - x_grid: (B, N, F_static) static features (grid embeddings)
    
    Output:
    - y_pred: (B, H, N, F_out) forecast
    """
    
    def __init__(self, 
                 N: int,                    # Number of sensors
                 F_dyn: int,                # Dynamic feature dim (4)
                 F_static: int,             # Static feature dim (4)
                 F_out: int,                # Output feature dim (3)
                 Tin: int = 100,            # Input history length
                 H: int = 10,               # Forecast horizon
                 hidden_dim: int = 64,
                 num_layers: int = 2):
        """
        Args:
            N: Number of nodes (sensors)
            F_dyn: Dynamic feature dimension (4: Δf, RoCoF, Δθ, ΔV)
            F_static: Static feature dimension (4: grid embedding)
            F_out: Output feature dimension (3: Δf, Δθ, ΔV)
            Tin: Input window size
            H: Forecast horizon
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
        """
        super().__init__()
        self.N = N
        self.F_dyn = F_dyn
        self.F_static = F_static
        self.F_out = F_out
        self.Tin = Tin
        self.H = H
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Concatenated feature dimension: F_dyn + F_static
        F_total = F_dyn + F_static
        
        # Graph convolutional layers (applied at each timestep)
        self.gc_layers = nn.ModuleList([
            GraphConvLayer(F_total if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Temporal GRU layer (aggregates Tin steps)
        self.gru_layer = TemporalGRULayer(hidden_dim, hidden_dim, dropout=0.1)
        
        # Output head: (B, N, hidden_dim) → (B, N, H*F_out)
        self.output_head = nn.Linear(hidden_dim, H * F_out)
    
    def forward(self, x_dyn: torch.Tensor, x_grid: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_dyn: (B, Tin, N, F_dyn) dynamic features
            x_grid: (B, N, F_static) static grid embeddings
            A: (N, N) adjacency matrix (normalized)
        
        Returns:
            y_pred: (B, H, N, F_out) forecast
        """
        B, Tin, N, F_dyn = x_dyn.shape
        assert N == self.N
        
        # Concatenate static features at each timestep: (B, Tin, N, F_dyn + F_static)
        x_grid_expanded = x_grid.unsqueeze(1).expand(B, Tin, N, self.F_static)
        x = torch.cat([x_dyn, x_grid_expanded], dim=-1)  # (B, Tin, N, F_dyn + F_static)
        
        # Graph convolutions: apply at each timestep, building new tensor with GCN outputs
        gcn_outputs = []
        for t in range(Tin):
            x_t = x[:, t, :, :]  # (B, N, F_dyn + F_static)
            
            # Apply GCN layers
            for gc_layer in self.gc_layers:
                x_t = gc_layer(A, x_t)  # (B, N, hidden_dim)
                x_t = torch.relu(x_t)
            
            gcn_outputs.append(x_t)  # (B, N, hidden_dim)
        
        # Stack into sequence: (B, Tin, N, hidden_dim)
        x_gcn = torch.stack(gcn_outputs, dim=1)
        
        # Apply temporal GRU: (B, Tin, N, hidden_dim) -> (B, N, hidden_dim)
        h_final = self.gru_layer(x_gcn)  # (B, N, hidden_dim)
        
        # Project to output features: (B, N, hidden_dim) → (B, N, H*F_out)
        y_flat = self.output_head(h_final)  # (B, N, H*F_out)
        
        # Reshape to (B, H, N, F_out)
        y_pred = y_flat.reshape(B, N, self.H, self.F_out).permute(0, 2, 1, 3)
        
        return y_pred


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix: A_norm = D^-0.5 * A * D^-0.5
    
    Args:
        A: (N, N) adjacency matrix
    
    Returns:
        A_norm: normalized adjacency
    """
    # Add self-loops
    A_self = A + torch.eye(A.shape[0], device=A.device)
    
    # Compute degree matrix
    D = torch.diag(torch.sum(A_self, dim=1))
    
    # Compute D^-0.5
    D_inv_sqrt = torch.inverse(torch.sqrt(D + 1e-6))
    
    # Normalize
    A_norm = D_inv_sqrt @ A_self @ D_inv_sqrt
    
    return A_norm
