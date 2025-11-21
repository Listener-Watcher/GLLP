import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor

try:
    from torch_geometric.nn import APPNP, SGConv
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError(
        "This script requires PyTorch Geometric. Install with: \n"
        "pip install torch torch_geometric torch_scatter torch_sparse --extra-index-url https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__.split("+")[0])').html"
    ) from e


class MLP(nn.Module):
    """
    Dense MLP: [Linear -> ReLU -> Dropout] x (L-1) + final Linear
    (Dropout applied after each hidden layer; no norm layers, per LINKX.)
    """
    def __init__(self, in_dim, hidden_dims, out_dim=None, dropout=0.5):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        self.dropout = dropout

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        self.out = None
        if out_dim is not None:
            self.out = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        for lin in self.layers:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.out is not None:
            x = self.out(x)
        return x


class SparseAdjEncoder(nn.Module):
    """
    Adjacency MLP with sparse first layer:
      h0 = A @ W0    (A sparse [N,N], W0 dense [N,H0] -> [N,H0])
      then standard dense layers: H0 -> H1 -> ... -> Hk
    """
    def __init__(self, num_nodes, hidden_dims, dropout=0.5):
        super().__init__()
        assert len(hidden_dims) >= 1, "Provide at least one hidden dim"

        self.num_nodes = num_nodes
        self.W0 = nn.Parameter(torch.empty(num_nodes, hidden_dims[0]))
        nn.init.xavier_uniform_(self.W0)

        # subsequent dense layers (if any)
        self.dense = None
        if len(hidden_dims) > 1:
            self.dense = MLP(hidden_dims[0], hidden_dims[1:], out_dim=None, dropout=dropout)
        self.dropout = dropout

    def forward(self, A_sparse):  # A_sparse: torch.sparse_coo_tensor [N,N]
        # First sparse × dense “linear” layer
        h = torch.sparse.mm(A_sparse, self.W0)  # [N, H0]
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Additional dense layers (optional)
        if self.dense is not None:
            h = self.dense(h)
        return h


class LINKX_Sparse(nn.Module):
    """
    Minimal LINKX-style model:
      - MLP_X encodes features X
      - SparseAdjEncoder encodes adjacency A
      - Concatenate [hX || hA] (optionally also raw X) and feed to output MLP
    """
    def __init__(
        self,
        num_nodes: int,
        in_feat_dim: int,
        x_hidden: list,          # e.g., [256] or [256,256]
        a_hidden: list,          # e.g., [256] or [256,256]
        out_hidden: list,        # e.g., [256]
        num_classes: int,
        dropout: float = 0.5,
        concat_raw_x: bool = False,  # skip raw X if you want (raw A is too large to concat)
    ):
        super().__init__()
        self.concat_raw_x = concat_raw_x

        self.mlp_x = MLP(in_feat_dim, x_hidden, out_dim=None, dropout=dropout)
        self.mlp_a = SparseAdjEncoder(num_nodes, a_hidden, dropout=dropout)

        in_out = x_hidden[-1] + a_hidden[-1]
        if concat_raw_x:
            in_out += in_feat_dim

        self.mlp_out = MLP(in_out, out_hidden, out_dim=num_classes, dropout=dropout)

    def forward(self, X, A_sparse):
        """
        X: [N, F] dense
        A_sparse: [N, N] sparse coalesced
        """
        hx = self.mlp_x(X)        # [N, Hx]
        ha = self.mlp_a(A_sparse) # [N, Ha]

        parts = [hx, ha]
        if self.concat_raw_x:
            parts.append(X)

        z = torch.cat(parts, dim=1)   # [N, Hx+Ha(+F)]
        logits = self.mlp_out(z)      # [N, C]
        return logits






# ---------------------
# Model definitions
# ---------------------
class APPNPNet(nn.Module):
    """APPNP model: MLP -> APPNP propagation.

    Args:
        in_channels: Node feature dimension
        hidden_channels: Width of the MLP hidden layer
        out_channels: #classes (logits)
        K: # propagation steps
        alpha: Teleport (1 - alpha) is the propagation strength
        dropout: Dropout applied in MLP and inside APPNP layer
        num_mlp_layers: 1 or 2 hidden layers (defaults to 1)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        num_mlp_layers: int = 1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        mlp_layers: List[nn.Module] = [nn.Dropout(dropout), nn.Linear(in_channels, hidden_channels), nn.ReLU()]
        if num_mlp_layers > 1:
            mlp_layers += [nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels), nn.ReLU()]
        mlp_layers += [nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels)]
        self.mlp = nn.Sequential(*mlp_layers)

        self.propagation = APPNP(K=K, alpha=alpha, dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        logits = self.mlp(x)
        logits = self.propagation(logits, edge_index, edge_weight)
        return logits


class SGCNet(nn.Module):
    """Simplifying Graph Convolution (SGC).

    This is essentially a linear classifier on top of K-step pre-smoothed features.

    Args:
        in_channels: Node feature dimension
        out_channels: #classes
        K: # propagation steps (>=1)
        cached: Cache the propagated features during training
        bias: Include bias in the classifier
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 2, cached: bool = True, bias: bool = True):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=K, cached=cached, bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        return self.conv(x, edge_index, edge_weight)
