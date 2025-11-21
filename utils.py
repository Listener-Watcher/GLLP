import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import ot 



from torch_geometric.utils import subgraph, add_self_loops as pyg_add_self_loops

def build_bag_subgraph_pyg(
    bag_nodes: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None,
    num_nodes: int = None, add_self_loops: bool = True, row_normalize: bool = True
):
    device = edge_index.device
    #print(f"device{device}")
    bag_nodes = bag_nodes.long().unique().to(device)
    # subgraph returns reindexed edge_index where bag_nodes map to [0..n_b-1]
    ei_b, ew_b = subgraph(bag_nodes, edge_index, edge_weight, num_nodes=num_nodes, relabel_nodes=True)
    n_b = bag_nodes.numel()
    if add_self_loops:
        if ew_b is None:
            ei_b, ew_b = pyg_add_self_loops(ei_b, num_nodes=n_b, fill_value=1.0)
        else:
            ei_b, ew_add = pyg_add_self_loops(ei_b, num_nodes=n_b, fill_value=1.0)
            ew_b = torch.cat([ew_b, ew_add])

    # Build sparse + row-normalize
    if ew_b is None:
        ew_b = torch.ones(ei_b.size(1), device=device)
    idx = ei_b
    A_b = torch.sparse_coo_tensor(idx, ew_b, (n_b, n_b), device=device).coalesce()
    if row_normalize:
        deg = torch.sparse.sum(A_b, dim=1).to_dense().clamp_min(1e-12)
        idc = A_b.indices()
        val = A_b.values() / deg[idc[0]]
        A_b = torch.sparse_coo_tensor(idc, val, (n_b, n_b), device=device).coalesce()
    return A_b
def symmetric_normalize(A_b: torch.Tensor):
    # A_b: sparse coo (un-normalized). Returns D^{-1/2} A D^{-1/2}.
    deg = torch.sparse.sum(A_b, dim=1).to_dense().clamp_min(1e-12)
    d_inv_sqrt = deg.pow(-0.5)
    idx = A_b.indices(); val = A_b.values()
    val = d_inv_sqrt[idx[0]] * val * d_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val, A_b.size(), device=A_b.device).coalesce()



def optimize_ot(PS, prop):
    N, K = PS.shape
    a = prop  # K
    b = np.ones(N) / N # N 
    a = np.array(a,dtype=float)
    b = np.array(b,dtype=float)
    M = -np.log(PS + 1e-10)
    #M = PS
    #M /= M.max() 

    T = ot.emd(b, a, M)  # KxN
    #T = ot.sinkhorn(a, b, M, 0.0001, verbose=True)
    #T = ot.sinkhorn(b, a, M, 0.00005, method = 'sinkhorn_stabilized')
    #print (T)
    argmaxes = np.argmax(T, 1)  # size N
    out = np.eye(K)[argmaxes]
    return out
def edge_index_to_sparse_adj(
    num_nodes: int,
    edge_index: torch.Tensor,           # shape [2, E], dtype long
    edge_weight: torch.Tensor = None,    # shape [E], optional
    symmetric: bool = True,              # make undirected
    add_self_loops: bool = False,
    dtype=torch.float32,
    device=None,
):
    """
    Build a coalesced torch.sparse_coo_tensor A \in R^{N x N} from edge_index.
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be [2, E]"
    if device is None:
        device = edge_index.device

    row, col = edge_index[0], edge_index[1]

    if symmetric:
        row = torch.cat([row, col], dim=0)
        col = torch.cat([col, row[:row.numel()//2]], dim=0)  # use original 'row' half

        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    if add_self_loops:
        loop_idx = torch.arange(num_nodes, device=device)
        row = torch.cat([row, loop_idx], dim=0)
        col = torch.cat([col, loop_idx], dim=0)
        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=device, dtype=edge_weight.dtype)])

    if edge_weight is None:
        edge_weight = torch.ones(row.numel(), device=device, dtype=dtype)

    indices = torch.stack([row, col], dim=0)  # [2, E’]
    A = torch.sparse_coo_tensor(indices, edge_weight, (num_nodes, num_nodes), dtype=dtype, device=device)
    A = A.coalesce()  # sums duplicate entries
    return A