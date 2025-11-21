#!/usr/bin/env python3
"""
LLP‑GNN training script (bag‑KL baseline) on Cora using PyTorch Geometric.

Consumes the outputs from the bag generator:
  - bags.jsonl (list of {"bag_id", "nodes"})
  - bag_proportions.npy (targets q_b) or bag_proportions_noisy.npy

Computes node logits for the *full graph* (full‑batch), aggregates per‑bag
mean class probabilities, and minimizes the bag‑level KL divergence.

Optional regularizers:
  - Laplacian smoothing on posteriors (homophily)
  - Entropy regularization (calibration / avoid overconfidence early)

Evaluation:
  - Node classification accuracy / macro‑F1 using ground truth (Cora labels)

Dependencies:
  - torch, torch_geometric, numpy, networkx (only for laplacian if needed)
  - (optional) torch_scatter for fast segment operations; otherwise a simple loop is used

Usage:
  python llp_gnn_train.py \
    --bags-dir out/ego2 \
    --target noisy   # choose: clean|noisy

"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple
from VAE import vat_consistency_loss_on_features,get_alpha
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
from torch_geometric.datasets import Planetoid,HeterophilousGraphDataset,Actor,Amazon,WikiCS
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected, get_laplacian, k_hop_subgraph
from geomloss import SamplesLoss
from utils import edge_index_to_sparse_adj,build_bag_subgraph_pyg,symmetric_normalize
from models import LINKX_Sparse,APPNPNet
from ot_train import ot_assign
try:
    from torch_scatter import scatter_add
    SCATTER_OK = True
except Exception:
    SCATTER_OK = False


# ===================== Config =====================
@dataclass
class TrainCfg:
    dataset_root: str
    name:str
    bags_dir: str
    out_dir: str
    target: str  # 'clean' or 'noisy'
    model: str   # 'gcn' or 'sage'
    hidden: int
    layers: int
    dropout: float
    lr: float
    wd: float
    max_epochs: int
    early_patience: int
    temp: float          # temperature on logits
    lam_smooth: float    # Laplacian smoothing weight
    lam_ent: float       # Entropy regularization weight
    lam_contrastive: float       # Entropy regularization weight
    loss:str
    add_bag:bool
    graphreg:float
    alpha:float
    beta:float
    k_aug:int
    seed:int
    T:int
    device: str


# ===================== Data loading =====================

def load_planetoid_cora(root: str):
    ds = Planetoid(root=root, name='Cora',split="geom-gcn")
    data = ds[0]
    data.train_mask = data.train_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    #print(ds.num_classes)
    return data, ds.num_classes
def load_planetoid_PubMed(root: str):
    ds = Planetoid(root=root, name='PubMed',split="geom-gcn")
    data = ds[0]
    # data.train_mask = data.train_mask[:,0]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    #print(ds.num_classes)
    return data, ds.num_classes
def load_planetoid_CiteSeer(root: str):
    ds = Planetoid(root=root, name='CiteSeer',split="geom-gcn")
    data = ds[0]
    # data.train_mask = data.train_mask[:,0]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    #print(ds.num_classes)
    return data, ds.num_classes
def load_computers(root: str):
    ds = Amazon(root=os.path.join("./data", "Amazon"), name='Computers')
    data = ds[0]
    # data.train_mask = data.train_mask[:,0]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    #print(ds.num_classes)
    return data, ds.num_classes
def load_photos(root: str):
    ds = Amazon(root=os.path.join("./data", "Amazon"), name='photo')
    data = ds[0]
    # data.train_mask = data.train_mask[:,0]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    #print(ds.num_classes)
    return data, ds.num_classes
def load_Roman(root: str):
    ds = HeterophilousGraphDataset(root=os.path.join("./data", "HeterophilousGraphDataset"), name="Roman-empire")
    data = ds[0]
    data.train_mask = data.train_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    return data, ds.num_classes
def load_Actor(root: str):
    ds = Actor(root=os.path.join("./data", "Actor"))
    data = ds[0]
    data.train_mask = data.train_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    return data, ds.num_classes
def load_amazon(root: str):
    ds = HeterophilousGraphDataset(root=os.path.join("./data", "HeterophilousGraphDataset"), name="Amazon-ratings")
    data = ds[0]
    data.train_mask = data.train_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    return data, ds.num_classes
def load_Wiki(root: str):
    ds = WikiCS(root=os.path.join("./data","WikiCS"),is_undirected=True)
    data = ds[0]
    # data.train_mask = data.train_mask[:,0]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    # data.edge_index = to_undirected(data.edge_index)
    return data, ds.num_classes
def load_Covid(root: str,day:int):
    data = torch.load(f'./data/graph_covid_1000_{day}.pt')
    num_class = len(torch.unique(data.y))
    # data.train_mask = data.train_mask[:,0]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    # data.edge_index = to_undirected(data.edge_index)
    data.edge_index = to_undirected(data.edge_index)
    # data.train_mask = data.train_mask[:,200]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    return data, num_class
def load_Covidbinary(root: str,day:int):
    data = torch.load(f'./data/graph_covidbinary_1000_{day}.pt')
    num_class = len(torch.unique(data.y))
    # data.train_mask = data.train_mask[:,0]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    # data.edge_index = to_undirected(data.edge_index)
    data.edge_index = to_undirected(data.edge_index)
    # data.train_mask = data.train_mask[:,200]
    # data.test_mask = data.test_mask[:,0]
    # data.val_mask = data.val_mask[:,0]
    return data, num_class
def load_bags(bags_dir: str, target: str = 'clean') -> Tuple[List[torch.Tensor], torch.Tensor]:
    bags = []
    path_bags = os.path.join(bags_dir, 'bags.jsonl')
    with open(path_bags, 'r') as f:
        for line in f:
            rec = json.loads(line)
            bags.append(torch.tensor(rec['nodes'], dtype=torch.long))
    if target == 'clean':
        Q = np.load(os.path.join(bags_dir, 'bag_proportions.npy'))
    else:
        Q = np.load(os.path.join(bags_dir, 'bag_proportions_noisy.npy'))
    Q = torch.tensor(Q, dtype=torch.float32)
    return bags, Q


# ===================== Model =====================


class GCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        if layers == 1:
            self.layers.append(GCNConv(in_dim, out_dim))
        else:
            self.layers.append(GCNConv(in_dim, hid))
            for _ in range(layers-2):
                self.layers.append(GCNConv(hid, hid))
            self.layers.append(GCNConv(hid, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers)-1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGE(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        if layers == 1:
            self.layers.append(SAGEConv(in_dim, out_dim))
        else:
            self.layers.append(SAGEConv(in_dim, hid))
            for _ in range(layers-2):
                self.layers.append(SAGEConv(hid, hid))
            self.layers.append(SAGEConv(hid, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers)-1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class RGCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, layers: int, linear_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        if layers == 1:
            self.layers.append(GCNConv(in_dim, out_dim))
        else:
            self.layers.append(GCNConv(in_dim, hid))
            for _ in range(layers-2):
                self.layers.append(GCNConv(hid, hid))
            self.layers.append(GCNConv(hid, out_dim))
        self.dropout = dropout
        self.linear_maps = nn.ModuleList()
        for i in range(linear_layers):
            self.linear_maps.append(nn.Linear(hid, hid))
        self.norm = nn.LayerNorm(hid)
    def forward(self, x, edge_index):
        x = F.dropout(x,p=self.dropout,training=self.training)
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers)-1:
                x = F.relu(x)
                X_linear = x
                for i, linear in enumerate(self.linear_maps):
                    if i < len(self.linear_maps)-1:
                        X_linear = F.relu(linear(X_linear))
                    else:
                        X_linear = (linear(X_linear))
                        X_linear = self.norm(X_linear)
                        X_linear = F.relu(X_linear)
                x = X_linear+x
        return x
# ===================== Loss pieces =====================




def kl_bag_loss(p_bags: torch.Tensor, q_bags: torch.Tensor) -> torch.Tensor:
    # p_bags, q_bags: [m, C]
    eps = 1e-10
    p = (p_bags + eps)
    q = (q_bags + eps)
    return torch.sum(q * (torch.log(q) - torch.log(p))) / p.shape[0]


def entropy_reg(p_node: torch.Tensor) -> torch.Tensor:
    # mean entropy over nodes
    eps = 1e-10
    p = torch.clamp(p_node, eps, 1.0)
    H = -torch.sum(p * torch.log(p), dim=-1).mean()
    return H
def node_contrastive_loss(h1, h2, tau=0.5):
    # h1, h2: [N, hidden]
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    sim = torch.mm(h1, h2.t()) / tau  # [N,N]
    labels = torch.arange(h1.size(0), device=h1.device)
    loss = F.cross_entropy(sim, labels)
    return loss

def laplacian_smoothing(p_node: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    # Sum over edges of squared differences of posteriors
    row, col = edge_index
    diff = p_node[row] - p_node[col]
    return (diff.pow(2).sum(dim=-1)).mean()

def augment_graph(x, edge_index, feat_drop=0.2, edge_drop=0.2):
    # feature dropout
    x_aug = F.dropout(x, p=feat_drop, training=True)
    # edge dropout (randomly drop edges)
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > edge_drop
    edge_index_aug = edge_index[:, mask]
    return x_aug, edge_index_aug




def bag_guided_dropout_fast(data, bags, Q, feat_drop=0.2, edge_drop=0.2):
    """
    Vectorized bag-guided dropout for features and edges.
    data: PyG Data object
    bags: list of lists of node indices
    Q: [num_bags, num_classes] bag label distributions
    feat_drop, edge_drop: base drop rates
    """
    device = data.x.device
    num_nodes = data.num_nodes

    # --- Node dropout ---
    node_keep_prob = torch.zeros(num_nodes, device=device)

    for bag_idx, bag_nodes in enumerate(bags):
        prob = Q[bag_idx].max()  # confidence from bag distribution
        node_keep_prob[bag_nodes] = torch.maximum(
            node_keep_prob[bag_nodes],
            torch.tensor(1 - feat_drop * (1 - prob), device=device)
        )

    rand_vals = torch.rand(num_nodes, device=device)
    node_mask = rand_vals < node_keep_prob  # True = keep

    data_aug = data.clone()
    data_aug.x[~node_mask] = 0  # drop features

    # --- Edge dropout ---
    src, dst = data.edge_index
    edge_keep_prob = torch.min(node_keep_prob[src], node_keep_prob[dst])
    rand_vals = torch.rand(edge_keep_prob.size(), device=device)
    edge_mask = rand_vals < edge_keep_prob
    data_aug.edge_index = data.edge_index[:, edge_mask]

    return data_aug.x,data_aug.edge_index




class LossPlateauEarlyStopping:
    def __init__(self, patience=50, delta=1e-2):
        """
        Early stopping to stop training when loss doesn't improve after a given patience.

        Args:
            patience (int): Number of epochs to wait after last improvement.
            delta (float): Minimum change in loss to qualify as improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.stopped_epoch = 0

    def step(self, current_loss, epoch):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch


def mycriterion(outputs, soft_targets_a):
    probs = F.softmax(outputs, dim=1)
    L_c_a = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets_a, dim=1))

    L_c = L_c_a

    # RCE
    probs = torch.clamp(probs, min=1e-7, max=1.0)
    soft_targets_a = torch.clamp(soft_targets_a, min=1e-4, max=1.0)
    rce_a = -torch.mean((torch.sum(probs * torch.log(soft_targets_a), dim=1)))

    rce = rce_a

    # Loss
    loss_new = L_c + rce

    return probs, loss_new






def add_graph_cost(
    p: torch.Tensor,
    adj: Optional[torch.Tensor],
    row_normalized: bool = True,
) -> torch.Tensor:
    """
    Adds a graph-aware disagreement term:
      cost[i,k] += lambda_g * (1 - avg_{j in N(i)} p[j,k])

    adj can be dense or sparse_coo. If not row-normalized, we row-normalize it.
    """
    if adj is None:
        return 0.0

    if not adj.is_sparse:
        adj = adj.to_sparse_coo()

    if row_normalized:
        row_norm = adj
    else:
        idx = adj.indices()
        vals = adj.values()
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-12)
        vals = vals / deg.index_select(0, idx[0])
        row_norm = torch.sparse_coo_tensor(idx, vals, adj.size(), device=adj.device)

    neighbor_p = torch.sparse.mm(row_norm, p)  # [n,K]
    penalty = neighbor_p
    return penalty
def graph_loss(logits,bags,Q,graph_b,graphreg):
    total_loss = 0.0
    for i,b in enumerate(bags):
        # logits_b: [n_b, K], q_b: [K], graph_adj_b: optional local adjacency
        logits_b = logits[b]
        q_b = Q[i]
        n_b = logits_b.shape[0]
        graph_adj_b = graph_b[i]
        temperature = 2
        cost = add_graph_cost(F.softmax(logits_b / temperature, dim=-1), graph_adj_b,row_normalized=True)
        loss = graphreg*kl_bag_loss(F.softmax(logits_b / temperature, dim=-1),cost)
        total_loss+=loss
    return total_loss

def CI_loss(p_bags_aug,proportions,sigmas,alpha):
    # p_bags, q_bags: [m, C]
    c_low = proportions-alpha*sigmas
    c_high = proportions+alpha*sigmas
    indicator = (p_bags_aug < c_low) | (p_bags_aug > c_high)
    eps = 1e-10
    p = (p_bags_aug + eps)
    q = (proportions + eps)
    return torch.sum(-q*indicator*torch.log(p)) / p.shape[0]
def ce_soft_loss(logits,bags,Q,graph_b,graphreg,T):
    total_loss = 0.0
    for i,b in enumerate(bags):
        # logits_b: [n_b, K], q_b: [K], graph_adj_b: optional local adjacency
        logits_b = logits[b]
        q_b = Q[i]
        n_b = logits_b.shape[0]
        graph_adj_b = graph_b[i]
        temperature = T
        res = ot_assign(
            logits_b, q_b,
            cost_type="neg_logp", temperature=temperature,
            epsilon=0.5, sinkhorn_iters=50,
            graph_adj=graph_adj_b, lambda_graph=graphreg,adj_row_normalized=True,
            return_hard=False
        )
        A    = res["A"]       # soft pseudo labels
        # # y_hat  = res.get("y_hat")  # hard labels (if requested)
        # # counts = res.get("counts") # integer quotas used
        # # p_b    = res["p"]       # model probs (may be useful for diagnostics)
        # #print(f"p_b:{p_b}")
        L_ce_soft = -(A.detach() * torch.log_softmax(logits_b, dim=-1)).sum() / n_b
        total_loss += L_ce_soft

        # Mix with bag prior to avoid degeneracy
        # beta = 0.2                        # try 0.1–0.3
        # A = (1 - beta) * A + beta * q_b[None, :]
        # A = A / A.sum(dim=1, keepdim=True)

        # logp = F.log_softmax(logits_b, dim=-1)
        # per_node_ce = -(A * logp).sum(dim=1)   # [n_b]

        # # (Optional) very gentle confidence weighting (squashed + capped)
        # K = A.size(1)
        # conf = A.max(dim=1).values
        # w = ((conf - 1.0 / K) / (1.0 - 1.0 / K)).clamp(min=0)  # [0,1]
        # w = (w ** 2).clamp(0.2, 0.8).detach()                  # squash + cap

        # L_ce = (w * per_node_ce).sum() / (w.sum().clamp_min(1e-8))

        # total_loss += L_ce       # 0.05–0.2 works

        #total_loss += L_ce_soft
    return total_loss

# ===================== Bag aggregation =====================

def aggregate_bag_probs_from_logits(logits: torch.Tensor, bags: List[torch.Tensor], temp: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # logits: [n, C]
    p = F.softmax(logits / temp, dim=-1)
    m = len(bags)
    C = p.shape[1]
    device = p.device
    p_bags = torch.zeros((m, C), device=device)
    # Aggregate by simple mean over nodes in each bag
    # (fast path with scatter if available)
    if SCATTER_OK:
        # Build a concatenated index and bag ids
        all_idx = torch.cat(bags)
        bag_ids = torch.cat([torch.full((len(b),), i, dtype=torch.long, device=device) for i, b in enumerate(bags)]).to(device)
        p_sel = p[all_idx.to(device)]
        sums = scatter_add(p_sel, bag_ids, dim=0, dim_size=m)
        counts = scatter_add(torch.ones((p_sel.size(0), 1), device=device), bag_ids, dim=0, dim_size=m)
        p_bags = sums / counts.clamp_min(1.0)
    else:
        for i, b in enumerate(bags):

            p_bags[i] = p[b.to(device)].mean(dim=0)

    return p, p_bags


def build_adj_list(edge_index: torch.LongTensor,
                   num_nodes: Optional[int] = None,
                   undirected: bool = True) -> List[List[int]]:
    """
    edge_index: shape [2, E], dtype long
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        adj[u].append(v)
        if undirected:
            adj[v].append(u)
    return adj

def augment_bags_with_ego(
    edge_index: torch.Tensor,
    original_bags: list[list[int]],
    Q: torch.Tensor,
    Q_global: torch.Tensor,
    *,
    bag_size: int = 50,
    k_hop: int = 2,
    k_aug_per_bag: int = 2,
    device: str = "cpu",
) -> tuple[list[torch.Tensor], list[dict]]:
    """
    For each original bag:
      - pick one seed node from the bag
      - take its k-hop ego-net over the full graph
      - build a new augmented bag of exactly bag_size nodes
      - track metadata (m_in, m_ego, seed, source bag idx)

    Returns:
      aug_bags : list of [bag_size] LongTensors
      aug_meta : list of dicts with keys:
          - b_idx : source original bag index
          - seed  : chosen seed node id
          - m_in  : # nodes in aug bag from original bag
          - m_ego : # nodes in aug bag from ego pool (outside bag)
    """
    aug_bags: list[torch.Tensor] = []
    aug_meta: list[dict] = []
    proportions: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []
    for b_idx, bag in enumerate(original_bags):
        if len(bag) == 0:
            continue
        bag_tensor = torch.tensor(bag, dtype=torch.long, device=device)
        Q_bag = Q[b_idx]
        for _ in range(k_aug_per_bag):
            # pick one seed from the original bag
            seed = bag_tensor[torch.randint(0, bag_tensor.numel(), (1,), device=device)]
            # print("seed",seed)
            # ego neighborhood
            subset, _, _, _ = k_hop_subgraph(
                seed, k_hop, edge_index, relabel_nodes=False
            )
            ego_nodes = subset.unique()
            # always include the seed in the augmented bag
            # union of ego_nodes and original bag (for safety)
            pool = ego_nodes

            # sample to exactly bag_size
            if pool.numel() >= bag_size:
                perm = torch.randperm(pool.numel(), device=device)[:bag_size-1]
                aug_nodes = pool[perm]
            else:
                # sample with replacement if pool is too small
                idx = torch.randint(0, pool.numel(), (bag_size-1,), device=device)
                aug_nodes = pool[idx]
            aug_nodes = torch.cat([seed.view(1), aug_nodes], dim=0)  # [bag_size]
            # print(aug_nodes)
            # compute m_in and m_ego
            bag_set = set(bag.tolist())
            aug_set = set(aug_nodes.tolist())
            # print(bag_set)
            # print(aug_set)
            in_nodes = aug_set.intersection(bag_set)
            m_in = len(in_nodes)
            m_ego = bag_size - m_in
            gamma = m_in/(m_in+m_ego)
            # print(m_ego)
            # print(gamma)
            # print(Q_bag)
            Q_bag = Q_bag+1e-8
            sigma = gamma*torch.sqrt(Q_bag*(1-Q_bag)/m_in)+(1-gamma)*torch.sqrt(Q_global*(1-Q_global)/m_ego)
            proportion = gamma*Q_bag+(1-gamma)*Q_global
            aug_bags.append(aug_nodes)
            aug_meta.append({
                "b_idx": b_idx,
                "seed": int(seed.item()),
                "m_in": m_in,
                "m_ego": m_ego,
                "sigma": sigma,
                "proportion": proportion
            })
            proportions.append(proportion)
            sigmas.append(sigma)

    return aug_bags, aug_meta, torch.stack(proportions,dim=0).to(device),torch.stack(sigmas,dim=0).to(device)

def augment_bags(
    original_bags: list[list[int]],
    Q: torch.Tensor,
    *,
    bag_size: int = 50,
    k_hop: int = 2,
    k_aug: int = 2,
    device: str = "cpu",
) -> tuple[list[torch.Tensor], list[dict]]:
    """
    For total k_aug bags:
        - Sample a probability gamma from a uniform distribution
        - Pick Two Random original bags
        - Pick int(gamma * bag_size) seed nodes from one bag and
          bag_size - int(gamma * bag_size) seed nodes from another bag
        - Build a new augmented bag with the picked nodes from two bags
        - Track metadata (m_in, m_ego, seed, source bag idx)

    Args:
        original_bags (list[list[int]]): List of original bags (each a list of node IDs)
        Q (torch.Tensor): Transition matrix or similarity matrix [N x N]
        bag_size (int): Desired size of each augmented bag
        k_hop (int): Number of hops (not used here directly but can be integrated later)
        k_aug (int): Number of augmented bags to generate
        device (str): Device for tensors

    Returns:
        aug_bags (list of torch.LongTensor): Augmented bags of shape [bag_size]
        aug_meta (list of dict): Metadata with keys:
            - b_idx: source bag index
            - seed: seed node IDs used for augmentation
            - m_in: number of nodes from bag A
            - m_ego: number of nodes from bag B
    """
    num_bags = len(original_bags)
    aug_bags: list[torch.Tensor] = []
    aug_meta: list[dict] = []
    proportions: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []
    for _ in range(k_aug):
        # Sample gamma ∈ [0, 1]
        gamma = random.uniform(0, 1)

        # Randomly choose 2 different bags
        idx_a, idx_b = random.sample(range(num_bags), 2)
        bag_a = list(original_bags[idx_a])
        bag_b = list(original_bags[idx_b])
        Q_a = Q[idx_a]
        Q_b = Q[idx_b]
        # Determine number of nodes to take from each
        n_a = int(gamma * bag_size)
        n_b = bag_size - n_a
        gamma2 = n_a/(n_a+n_b)
        # print(m_ego)
        # print(gamma)
        # print(Q_bag)
        Q_a = Q_a+1e-10
        Q_b = Q_b+1e-10
        sigma = gamma2*torch.sqrt(Q_a*(1-Q_a)/n_a)+(1-gamma2)*torch.sqrt(Q_b*(1-Q_b)/n_b)
        proportion = gamma2*Q_a+(1-gamma2)*Q_b
        # Sample seed nodes (with replacement if bag is too small)
        seed_a = random.choices(bag_a, k=n_a) if len(bag_a) < n_a else random.sample(bag_a, n_a)
        seed_b = random.choices(bag_b, k=n_b) if len(bag_b) < n_b else random.sample(bag_b, n_b)

        # Combine to form augmented bag
        aug_nodes = seed_a + seed_b
        aug_tensor = torch.tensor(aug_nodes, dtype=torch.long, device=device)

        # Create metadata
        aug_meta.append({
            "b_idx": (idx_a, idx_b),
            "seed": (seed_a, seed_b),
            "m_in": n_a,
            "m_ego": n_b,
        })
        proportions.append(proportion)
        sigmas.append(sigma)
        aug_bags.append(aug_tensor)

    return aug_bags, aug_meta,torch.stack(proportions,dim=0).to(device),torch.stack(sigmas,dim=0).to(device)



# ===================== Training loop =====================

def train(cfg: TrainCfg):
    device = torch.device(cfg.device)
    if cfg.name == 'Cora':
        data, num_classes = load_planetoid_cora(cfg.dataset_root)
    elif cfg.name == 'PubMed':
        data, num_classes = load_planetoid_PubMed(cfg.dataset_root)
    elif cfg.name == 'Amazon':
        data,num_classes = load_amazon(cfg.dataset_root)
    elif cfg.name == 'Roman':
        data,num_classes = load_Roman(cfg.dataset_root)
    elif cfg.name == 'Actor':
        data,num_classes = load_Actor(cfg.dataset_root)
    elif cfg.name == 'CiteSeer':
        data,num_classes = load_planetoid_CiteSeer(cfg.dataset_root)
    elif cfg.name == 'Computers':
        data,num_classes = load_computers(cfg.dataset_root)
    elif cfg.name == 'Photos':
        data,num_classes = load_photos(cfg.dataset_root)
    elif cfg.name == 'WikiCS':
        data,num_classes = load_Wiki(cfg.dataset_root)
    elif cfg.name == 'Covid':
        data,num_classes = load_Covid(cfg.dataset_root,500)
    elif cfg.name == 'Covid700':
        data,num_classes = load_Covid(cfg.dataset_root,700)
    elif cfg.name == 'Covidbinary':
        data,num_classes = load_Covidbinary(cfg.dataset_root,500)
    else:
        return
    #data, num_classes = load_planetoid_PubMed(cfg.dataset_root)

    data = data.to(device)
    y = data.y.to(device)
    bag_label = []
    bags, Q = load_bags(cfg.bags_dir, cfg.target)
    num_bags = len(bags)
    print("number of bags:",num_bags)
    train_mask = torch.zeros(y.shape, dtype=torch.bool)
    graph_b = []
    train_mask2 = torch.zeros(y.shape, dtype=torch.bool)
    #print(bags)
    for i,b in enumerate(bags):
        #print(f"bags:{i}:{b}")
        train_mask[b]=True # obtain first 5 labels for cheating
        train_mask2[b[0:5]]=True # obtain first 5 labels for cheating
        graph_b.append(build_bag_subgraph_pyg(b, data.edge_index,row_normalize=True).to(device))
    # print(f"Q.shape{Q.shape}")
    train_mask2 = train_mask2.to(device)
    train_mask = train_mask.to(device)
    test_mask = ~train_mask   # logical NOT
    Q = Q.to(device)
    in_dim = data.x.size(1)
    Q_global = torch.mean(Q,dim=0)
    # print("estimate:",Q_global)
    # Count distribution
    counts = torch.bincount(y, minlength=y.max().item()+1)
    dist = counts.float() / counts.sum()
    # print("true dis",dist)  # tensor([0.2857, 0.2857, 0.4286])
    if cfg.add_bag:
        #aug_bags,meta,proportions,sigmas = augment_bags_with_ego(data.edge_index, bags,Q,Q_global,bag_size=50, k_hop=3, k_aug_per_bag=k_aug_per_bag,device=data.edge_index.device)
        aug_bags,meta,proportions,sigmas = augment_bags(bags,Q,bag_size=50,k_hop=3,k_aug=cfg.k_aug,device=data.edge_index.device)
    if cfg.model == 'gcn':
        model = GCN(in_dim, cfg.hidden, num_classes, cfg.layers, cfg.dropout)
    elif cfg.model == 'sage':
        model = SAGE(in_dim, cfg.hidden, num_classes, cfg.layers, cfg.dropout)
    elif cfg.model == 'rgcn':
        model = RGCN(in_dim, cfg.hidden, num_classes, cfg.layers, 2, cfg.dropout)
    elif cfg.model == 'appnp':
        model = APPNPNet(in_dim,cfg.hidden,num_classes,10, 0.1,cfg.dropout,cfg.layers)
    elif cfg.model =='linkx':
        model = LINKX_Sparse(
            num_nodes=data.x.shape[0],
            in_feat_dim=data.x.shape[1],
            x_hidden=[cfg.hidden],
            a_hidden=[cfg.hidden],
            out_hidden=[cfg.hidden],
            num_classes=num_classes,
            dropout=0.5,
            concat_raw_x=False,   # set True if you want to also feed raw X to the head
        )
        A = edge_index_to_sparse_adj(
            num_nodes=data.x.shape[0],
            edge_index=data.edge_index,
            edge_weight=None,
            symmetric=True,
            add_self_loops=False,
            dtype=torch.float32,
            device=device,
        )
    model = model.to(device)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    # opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)  # we'll set WD per step
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)  # example
    # wd_max = cfg.wd                                # your peak WD
    # base_lrs = [g.get('initial_lr', cfg.lr) for g in opt.param_groups]  # ref LR per group

    # def update_wd_proportional_to_lr(optimizer, last_lrs, base_lrs, wd_max):
    #     # Set WD := wd_max * (lr_t / lr_base) for each group
    #     for pg, lr, base in zip(optimizer.param_groups, last_lrs, base_lrs):
    #         scale = float(lr) / float(base)
    #         pg['weight_decay'] = wd_max * scale
    best_val = -1e9
    best_state = None
    patience = cfg.early_patience
    patience_count = 0
    # early_stopper = LossPlateauEarlyStopping(patience=patience, delta=1e-3)
    best_loss = 1e9
    Loss_list = []
    acc_list = []

    cecriterion = nn.CrossEntropyLoss()
    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        opt.zero_grad()
        # x1, edge1 = augment_graph(data.x, data.edge_index)
        # x2, edge2 = augment_graph(data.x, data.edge_index)
        #x1,edge1 = bag_guided_dropout_fast(data, bags, Q, feat_drop=0.2, edge_drop=0.2)
        #x2,edge2 = bag_guided_dropout_fast(data, bags, Q, feat_drop=0.2, edge_drop=0.2)
        # h1 = model(x1, edge1)  # [n, C]
        # h2 = model(x2, edge2)  # [n, C]
        # loss_contrastive = node_contrastive_loss(h1, h2)
        if cfg.model == 'linkx':
            logits = model(data.x, A)  # [N, C]
        else:
            logits = model(data.x, data.edge_index)  # [n, C]
        p_node, p_bags = aggregate_bag_probs_from_logits(logits, bags, cfg.temp)
        if cfg.add_bag:
            p_node2,p_bags_aug = aggregate_bag_probs_from_logits(logits, aug_bags, cfg.temp)
            loss_kl_reg = CI_loss(p_bags_aug,proportions,sigmas,cfg.alpha)
        # split = 15

        # or
        # L_ce_hard = F.cross_entropy(logits_b, y_hat)
        # + your graph Dirichlet/consistency/etc
        # num_classes = p_bags.shape[1]
        # class_positions = torch.arange(num_classes, device=p_bags.device).float().reshape(1, num_classes, 1)  # [1,7,1]
        # class_positions = class_positions.repeat(p_bags.shape[0], 1, 1)  # [B,7,1]
        # alpha = p_bags.unsqueeze(-1)
        # beta  = Q.unsqueeze(-1)
        # alpha = alpha.clamp(min=1e-6)
        # beta  = beta.clamp(min=1e-6)
        # alpha /= alpha.sum(dim=1, keepdim=True)
        # beta  /= beta.sum(dim=1, keepdim=True)
        # class_positions = class_positions.contiguous()
        # alpha = alpha.contiguous()
        # beta = beta.contiguous()
        # Define the Sinkhorn loss
        # loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        # loss_ot = loss_fn(class_positions, class_positions, alpha, beta)
        # loss_ot_train = loss_ot[0:split].mean()
        # loss_ot_val = loss_ot[split:split+1].mean()
        # loss_vl = loss_kl+loss_ot
    
        # loss_vl = kl_bag_loss(p_bags, Q)
        # loss = loss_ot_train+cfg.lam_contrastive*loss_contrastive
        # loss = (1-cfg.lam_contrastive)*loss_kl_train+cfg.lam_contrastive*loss_contrastive
        if cfg.loss=="klce":
            loss_kl_train = kl_bag_loss(p_bags, Q)
            loss_ce = cecriterion(logits[train_mask], y[train_mask])
            loss = loss_kl_train+loss_ce
        elif cfg.loss=="kl":
            loss_kl_train = kl_bag_loss(p_bags, Q)
            loss = loss_kl_train
        elif cfg.loss=="ce":
            # Get indices where train_mask is True
            idx = train_mask2.nonzero(as_tuple=True)[0]   # tensor([0, 2, 3])

            # Permute those indices
            perm = idx[torch.randperm(idx.size(0))]

            # Apply permutation (in-place update)
            y[idx] = y[perm]
            loss_ce = cecriterion(logits[train_mask2], y[train_mask2])
            loss = loss_ce
        elif cfg.loss=="klreg":
            loss_kl_train = kl_bag_loss(p_bags, Q)
            loss = loss_kl_train+cfg.beta*loss_kl_reg
        elif cfg.loss =='ot':
            L_ce_soft = ce_soft_loss(logits,bags,Q,graph_b,cfg.graphreg,cfg.T)
            loss = L_ce_soft
        elif cfg.loss =='otkl':
            L_ce_soft = ce_soft_loss(logits,bags,Q,graph_b,cfg.graphreg,cfg.T)
            loss_kl_train = kl_bag_loss(p_bags, Q)
            loss = cfg.beta*L_ce_soft+loss_kl_train
        elif cfg.loss =='otklreg':
            L_ce_soft = ce_soft_loss(logits,bags,Q,graph_b,cfg.graphreg,cfg.T)
            loss_kl_train = kl_bag_loss(p_bags, Q)
            loss = cfg.beta*L_ce_soft+loss_kl_train+0.05*loss_kl_reg
        elif cfg.loss == 'vae':
            alpha = get_alpha(epoch, max_alpha=1.0, rampup_length=20, mode="sigmoid")
            vat_loss = vat_consistency_loss_on_features(
                lambda feats: model(feats, data.edge_index),  # forward_fn
                data.x,                                      # node features to perturb
                eps=2.0, xi=1e-6, ip=1,
                clip_range=None,                             # or (0,1) if features are normalized
                mask=train_mask.float(),                # optional: apply only on train nodes
                reduction="mean",
            )

            loss_kl_train = kl_bag_loss(p_bags, Q)
            loss = loss_kl_train  + alpha * vat_loss
        elif cfg.loss == 'klgraph':
            loss_kl_train = kl_bag_loss(p_bags, Q)
            graphloss = graph_loss(logits,bags,Q,graph_b,cfg.graphreg)
            # print("graphloss",graphloss)
            loss = loss_kl_train+graphloss

        # loss = loss_ot
        loss_vl = loss
        if cfg.lam_smooth > 0:
            loss = loss + cfg.lam_smooth * laplacian_smoothing(p_node, data.edge_index)
        if cfg.lam_ent != 0:
            # negative sign: encourage entropy if lam_ent>0
            #print(entropy_reg(p_node))
            loss = loss - cfg.lam_ent * entropy_reg(p_node)
        loss.backward()
        # # sync WD with current LR (per param group)
        # last_lrs = sched.get_last_lr()
        # update_wd_proportional_to_lr(opt, last_lrs, base_lrs, wd_max)
        opt.step()
        Loss_list.append(loss.item())
        with torch.no_grad():
            if cfg.model == 'linkx':
                logits = model(data.x, A)  # [N, C]
            else:
                logits = model(data.x, data.edge_index)  # [n, C]
            pred = logits.argmax(dim=-1)
            y = data.y
            #test_acc = (pred[data.test_mask] == y[data.test_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
            acc_list.append(test_acc)
            # print(f"test={test_acc:.3f} train_loss ={loss.item():.3f} val_loss={loss_vl.item():.3f} at {epoch}")
        if loss_vl.item()<best_loss:
            best_loss = loss_vl.item()
            # print(f"best loss{best_loss} at {epoch}")
            patience_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count +=1
        if patience_count>=patience:
            break
        # early_stopper.step(loss.item(), epoch)
        # if early_stopper.early_stop:
        #     print(f"Stopping at epoch {epoch} due to loss plateau.")
        #     break
        # # ---- Eval (on true labels) ----
        # model.eval()
        # with torch.no_grad():
        #     logits = model(data.x, data.edge_index)
        #     pred = logits.argmax(dim=-1)
        #     y = data.y
        #     train_acc = (pred[data.train_mask] == y[data.train_mask]).float().mean().item()
        #     val_acc = (pred[data.val_mask] == y[data.val_mask]).float().mean().item()
        #     test_acc = (pred[data.test_mask] == y[data.test_mask]).float().mean().item()

        # Simple early stopping on val_acc
        # if val_acc > best_val:
        #     best_val = val_acc
        #     best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        #     patience = cfg.early_patience
        # else:
        #     patience -= 1
        # print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")
        # if patience <= 0:
        #     break
    # if cfg.loss=='kl':
    #     np.save('./motivation/loss_kl.npy',np.array(Loss_list))
    #     np.save('./motivation/acc_kl.npy',np.array(acc_list))
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        if cfg.model == 'linkx':
            logits = model(data.x, A)  # [N, C]
        else:
            logits = model(data.x, data.edge_index)  # [n, C]
        pred = logits.argmax(dim=-1)
        y = data.y
        #test_acc = (pred[data.test_mask] == y[data.test_mask]).float().mean().item()
        out = F.one_hot(pred,num_classes=logits.size(-1)).float()
        os.makedirs(os.path.join(cfg.out_dir,cfg.model), exist_ok=True)  # creates it (and parents) if missing
        # if cfg.loss == "klce":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_klce.pt')
        # elif cfg.loss=="kl":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_kl.pt')
        # elif cfg.loss=="ce":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_ce.pt')
        # elif cfg.loss=="klreg":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_klreg.pt')
        # elif cfg.loss =="ot":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_ot.pt')
        # elif cfg.loss =="otkl":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_otkl.pt')
        # elif cfg.loss =="otklreg":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_klgraph.pt')
        # elif cfg.loss =="klgraph":
        #     filepath =  os.path.join(cfg.out_dir, cfg.model+'/logits_klgraph.pt')

        if cfg.loss == "klce":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_klce.pt')
        elif cfg.loss=="kl":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_kl.pt')
            torch.save(out,filepath)
        elif cfg.loss=="ce":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_ce.pt')
        elif cfg.loss=="klreg":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_klreg.pt')
        elif cfg.loss =="ot":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_ot.pt')
        elif cfg.loss =="otkl":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_otkl.pt')
        elif cfg.loss =="otklreg":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_klgraph.pt')
        elif cfg.loss =="klgraph":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_klgraph.pt')
        elif cfg.loss =="vae":
            filepath =  os.path.join(cfg.out_dir, cfg.model+'/labels_vae.pt')
        #torch.save(logits,filepath)
        #torch.save(out,filepath)
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        print(f"final test={test_acc:.3f}")
    # filepath2 =  os.path.join(cfg.bags_dir, cfg.model+'/loss_record.npy')
    # filepath3 =  os.path.join(cfg.bags_dir, cfg.model+'/acc_record.npy')
    # np.save(filepath2,np.array(Loss_list))
    # np.save(filepath3,np.array(acc_list))
    return model, logits,y









# ===================== CLI =====================

def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser(description="LLP‑GNN bag‑KL trainer on Cora")
    ap.add_argument('--dataset-root', type=str, default='./data/Planetoid')
    ap.add_argument('--name', type=str, default='Cora')
    ap.add_argument('--bags-dir', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--target', type=str, choices=['clean','noisy'], default='clean')
    ap.add_argument('--model', type=str, choices=['gcn','sage','rgcn','linkx','appnp'], default='gcn')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--max-epochs', type=int, default=400)
    ap.add_argument('--early-patience', type=int, default=50)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--lam-smooth', type=float, default=0)
    ap.add_argument('--lam-ent', type=float, default=0.0)
    ap.add_argument('--lam-contrastive', type=float, default=0.0)
    ap.add_argument('--loss', type=str, choices=['ce','kl','klce','klreg','ot','otkl','klgraph','otklreg','vae'], default='ce')
    ap.add_argument('--add-bag',action='store_true')
    ap.add_argument('--graphreg',type=float,default=0.0)
    ap.add_argument('--alpha',type=float,default=2.576)
    ap.add_argument('--beta',type=float,default=1)
    ap.add_argument('--k-aug',type=int,default=2)
    ap.add_argument('--seed',type=int,default=60)
    ap.add_argument('--T', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    return TrainCfg(**vars(args))


if __name__ == '__main__':
    cfg = parse_args()
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(cfg)



    # seeds = [10, 20, 30, 40]  # list of seeds for ensemble
    # device = torch.device(cfg.device)

    # all_preds = []  # to store predictions from each model

    # for seed in seeds:
    #     # Set seed
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    #     # Train the model (returns trained model or logits)
    #     model, logits,y = train(cfg)  # modify train() to optionally return logits

    #     # Store soft predictions (probabilities)
    #     probs = torch.softmax(logits, dim=-1)  # [num_nodes, num_classes]
    #     all_preds.append(probs)

    # # Ensemble: average probabilities across seeds
    # ensemble_probs = torch.stack(all_preds, dim=0).mean(dim=0)

    # # Final predictions
    # ensemble_pred = ensemble_probs.argmax(dim=-1)
    # test_acc = (ensemble_pred == y).float().mean().item()
    # print(f"Ensemble test accuracy: {test_acc:.4f}")
