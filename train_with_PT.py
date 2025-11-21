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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.datasets import Planetoid,HeterophilousGraphDataset,Actor,Amazon,WikiCS
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected, get_laplacian
from geomloss import SamplesLoss
from utils import optimize_ot
from utils import edge_index_to_sparse_adj
from models import LINKX_Sparse
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
    ot:bool
    cor:bool
    loss:str
    seed:int
    device: str


# ===================== Data loading =====================
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
    data.train_mask = data.train_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.edge_index = to_undirected(data.edge_index)
    #print(ds.num_classes)
    return data, ds.num_classes
def load_planetoid_CiteSeer(root: str):
    ds = Planetoid(root=root, name='CiteSeer',split="geom-gcn")
    data = ds[0]
    data.train_mask = data.train_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    data.val_mask = data.val_mask[:,0]
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
    #probs = outputs
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
    elif cfg.name == 'Covidbinary':
        data,num_classes = load_Covidbinary(cfg.dataset_root,500)
    else:
        return
    data = data.to(device)
    y = data.y.to(device)
    bags, Q = load_bags(cfg.bags_dir, cfg.target)
    train_mask = torch.zeros(y.shape, dtype=torch.bool)
    for i,b in enumerate(bags):
        train_mask[b]=True # obtain first 5 labels for cheating
    train_mask = train_mask.to(device)
    test_mask = ~train_mask   # logical NOT
    Q = Q.to(device)
    if cfg.loss =='ce':
        filepath =  os.path.join(cfg.bags_dir, cfg.model+'/labels_ce.pt')
    elif cfg.loss =='klce':
        filepath =  os.path.join(cfg.bags_dir, cfg.model+'/labels_klce.pt')
    else:
        filepath =  os.path.join(cfg.bags_dir, cfg.model+'/labels_kl.pt')

    pseudo_label = torch.load(filepath).to(device)
    # num classes from the pseudo label matrix (safer than hardcoding 7)
    if cfg.cor:
        num_classes = pseudo_label.size(1)

        y_onehot = F.one_hot(y.to(torch.long), num_classes=num_classes).float()
        pseudo_label[train_mask] = y_onehot[train_mask]
    in_dim = data.x.size(1)
    if cfg.model == 'gcn':
        model = GCN(in_dim, cfg.hidden, num_classes, cfg.layers, cfg.dropout)
    elif cfg.model == 'sage':
        model = SAGE(in_dim, cfg.hidden, num_classes, cfg.layers, cfg.dropout)
    elif cfg.model == 'rgcn':
        model = RGCN(in_dim, cfg.hidden, num_classes, cfg.layers,2, cfg.dropout)
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

    best_val = -1e9
    best_state = None
    patience = cfg.early_patience
    patience_count = 0
    # early_stopper = LossPlateauEarlyStopping(patience=patience, delta=1e-3)
    best_loss = 1e9
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
        probs,loss = mycriterion(logits,pseudo_label)
        # split = 15
        #loss_kl_train = kl_bag_loss(p_bags[0:split], Q[0:split])
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
    
        loss_vl = loss
        # loss = loss_ot_train+cfg.lam_contrastive*loss_contrastive
        # loss = (1-cfg.lam_contrastive)*loss_kl_train+cfg.lam_contrastive*loss_contrastive
        # loss = loss_kl
        # loss = loss_ot
        if cfg.lam_smooth > 0:
            loss = loss + cfg.lam_smooth * laplacian_smoothing(p_node, data.edge_index)
        if cfg.lam_ent != 0:
            # negative sign: encourage entropy if lam_ent>0
            #print(entropy_reg(p_node))
            loss = loss - cfg.lam_ent * entropy_reg(p_node)
        loss.backward()
        opt.step()
        if cfg.ot:
            for i, b in enumerate(bags):
                partial = probs[b.to(device)]
                results = optimize_ot(partial.cpu().detach().numpy(),p_bags[i].cpu().detach().numpy())
                pseudo_label[b] = torch.FloatTensor(results).to(device)

        with torch.no_grad():
            if cfg.model == 'linkx':
                logits = model(data.x, A)  # [N, C]
            else:
                logits = model(data.x, data.edge_index)  # [n, C]
            pred = logits.argmax(dim=-1)
            y = data.y
            #test_acc = (pred[data.test_mask] == y[data.test_mask]).float().mean().item()
            test_acc = (pred == y).float().mean().item()

            #print(f"test={test_acc:.3f} train_loss ={loss.item():.3f} val_loss={loss_vl.item():.3f} at {epoch}")
        if loss_vl.item()<best_loss:
            best_loss = loss_vl.item()
            #print(f"best loss{best_loss} at {epoch}")
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
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        print(f"final test={test_acc:.3f}")
    return model, logits,y


# ===================== CLI =====================

def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser(description="LLP‑GNN bag‑KL trainer on Cora")
    ap.add_argument('--dataset-root', type=str, default='./data/Planetoid')
    ap.add_argument('--name', type=str, default='Cora')
    ap.add_argument('--bags-dir', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--target', type=str, choices=['clean','noisy'], default='clean')
    ap.add_argument('--model', type=str, choices=['gcn','sage','rgcn','linkx'], default='gcn')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--max-epochs', type=int, default=400)
    ap.add_argument('--early-patience', type=int, default=50)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--lam-smooth', type=float, default=0.1)
    ap.add_argument('--lam-ent', type=float, default=0.0)
    ap.add_argument('--lam-contrastive', type=float, default=0.0)
    ap.add_argument('--ot',action="store_true")
    ap.add_argument('--cor',action="store_true")
    ap.add_argument('--loss', type=str, choices=['ce','kl','klce'], default='ce')
    ap.add_argument('--seed',type=int)
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