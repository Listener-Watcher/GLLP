
#!/usr/bin/env python3
"""
Cora Bag Generator for LLP-on-Graphs experiments

Generates multiple *active* bagging schemes over the Cora citation graph and
exports bag membership and bag-level label proportions.

Schemes implemented:
  - random_iid: ignores edges, uniform node sampling into bags
  - community: spectral clustering over the graph (fallback if Louvain unavailable)
  - ego_khop: k-hop ego-nets around sampled seeds
  - degree_stratified: buckets by degree, then sample within buckets
  - random_walk: sliding windows from short random walks
  - mixed: configurable mixture of the above schemes

Outputs (per run):
  - bags.jsonl: list of {"bag_id": int, "nodes": [int,...]}
  - bag_proportions.npy: (m, C) array with empirical bag label proportions
  - meta.json: run configuration + dataset/class stats + coverage / rank metrics

Usage examples:
  python cora_bag_generator.py --scheme random_iid --num-bags 800 --bag-size 50
  python cora_bag_generator.py --scheme ego_khop --num-bags 800 --bag-size 60 --k 2
  python cora_bag_generator.py --scheme random_walk --num-bags 1000 --bag-size 40 --walk-len 80 --window 40
  python cora_bag_generator.py --scheme community --target-communities 7 --bag-size-cap 200
  python cora_bag_generator.py --scheme mixed --mixture "random_iid:0.3,ego_khop:0.4,random_walk:0.3" --num-bags 900 --bag-size 50

Dependencies:
  - numpy, scipy, networkx
  - torch, torch_geometric (to load Cora Planetoid); if not available, a fallback URL-free loader is not provided.
  - scikit-learn (for spectral clustering)

"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import networkx as nx

# Optional imports
try:
    import torch
    from torch_geometric.datasets import Planetoid,HeterophilousGraphDataset,Actor,Amazon,WikiCS
    from torch_geometric.utils import to_networkx
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

try:
    from sklearn.cluster import SpectralClustering
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

rng_global = np.random.default_rng(0)

@dataclass
class Config:
    scheme: str
    iid_disjoint: bool
    name:str
    seed: int
    num_bags: int
    bag_size: int
    k: int
    walks_per_seed: int
    walk_len: int
    window: int
    target_communities: int
    bag_size_cap: int
    noise_dirichlet: float
    noise_gaussian: float
    mixture: str
    fraction:float
    out_dir: str


def set_seed(seed: int):
    global rng_global
    rng_global = np.random.default_rng(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    if 'torch' in globals() and PYG_AVAILABLE:
        torch.manual_seed(seed)


def load_cora() -> Tuple[nx.Graph, np.ndarray]:
    assert PYG_AVAILABLE, "torch_geometric is required to load Cora. Install torch-geometric."
    dataset = Planetoid(root=os.path.join("./data", "Planetoid"), name="Cora")
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_computers() -> Tuple[nx.Graph, np.ndarray]:
    assert PYG_AVAILABLE, "torch_geometric is required to load Cora. Install torch-geometric."
    dataset = Amazon(root=os.path.join("./data", "Amazon"), name='Computers')
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_photos() -> Tuple[nx.Graph, np.ndarray]:
    assert PYG_AVAILABLE, "torch_geometric is required to load Cora. Install torch-geometric."
    dataset = Amazon(root=os.path.join("./data", "Amazon"), name='photo')
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_amazon() -> Tuple[nx.Graph,np.ndarray]:
    dataset = HeterophilousGraphDataset(root=os.path.join("./data", "HeterophilousGraphDataset"), name="Amazon-ratings")
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_Roman() -> Tuple[nx.Graph,np.ndarray]:
    dataset = HeterophilousGraphDataset(root=os.path.join("./data", "HeterophilousGraphDataset"), name="Roman-empire")
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_Actor() -> Tuple[nx.Graph,np.ndarray]:
    dataset = Actor(root=os.path.join("./data", "Actor"))
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_Pubmed() -> Tuple[nx.Graph,np.ndarray]:
    dataset = Planetoid(root=os.path.join("./data", "Planetoid"), name="PubMed")
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_CiteSeer() -> Tuple[nx.Graph,np.ndarray]:
    dataset = Planetoid(root=os.path.join("./data", "Planetoid"), name="CiteSeer")
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_Wiki() -> Tuple[nx.Graph,np.ndarray]:
    dataset = WikiCS(root=os.path.join("./data", "WikiCS"))
    print("load data succ")
    data = dataset[0]
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_Covid(day) -> Tuple[nx.Graph,np.ndarray]:
    data = torch.load(f'./data/graph_covid_1000_{day}.pt')
    print("load data succ")
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
def load_Covidbinary(day) -> Tuple[nx.Graph,np.ndarray]:
    data = torch.load(f'./data/graph_covidbinary_1000_{day}.pt')
    print("load data succ")
    G: nx.Graph = to_networkx(data, to_undirected=True)
    y = data.y.cpu().numpy()
    return G, y
# ============== Bag builders ==============

def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def _bag_proportions(bags: List[List[int]], y: np.ndarray, num_classes: int) -> np.ndarray:
    m = len(bags)
    Q = np.zeros((m, num_classes), dtype=float)
    for b, nodes in enumerate(bags):
        cls_counts = np.bincount(y[nodes], minlength=num_classes).astype(float)
        Q[b] = _normalize_counts(cls_counts)
    return Q


def _cap_and_unique(nodes: Iterable[int], cap: Optional[int]=None) -> List[int]:
    arr = list(dict.fromkeys(nodes))  # preserve order, remove dups
    if cap is not None:
        return arr[:cap]
    return arr


def build_random_iid_bags_disjoint(n: int, num_bags: int, bag_size: int) -> List[List[int]]:
    """
    Make IID-looking bags with no overlap by shuffling all nodes once and slicing.
    Behavior:
      - If num_bags * bag_size <= n: returns exactly num_bags bags of size bag_size.
      - If num_bags * bag_size  > n: returns floor(n / bag_size) full bags and
        (optionally) one last partial bag if there are leftovers.
    """
    order = rng_global.permutation(n).tolist()
    needed = num_bags * bag_size

    if needed <= n:
        # exact number of disjoint bags as requested
        bags = [order[i*bag_size:(i+1)*bag_size] for i in range(num_bags)]
        return bags
    else:
        print("error, number of total nodes is smaller than requested, disjoint impossible")
        return



def build_random_iid_bags(n: int, num_bags: int, bag_size: int) -> List[List[int]]:
    bags = []
    for _ in range(num_bags):
        bag = rng_global.choice(n, size=bag_size, replace=False).tolist()
        bags.append(bag)
    return bags


def build_community_bags(G: nx.Graph, target_communities: int, bag_size_cap: int) -> List[List[int]]:
    n = G.number_of_nodes()
    if SKLEARN_AVAILABLE:
        # Spectral clustering on adjacency for approximate communities
        A = nx.to_scipy_sparse_array(G, format='csr')
        sc = SpectralClustering(n_clusters=target_communities, assign_labels='kmeans', affinity='precomputed', random_state=rng_global.integers(1<<30))
        labels = sc.fit_predict(A)
        comms: Dict[int, List[int]] = {}
        for i, c in enumerate(labels):
            comms.setdefault(int(c), []).append(i)
        # shard large communities to respect bag_size_cap
        bags: List[List[int]] = []
        for c, nodes in comms.items():
            if len(nodes) <= bag_size_cap:
                bags.append(nodes)
            else:
                # split into chunks of size cap
                for s in range(0, len(nodes), bag_size_cap):
                    bags.append(nodes[s:s+bag_size_cap])
        return bags
    else:
        # Fallback: degree-ordered partitioning as a weak community proxy
        degs = sorted(G.degree, key=lambda x: (-x[1], x[0]))
        order = [i for i,_ in degs]
        chunk = max(10, bag_size_cap)
        bags = [order[i:i+chunk] for i in range(0, len(order), chunk)]
        return bags

def build_ego_khop_bags_disjoint_full(
    G: nx.Graph,
    num_bags: int,
    k: int,
    target_size: int,
    *,
    keep_all_seeds: bool = True,
    candidate_batches: int = 2,       # how many greedy rounds to find a good extra seed
    candidates_per_round: int = 64,   # how many candidates to try per round
    rng: np.random.Generator = None,
) -> List[List[int]]:
    """
    Build DISJOINT bags of EXACT size `target_size` by greedily unioning k-hop ego-nets
    of multiple seeds chosen from the remaining (unused) nodes.

    Preconditions:
      - num_bags * target_size <= G.number_of_nodes()  (otherwise impossible to be full + disjoint)

    Heuristic:
      - Primary seed ~ sqrt(deg+1) over remaining nodes to avoid hub dominance.
      - While union < target_size, pick an extra seed that maximizes marginal gain
        (approximate greedy maximum coverage). All nodes always restricted to `available`.
      - Finalize by truncating to exactly target_size; keep seeds if `keep_all_seeds=True`
        (or at least keep the primary seed if False).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = G.number_of_nodes()
    assert num_bags * target_size <= n, (
        "Impossible to create full, disjoint bags: num_bags * target_size > #nodes."
    )

    # Precompute degree-based base weights
    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    base_w = np.sqrt(deg + 1.0)
    base_w = base_w / base_w.sum()

    # Fast neighbor access
    nbrs = {u: list(G.neighbors(u)) for u in range(n)}

    available: set[int] = set(range(n))
    bags: List[List[int]] = []

    for b in range(num_bags):
        # If the remaining pool cannot fill a bag, we must stop (shouldn't happen due to assert)
        if len(available) < target_size:
            break

        # ---- 1) choose primary seed from available (weighted) ----
        avail_arr = np.fromiter(available, dtype=int)
        w = base_w[avail_arr]
        w = w / (w.sum() + 1e-12)
        s0 = int(rng.choice(avail_arr, p=w))

        seeds = [s0]
        covered = set(nx.ego_graph(G, s0, radius=k).nodes()) & available

        # ---- 2) keep adding seeds until union reaches target_size ----
        # greedy: pick next seed from candidates that maximizes marginal gain
        rounds = 0
        while len(covered) < target_size:
            rounds += 1
            # candidate pool = available \ covered
            cand_pool = list(available - covered)
            if not cand_pool:  # safety, shouldn't happen due to capacity assert
                break

            # sample a batch of candidates using degree weights to bias toward productive seeds
            cand_pool_arr = np.array(cand_pool, dtype=int)
            w_cand = base_w[cand_pool_arr]
            w_cand = w_cand / (w_cand.sum() + 1e-12)

            best_gain = -1
            best_seed = None

            for _ in range(candidate_batches):
                take = min(candidates_per_round, len(cand_pool_arr))
                # sample without replacement; if pool < needed, numpy handles it
                idx = rng.choice(len(cand_pool_arr), size=take, replace=False, p=w_cand)
                batch = cand_pool_arr[idx]

                for s in batch:
                    # compute marginal gain if we add s's k-hop ego (restricted to available)
                    ego_s = set(nx.ego_graph(G, int(s), radius=k).nodes()) & available
                    gain = len(ego_s - covered)
                    if gain > best_gain:
                        best_gain = gain
                        best_seed = int(s)

            # If no gain (all candidates overlap), just pick a random remaining node as seed.
            if best_seed is None or best_gain <= 0:
                best_seed = int(rng.choice(cand_pool_arr))

            seeds.append(best_seed)
            covered |= (set(nx.ego_graph(G, best_seed, radius=k).nodes()) & available)

            # Extra quick expansion via one-hop layers if still short
            if len(covered) < target_size:
                frontier = list(covered)
                # One (optional) neighbor expansion step inside available
                extra = []
                for u in frontier:
                    for v in nbrs[u]:
                        if v in available and v not in covered:
                            extra.append(v)
                rng.shuffle(extra)
                need = target_size - len(covered)
                covered.update(extra[:need])

            # If still short, loop and add another seed

        # ---- 3) finalize a full bag of EXACT size target_size ----
        covered_list = list(covered)
        rng.shuffle(covered_list)

        if keep_all_seeds:
            # ensure all seeds are included (if many seeds, keep as many as possible)
            seed_set = list(dict.fromkeys(seeds))  # unique, preserve order
            keep = seed_set[:min(len(seed_set), target_size)]
            rest_pool = [u for u in covered_list if u not in keep]
            rng.shuffle(rest_pool)
            need = target_size - len(keep)
            chosen = keep + rest_pool[:need]
        else:
            # guarantee the primary seed is in; fill the rest from covered
            rest_pool = [u for u in covered_list if u != s0]
            need = target_size - 1
            chosen = [s0] + rest_pool[:need]

        # safety: ensure exact size and disjointness
        chosen = chosen[:target_size]
        assert len(set(chosen)) == len(chosen), "Internal error: duplicates slipped in."
        assert set(chosen).issubset(available), "Internal error: picked a used node."

        bags.append(chosen)
        # mark nodes as used (disjointness)
        available.difference_update(chosen)

    return bags




def build_degree_stratified_bags(G: nx.Graph, num_bags: int, bag_size: int, buckets: int = 5) -> List[List[int]]:
    n = G.number_of_nodes()
    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    # Create degree quantile buckets
    qs = np.quantile(deg, q=np.linspace(0, 1, buckets+1))
    indices = [np.where((deg >= qs[i]) & (deg <= qs[i+1]))[0].tolist() for i in range(buckets)]
    # Ensure non-empty buckets by merging sparse ones
    indices = [idx for idx in indices if len(idx) > 0]
    bags = []
    for _ in range(num_bags):
        b = rng_global.integers(len(indices))
        pool = indices[b]
        if len(pool) < bag_size:
            # sample with replacement if necessary
            bag = rng_global.choice(pool, size=bag_size, replace=True).tolist()
        else:
            bag = rng_global.choice(pool, size=bag_size, replace=False).tolist()
        bags.append(bag)
    return bags


def random_walk_windows(G: nx.Graph, start: int, walk_len: int) -> List[int]:
    path = [start]
    cur = start
    for _ in range(walk_len-1):
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        cur = int(rng_global.choice(nbrs))
        path.append(cur)
    return path


def build_random_walk_bags(G: nx.Graph, num_bags: int, bag_size: int, walks_per_seed: int, walk_len: int, window: int) -> List[List[int]]:
    n = G.number_of_nodes()
    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    weights = (deg + 1.0)
    weights /= weights.sum()
    bags = []
    while len(bags) < num_bags:
        s = rng_global.choice(n, p=weights)
        for _ in range(walks_per_seed):
            path = random_walk_windows(G, int(s), walk_len)
            if len(path) < 2:
                continue
            # take a window and sample unique nodes up to bag_size
            if window > len(path):
                window = len(path)
            start_idx = rng_global.integers(0, len(path) - window + 1)
            nodes = list(dict.fromkeys(path[start_idx:start_idx+window]))
            if len(nodes) >= bag_size:
                nodes = rng_global.choice(nodes, size=bag_size, replace=False).tolist()
            else:
                # try to expand with neighbors if short
                exp = []
                for u in nodes:
                    exp.extend(G.neighbors(u))
                exp = [v for v in exp if v not in nodes]
                rng_global.shuffle(exp)
                need = bag_size - len(nodes)
                nodes.extend(exp[:need])
            bags.append(nodes)
            if len(bags) >= num_bags:
                break
    return bags


def build_mixed_bags(G: nx.Graph, cfg: Config, n_nodes: int) -> List[List[int]]:
    # Parse mixture spec like "random_iid:0.3,ego_khop:0.4,random_walk:0.3"
    parts = [p.strip() for p in cfg.mixture.split(',') if p.strip()]
    pairs = []
    for p in parts:
        name, frac = p.split(':')
        pairs.append((name.strip(), float(frac)))
    total = sum(f for _, f in pairs) or 1.0
    pairs = [(n, f/total) for (n, f) in pairs]

    result: List[List[int]] = []
    for name, frac in pairs:
        kbags = max(1, int(round(cfg.num_bags * frac)))
        if name == 'random_iid':
            result += build_random_iid_bags(n_nodes, kbags, cfg.bag_size)
        elif name == 'ego_khop':
            result += build_ego_khop_bags(G, kbags, cfg.k, cfg.bag_size)
        elif name == 'random_walk':
            result += build_random_walk_bags(G, kbags, cfg.bag_size, cfg.walks_per_seed, cfg.walk_len, cfg.window)
        elif name == 'degree_stratified':
            result += build_degree_stratified_bags(G, kbags, cfg.bag_size)
        elif name == 'community':
            result += build_community_bags(G, cfg.target_communities, cfg.bag_size_cap)
        else:
            raise ValueError(f"Unknown scheme in mixture: {name}")
    # If community added too few, pad with random_iid
    while len(result) < cfg.num_bags:
        result += build_random_iid_bags(n_nodes, 1, cfg.bag_size)
    return result[:cfg.num_bags]


def _iid_bag_from_available(available: set, bag_size: int) -> List[int]:
    """Uniform-without-replacement from remaining nodes."""
    if len(available) < bag_size:
        raise ValueError("Not enough nodes left to make a full IID bag disjointly.")
    pool = np.fromiter(available, dtype=int)
    pick = rng_global.choice(pool, size=bag_size, replace=False).tolist()
    return pick

def _ego_union_bag_from_available(
    G: nx.Graph,
    available: set,
    k_hop: int,
    bag_size: int,
    deg_weights: np.ndarray,
    candidate_trials: int = 64
) -> List[int]:
    """
    Build one disjoint ego-style bag:
      1) choose a primary seed from `available` (weighted by sqrt(deg+1))
      2) take its k-hop ego ∩ available
      3) while |union| < bag_size: pick another seed (from remaining) that maximizes
         marginal gain; union its k-hop ego ∩ available
      4) truncate to bag_size, keep the primary seed
    """
    if len(available) < bag_size:
        raise ValueError("Not enough nodes left to make a full ego bag disjointly.")

    avail_arr = np.fromiter(available, dtype=int)
    w = deg_weights[avail_arr]; w = w / (w.sum() + 1e-12)
    s0 = int(rng_global.choice(avail_arr, p=w))

    covered = set(nx.ego_graph(G, s0, radius=k_hop).nodes()) & available
    seeds = [s0]

    # Greedily add seeds to grow coverage
    while len(covered) < bag_size:
        # candidate pool: available \ covered
        cand_pool = list(available - covered)
        if not cand_pool:
            break  # defensive; should not happen if capacity holds

        cand_arr = np.array(cand_pool, dtype=int)
        w_cand = deg_weights[cand_arr]; w_cand = w_cand / (w_cand.sum() + 1e-12)

        best_seed, best_gain = None, -1
        # Try a batch of candidates, pick the one with largest marginal gain
        take = min(candidate_trials, len(cand_arr))
        idxs = rng_global.choice(len(cand_arr), size=take, replace=False, p=w_cand)
        batch = cand_arr[idxs]
        for s in batch:
            ego_s = set(nx.ego_graph(G, int(s), radius=k_hop).nodes()) & available
            gain = len(ego_s - covered)
            if gain > best_gain:
                best_gain, best_seed = gain, int(s)

        if best_seed is None:
            # Fallback: pick any remaining node as seed
            best_seed = int(rng_global.choice(cand_arr))
        seeds.append(best_seed)
        covered |= (set(nx.ego_graph(G, best_seed, radius=k_hop).nodes()) & available)

        # Optional tiny expansion by one-hop layer, within `available`
        if len(covered) < bag_size:
            extra = []
            for u in list(covered):
                extra.extend([v for v in G.neighbors(u) if (v in available and v not in covered)])
            rng_global.shuffle(extra)
            need = bag_size - len(covered)
            covered.update(extra[:need])

    # Finalize: exact size, keep the primary seed
    cov_list = list(covered)
    rng_global.shuffle(cov_list)
    if s0 in cov_list:
        cov_list.remove(s0)
    chosen = [s0] + cov_list[:bag_size-1]
    assert len(chosen) == bag_size
    return chosen

def build_mixed_disjoint_bags(
    G: nx.Graph,
    num_bags: int,
    bag_size: int,
    *,
    k_hop: int = 2,
    ego_fraction: float = 0.5,
    interleave: bool = True,
) -> List[List[int]]:
    """
    Make `num_bags` disjoint bags of size `bag_size`, mixing `ego_khop` and `random_iid`.

    - `ego_fraction` is the fraction of bags built with ego_khop (rest are IID).
    - If `interleave=True`, we alternate ego/IID to avoid depleting easy nodes early.
    """
    n = G.number_of_nodes()
    if num_bags * bag_size > n:
        raise ValueError("Impossible to make all bags full and disjoint: num_bags*bag_size > #nodes.")

    # Precompute degree weights for seeds: sqrt(deg+1)
    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    deg_weights = np.sqrt(deg + 1.0)

    k_ego = int(round(num_bags * ego_fraction))
    k_iid = num_bags - k_ego

    order = (["ego"] * k_ego) + (["iid"] * k_iid)
    if interleave:
        # interleave like ego,iid,ego,iid,...
        ego_list = ["ego"] * k_ego
        iid_list = ["iid"] * k_iid
        order = []
        while ego_list or iid_list:
            if ego_list:
                order.append(ego_list.pop())
            if iid_list:
                order.append(iid_list.pop())

    available: set[int] = set(range(n))
    out: List[List[int]] = []

    for tag in order:
        if tag == "iid":
            bag = _iid_bag_from_available(available, bag_size)
        else:  # "ego"
            bag = _ego_union_bag_from_available(G, available, k_hop, bag_size, deg_weights)
        out.append(bag)
        # enforce disjointness
        available.difference_update(bag)

    return out




















# ============== Noise & Metrics ==============

def add_proportion_noise(Q: np.ndarray, dirichlet_alpha: float = 0.0, gaussian_sigma: float = 0.0) -> np.ndarray:
    """Return *noisy* targets Q_tilde while preserving simplex constraints.
       - dirichlet_alpha>0: sample from Dir(alpha * Q[b])
       - gaussian_sigma>0: add N(0, sigma^2) then project onto simplex
    """
    m, C = Q.shape
    Q_noisy = Q.copy()
    if dirichlet_alpha > 0:
        for b in range(m):
            alpha = dirichlet_alpha * np.clip(Q[b], 1e-8, 1.0)
            Q_noisy[b] = rng_global.dirichlet(alpha)
    if gaussian_sigma > 0:
        Q_noisy += rng_global.normal(0.0, gaussian_sigma, size=Q_noisy.shape)
        # project onto simplex per-row
        Q_noisy = project_rows_to_simplex(Q_noisy)
    return Q_noisy


def project_rows_to_simplex(M: np.ndarray) -> np.ndarray:
    # Euclidean projection onto probability simplex per row (Michelot 1986)
    X = M.copy()
    for i in range(X.shape[0]):
        x = X[i]
        u = np.sort(x)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        X[i] = np.maximum(x - theta, 0)
    return X


def incidence_matrix(bags: List[List[int]], n_nodes: int) -> np.ndarray:
    # Row-normalized incidence (each row sums to 1)
    m = len(bags)
    A = np.zeros((m, n_nodes), dtype=float)
    for r, nodes in enumerate(bags):
        if not nodes:
            continue
        A[r, nodes] = 1.0 / len(nodes)
    return A


def coverage_stats(bags: List[List[int]], n_nodes: int) -> Dict[str, float]:
    counts = np.zeros(n_nodes, dtype=int)
    for nodes in bags:
        counts[nodes] += 1
    return {
        "min_bag_count": int(counts.min()),
        "max_bag_count": int(counts.max()),
        "mean_bag_count": float(counts.mean()),
        "median_bag_count": float(np.median(counts)),
        "unused_nodes": int((counts == 0).sum()),
    }


# ============== Orchestrator ==============

def run(cfg: Config):
    set_seed(cfg.seed)
    if cfg.name == 'Cora':
        G, y = load_cora()
    elif cfg.name == 'Amazon':
        G, y = load_amazon()
    elif cfg.name == 'PubMed':
        G, y = load_Pubmed()
    elif cfg.name == 'Roman':
        G, y = load_Roman()
    elif cfg.name == 'Actor':
        G, y = load_Actor()
    elif cfg.name == 'CiteSeer':
        G, y = load_CiteSeer()
    elif cfg.name == 'Photos':
        G, y = load_photos()
    elif cfg.name == 'Computers':
        G, y = load_computers()
    elif cfg.name=='WikiCS':
        G, y = load_Wiki()
    elif cfg.name =='Covid':
        G, y = load_Covid(500)
    elif cfg.name =='Covid700':
        G, y = load_Covid(700)
    elif cfg.name =='Covidbinary':
        G, y = load_Covidbinary(500)
    else:
        return
    n = G.number_of_nodes()
    C = int(y.max()) + 1

    if cfg.scheme == 'random_iid':
        if getattr(cfg, 'iid_disjoint', False):
            print("disjoint")
            bags = build_random_iid_bags_disjoint(n, cfg.num_bags, cfg.bag_size)
        else:
            bags = build_random_iid_bags(n, cfg.num_bags, cfg.bag_size)
    elif cfg.scheme == 'community':
        bags = build_community_bags(G, cfg.target_communities, cfg.bag_size_cap)
    elif cfg.scheme == 'ego_khop':
        if getattr(cfg,'iid_disjoint',False):
            print("disjoint")
            bags = build_ego_khop_bags_disjoint_full(G,cfg.num_bags,cfg.k,cfg.bag_size)
        # bags = build_ego_khop_bags(G, cfg.num_bags, cfg.k, cfg.bag_size)
    elif cfg.scheme == 'degree_stratified':
        bags = build_degree_stratified_bags(G, cfg.num_bags, cfg.bag_size)
    elif cfg.scheme == 'random_walk':
        bags = build_random_walk_bags(G, cfg.num_bags, cfg.bag_size, cfg.walks_per_seed, cfg.walk_len, cfg.window)
    elif cfg.scheme == 'mixed':
        if getattr(cfg,'iid_disjoint',False):
            print("disjoint")
            bags = build_mixed_disjoint_bags(G,cfg.num_bags, cfg.bag_size,k_hop = cfg.k,ego_fraction=cfg.fraction)
        else:
            bags = build_mixed_bags(G, cfg, n)
    else:
        raise ValueError(f"Unknown scheme: {cfg.scheme}")

    # Compute clean proportions and optionally a noisy target version
    Q_clean = _bag_proportions(bags, y, C)
    Q_noisy = add_proportion_noise(Q_clean, cfg.noise_dirichlet, cfg.noise_gaussian)

    # Metrics
    A = incidence_matrix(bags, n)
    rank = int(np.linalg.matrix_rank(A))
    cov = coverage_stats(bags, n)

    # Output
    os.makedirs(cfg.out_dir, exist_ok=True)
    bags_path = os.path.join(cfg.out_dir, 'bags.jsonl')
    props_path = os.path.join(cfg.out_dir, 'bag_proportions.npy')
    props_noisy_path = os.path.join(cfg.out_dir, 'bag_proportions_noisy.npy')
    meta_path = os.path.join(cfg.out_dir, 'meta.json')

    with open(bags_path, 'w') as f:
        for i, nodes in enumerate(bags):
            f.write(json.dumps({"bag_id": i, "nodes": nodes}) + "\n")

    np.save(props_path, Q_clean)
    np.save(props_noisy_path, Q_noisy)

    meta = {
        "config": asdict(cfg),
        "dataset": {
            "name": "Cora",
            "num_nodes": n,
            "num_edges": int(G.number_of_edges()),
            "num_classes": C,
            "class_hist": np.bincount(y, minlength=C).tolist(),
        },
        "bag_stats": {
            "num_bags": len(bags),
            "avg_bag_size": float(np.mean([len(b) for b in bags])) if bags else 0.0,
            "incidence_rank": rank,
        },
        "coverage": cov,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(bags)} bags -> {cfg.out_dir}")
    print(f"Incidence rank: {rank} / {n}")
    print(f"Coverage: {cov}")


# ============== CLI ==============

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Cora LLP bag generator")
    p.add_argument('--scheme', type=str, required=True,
                   choices=['random_iid','community','ego_khop','degree_stratified','random_walk','mixed'])
    # in parse_args()
    p.add_argument('--iid-disjoint', action='store_true',
                   help='For random_iid, make bags disjoint over nodes (no overlap).')
    p.add_argument('--name', type=str, default="Cora")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-bags', type=int, default=800)
    p.add_argument('--bag-size', type=int, default=50)
    p.add_argument('--k', type=int, default=2, help='k-hop for ego_khop')
    p.add_argument('--walks-per-seed', type=int, default=3)
    p.add_argument('--walk-len', type=int, default=80)
    p.add_argument('--window', type=int, default=40)
    p.add_argument('--target-communities', type=int, default=7)
    p.add_argument('--bag-size-cap', type=int, default=200)
    p.add_argument('--noise-dirichlet', type=float, default=0.0)
    p.add_argument('--noise-gaussian', type=float, default=0.0)
    p.add_argument('--mixture', type=str, default="random_iid:0.3,ego_khop:0.4,random_walk:0.3")
    p.add_argument('--fraction',type=float,default=0.5)
    p.add_argument('--out-dir', type=str, default="./bags_cora")
    args = p.parse_args()
    return Config(**vars(args))


if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)
