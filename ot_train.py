# ot_only.py
from typing import Optional, Literal, Dict
import torch
import torch.nn.functional as F

# ---------- Cost builders ----------
class OTCost:
    @staticmethod
    def neg_logp(logits: torch.Tensor, temperature: float = 1.0):
        """
        cost[i,k] = -log softmax_k(z_i / T)
        Returns (cost, probs p)
        """
        p = F.softmax(logits / temperature, dim=-1)
        cost = -torch.log(p.clamp_min(1e-20))
        return cost, p

    @staticmethod
    def neg_logit(logits: torch.Tensor):
        """
        cost[i,k] = -z_{i,k}
        """
        p = F.softmax(logits, dim=-1)
        cost = -logits
        return cost, p

    @staticmethod
    def margin(logits: torch.Tensor):
        """
        cost[i,k] = max_j z_{i,j} - z_{i,k}
        """
        p = F.softmax(logits, dim=-1)
        top = logits.max(dim=-1, keepdim=True).values
        cost = (top - logits)
        return cost, p



def dirichlet_energy_from_adj_sparse(A: torch.Tensor, P: torch.Tensor, symmetrize: bool = True) -> torch.Tensor:
    A = A.coalesce()
    if symmetrize:
        i, j = A.indices()
        v = A.values()
        AT = torch.sparse_coo_tensor(torch.stack([j, i], 0), v, A.size(), device=A.device).coalesce()
        A = torch.sparse_coo_tensor(
            torch.cat([A.indices(), AT.indices()], dim=1),
            torch.cat([A.values(), AT.values()], dim=0),
            A.size(), device=A.device
        ).coalesce()
        A = torch.sparse_coo_tensor(A.indices(), 0.5 * A.values(), A.size(), device=A.device).coalesce()

    d = torch.sparse.sum(A, dim=1).to_dense().unsqueeze(1)  # [n,1]
    AP = torch.sparse.mm(A, P)                              # [n,K]
    LP = d * P - AP
    return (P * LP).sum()


def add_graph_cost(
    cost: torch.Tensor,
    p: torch.Tensor,
    adj: Optional[torch.Tensor],
    lambda_g: float = 0.0,
    row_normalized: bool = True,
) -> torch.Tensor:
    """
    Adds a graph-aware disagreement term:
      cost[i,k] += lambda_g * (1 - avg_{j in N(i)} p[j,k])

    adj can be dense or sparse_coo. If not row-normalized, we row-normalize it.
    """
    if adj is None or lambda_g == 0.0:
        return cost

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

    #penalty = -dirichlet_energy_from_adj_sparse(adj,p)
    neighbor_p = torch.sparse.mm(row_norm, p)  # [n,K]
    #penalty = -p*neighbor_p
    # print(penalty.shape)
    penalty = (1.0 - neighbor_p)
    return cost + lambda_g * penalty


# ---------- Sinkhorn (log-domain, stable) ----------
def sinkhorn_transport(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100,
    tol: float = 0.0,
) -> torch.Tensor:
    """
    Entropic OT with cost and marginals a (rows), b (cols).
    cost: [n,K], a: [n] sum=1, b: [K] sum=1
    Returns T with row sums a and col sums b.
    """
    n, K = cost.shape
    Klog = -cost / epsilon
    u = torch.zeros(n, device=cost.device)
    v = torch.zeros(K, device=cost.device)
    for _ in range(max_iter):
        u_prev = u
        u = torch.log(a + 1e-20) - torch.logsumexp(Klog + v[None, :], dim=1)
        v = torch.log(b + 1e-20) - torch.logsumexp(Klog + u[:, None], dim=0)

        if tol > 0.0 and torch.max(torch.abs(u - u_prev)) < tol:
            break
    T = torch.exp(Klog + u[:, None] + v[None, :])
    # Given: cost [n,K], a [n], b [K], T [n,K]
    # 1) Marginal errors
    # row_err = (T.sum(dim=1) - a).abs().max().item()   # want ~1e-6 to 1e-4
    # col_err = (T.sum(dim=0) - b).abs().max().item()

    # # 2) Pseudo-labels and their row-wise entropies
    # A = T / (a[:, None] + 1e-20)                      # row-stochastic
    # A_safe = A.clamp_min(1e-12)
    # row_entropy = -(A_safe * A_safe.log()).sum(dim=1) # per-node entropy (nats)
    # H_mean = row_entropy.mean().item()
    # H_median = row_entropy.median().item()

    # # (Optional) confidence (avg max prob per row)
    # mean_conf = A.max(dim=1).values.mean().item()

    # print(f"row_err={row_err:.2e}, col_err={col_err:.2e}, H_mean={H_mean:.3f}, mean_conf={mean_conf:.3f}")

    return T


# ---------- Quota rounding (for hard labels) ----------
@torch.no_grad()
def round_counts(n: int, q: torch.Tensor) -> torch.Tensor:
    """
    Largest remainder method for integer class counts summing to n.
    q: [K], sum≈1
    """
    q = q / q.sum().clamp_min(1e-20)
    c_float = n * q
    c_floor = torch.floor(c_float)
    r = int(n - int(c_floor.sum().item()))
    c = c_floor.long()
    if r > 0:
        frac = (c_float - c_floor)
        top = torch.topk(frac, r).indices
        c[top] += 1
    return c  # [K], sum=n


@torch.no_grad()
def greedy_quota_assign(cost: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """
    Greedy min-cost assignment under class quotas.
    cost: [n,K], counts: [K] integers, sum = n
    Returns y_hat [n] (class per instance).
    """
    n, K = cost.shape
    scores = -cost
    i_idx = torch.arange(n, device=cost.device)[:, None].expand(n, K).reshape(-1)
    k_idx = torch.arange(K, device=cost.device)[None, :].expand(n, K).reshape(-1)
    flat_scores = scores.reshape(-1)
    order = torch.argsort(flat_scores, descending=True)

    assigned = torch.full((n,), -1, device=cost.device, dtype=torch.long)
    rem = counts.clone()

    for idx in order:
        i = i_idx[idx]; k = k_idx[idx]
        if assigned[i] == -1 and rem[k] > 0:
            assigned[i] = k
            rem[k] -= 1
        if (assigned != -1).all():
            break

    # Fallback if anything remains
    unassigned = (assigned == -1).nonzero(as_tuple=True)[0]
    if unassigned.numel() > 0:
        best_k = scores[unassigned].argmax(dim=1)
        assigned[unassigned] = best_k
    return assigned  # [n]
def exactly_2hop_only_sparse_cuda(A):
    # A: sparse CUDA tensor adjacency matrix
    
    A = A.coalesce()
    A2 = torch.sparse.mm(A, A).coalesce()
    
    # Remove diagonal from A2
    indices = A2.indices()
    values = A2.values()
    non_diag_mask = indices[0] != indices[1]
    indices_no_diag = indices[:, non_diag_mask]
    values_no_diag = values[non_diag_mask]
    A2_no_diag = torch.sparse_coo_tensor(indices_no_diag, values_no_diag, A2.shape, device=A.device).coalesce()
    
    # Get sets of edges
    A_edges = set([ (i.item(), j.item()) for i,j in zip(A.indices()[0], A.indices()[1]) ])
    A2_edges = [ (i.item(), j.item()) for i,j in zip(indices_no_diag[0], indices_no_diag[1]) ]
    
    # Filter edges in A2 that are NOT in A (exclude 1-hop edges)
    filtered_edges = [e for e in A2_edges if e not in A_edges]
    
    if len(filtered_edges) == 0:
        # Return empty sparse tensor if no edges
        A2_only = torch.sparse_coo_tensor(torch.empty((2,0), dtype=torch.long, device=A.device),
                                          torch.tensor([], device=A.device),
                                          A.shape,
                                          device=A.device)
    else:
        filtered_indices = torch.tensor(filtered_edges, dtype=torch.long, device=A.device).t()
        filtered_values = torch.ones(filtered_indices.shape[1], device=A.device)
        A2_only = torch.sparse_coo_tensor(filtered_indices, filtered_values, A.shape, device=A.device).coalesce()
    
    return A2_only


# ---------- Main OT interface (no CE) ----------
def ot_assign(
    logits_b: torch.Tensor,             # [n_b, K]
    q_b: torch.Tensor,                  # [K], sum≈1
    *,
    cost_type: Literal["neg_logp", "neg_logit", "margin"] = "neg_logp",
    temperature: float = 1.0,
    epsilon: float = 0.1,
    sinkhorn_iters: int = 100,
    graph_adj: Optional[torch.Tensor] = None,  # local adj (dense or sparse_coo) for nodes in this bag
    lambda_graph: float = 0.0,
    adj_row_normalized: bool = True,
    return_hard: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Pure OT module: returns pseudo labels without computing CE.
    Outputs:
      A : [n_b,K] soft pseudo labels (rows sum to 1)
      T : [n_b,K] transport plan (rows sum to 1/n_b, cols to q_b)
      p : [n_b,K] model probabilities used to build cost
      (optional)
      y_hat : [n_b] hard labels obeying bag quotas
      counts: [K] integer class counts used for y_hat
    """
    device = logits_b.device
    n_b, K = logits_b.shape
    a = torch.full((n_b,), 1.0 / n_b, device=device)
    b = (q_b / q_b.sum().clamp_min(1e-20)).to(device)

    # Build base cost + probs
    if cost_type == "neg_logp":
        cost, p = OTCost.neg_logp(logits_b, temperature)
    elif cost_type == "neg_logit":
        cost, p = OTCost.neg_logit(logits_b)
    elif cost_type == "margin":
        cost, p = OTCost.margin(logits_b)
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")

    # Optional graph-aware augmentation
    #adj_row_normalized = False
    #graph_adj = exactly_2hop_only_sparse_cuda(graph_adj)
    cost = add_graph_cost(cost, p, graph_adj, lambda_g=lambda_graph, row_normalized=adj_row_normalized)

    # Solve OT
    #print("waiting for solving OT")
    T = sinkhorn_transport(cost, a, b, epsilon=epsilon, max_iter=sinkhorn_iters)
    A = T / (a[:, None] + 1e-20)  # row-stochastic pseudo labels
    #print("OT finished")
    out: Dict[str, torch.Tensor] = {"A": A, "T": T, "p": p}
    #print("waiting for hard return")
    if return_hard:
        counts = round_counts(n_b, b)
        y_hat = greedy_quota_assign(cost, counts)
        out.update({"y_hat": y_hat, "counts": counts})
    #print("hard finished")
    return out
