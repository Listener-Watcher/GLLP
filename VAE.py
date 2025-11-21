# -----------------------
# VAT for node features
# -----------------------
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Any
import math
@torch.no_grad()
def _softmax_detached(logits: torch.Tensor) -> torch.Tensor:
    """Return softmax(logits) with no grad (target distribution)."""
    return F.softmax(logits, dim=-1)

def _l2_normalize(v: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize over `dim`, safe for zero vectors."""
    norm = torch.linalg.norm(v, ord=2, dim=dim, keepdim=True).clamp_min(eps)
    return v / norm

def _kl_p_q_logit(p: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(p || q) where p is a probability (no grad) and q is given as logits.
    Returns per-row KL (no reduction).
    """
    log_q = F.log_softmax(q_logits, dim=-1)
    log_p = torch.log(p.clamp_min(1e-12))
    return torch.sum(p * (log_p - log_q), dim=-1)

def vat_perturbation_on_features(
    forward_fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    *forward_args: Any,
    eps: float = 2.0,
    xi: float = 1e-6,
    ip: int = 1,
    clip_range: Optional[Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute r_adv to add to node features x using 1-step power iteration (default).
    - forward_fn(x, *forward_args) -> logits [N, C]
    - x: [N, F] node features (float)
    - mask: optional boolean/0-1 mask over nodes to restrict VAT to a subset (e.g. training set)
    - clip_range: (low, high) to clamp (x + r) during the inner-gradient step, if your features need bounds

    Returns r_adv with the same shape as x (detach'd, no grad).
    """
    # 1) target p = softmax(logits(x)), detached
    with torch.no_grad():
        p = _softmax_detached(forward_fn(x, *forward_args))  # [N, C]

    # 2) init tiny random direction in feature space
    d = torch.randn_like(x)
    if mask is not None:
        d = d * mask.unsqueeze(-1)  # zero out non-masked nodes
    d = _l2_normalize(d, dim=-1)
    r = xi * d

    # 3) power iteration (usually 1 step is enough)
    for _ in range(ip):
        r.requires_grad_(True)

        # Perturb inside feature space (optionally clip into valid range)
        x_pert = x + r
        if clip_range is not None:
            low, high = clip_range
            x_pert = x_pert.clamp(low, high)

        q_logits = forward_fn(x_pert, *forward_args)  # [N, C]
        # KL(p || q), average over masked nodes only if mask provided
        kl_per_node = _kl_p_q_logit(p, q_logits)
        if mask is not None:
            denom = mask.sum().clamp_min(1)
            kl = (kl_per_node * mask).sum() / denom
        else:
            kl = kl_per_node.mean()

        g = torch.autograd.grad(kl, r, retain_graph=False, create_graph=False)[0]
        if mask is not None:
            g = g * mask.unsqueeze(-1)

        r = xi * _l2_normalize(g.detach(), dim=-1)

    # 4) scale to eps
    r_adv = eps * _l2_normalize(r, dim=-1).detach()
    if mask is not None:
        r_adv = r_adv * mask.unsqueeze(-1)
    return r_adv

def vat_consistency_loss_on_features(
    forward_fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    *forward_args: Any,
    eps: float = 2.0,
    xi: float = 1e-6,
    ip: int = 1,
    clip_range: Optional[Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the VAT consistency loss (mean KL by default) for node features.
    - reduction: 'mean' | 'sum' | 'none'
    """
    with torch.no_grad():
        p = _softmax_detached(forward_fn(x, *forward_args))  # [N, C]

    r_adv = vat_perturbation_on_features(
        forward_fn, x, *forward_args, eps=eps, xi=xi, ip=ip,
        clip_range=clip_range, mask=mask
    )

    x_hat = x + r_adv
    if clip_range is not None:
        low, high = clip_range
        x_hat = x_hat.clamp(low, high)

    q_logits = forward_fn(x_hat, *forward_args)  # [N, C]
    kl_per_node = _kl_p_q_logit(p, q_logits)     # [N]

    if mask is not None:
        if reduction == "none":
            return kl_per_node * mask
        denom = mask.sum().clamp_min(1)
        val = (kl_per_node * mask).sum() / denom
        return val
    else:
        if reduction == "mean":
            return kl_per_node.mean()
        elif reduction == "sum":
            return kl_per_node.sum()
        elif reduction == "none":
            return kl_per_node
        else:
            raise ValueError("reduction must be 'mean' | 'sum' | 'none'")


def sigmoid_rampup(current: int, rampup_length: int) -> float:
    """Sigmoid rampup from 0 to 1 over rampup_length steps."""
    if rampup_length == 0:
        return 1.0
    current = float(current)
    phase = 1.0 - min(current / rampup_length, 1.0)
    return float(math.exp(-5.0 * phase * phase))

def get_alpha(epoch: int, max_alpha: float, rampup_length: int, mode="sigmoid") -> float:
    if mode == "linear":
        return max_alpha * min(1.0, epoch / rampup_length)
    elif mode == "sigmoid":
        return max_alpha * sigmoid_rampup(epoch, rampup_length)
    else:
        return max_alpha
