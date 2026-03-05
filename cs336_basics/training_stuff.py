from typing import Dict, Optional, Callable
import math

import torch
import numpy as np
from einops import einsum


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    logits_max = logits.max(dim=-1)[0]

    norm_logits = logits - logits_max[..., None]
    token_logit = norm_logits.gather(-1, targets[..., None])[..., 0]
    neg_log_p = torch.log(einsum(torch.exp(norm_logits), "... i -> ...")) - token_logit

    return neg_log_p.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps=1e-8):
        defaults = {
            "alpha": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "epsilon": eps,
            "lam": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            epsilon = group["epsilon"]
            lam = group["lam"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))

                g = p.grad.data
                m *= beta_1
                m += (1 - beta_1) * g
                v *= beta_2
                v += (1 - beta_2) * torch.square(g)

                alpha_t = (
                    alpha
                    * math.sqrt(1 - math.pow(beta_2, t))
                    / (1 - math.pow(beta_1, t))
                )

                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)
                p.data *= 1 - alpha * lam

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss


def cosine_schedule(t: int, a_max: float, a_min: float, t_w: int, t_c: int):
    if t < t_w:
        return (t / t_w) * a_max
    if t > t_c:
        return a_min

    return a_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (
        a_max - a_min
    )


def gradient_clipping(params, max_norm, eps=1e-6):
    sum_sq = 0
    for p in params:
        if p.grad is None:
            continue
        sum_sq += torch.linalg.vector_norm(p.grad.data) ** 2

    l2 = math.sqrt(sum_sq)
    if l2 < max_norm:
        return

    for p in params:
        if p.grad is None:
            continue
        p.grad.data *= max_norm / (l2 + eps)
