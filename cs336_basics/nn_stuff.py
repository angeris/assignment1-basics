import math

import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.W = torch.empty(out_features, in_features, dtype=dtype, device=device)
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "i j, ... j -> ... i")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.embed_mat = torch.empty(
            num_embeddings, embedding_dim, dtype=dtype, device=device
        )
        nn.init.trunc_normal_(self.embed_mat, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_mat[token_ids, :]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.gains = torch.ones(d_model, device=device, dtype=dtype)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sum_sq = einsum(x, x, "... i, ... i -> ...")
        rms = torch.sqrt(sum_sq / len(self.gains) + self.eps)

        gain_prod = einsum(self.gains, x, "i, ... i -> ... i")
        result = einsum(gain_prod, 1 / rms, "... j, ... -> ... j")

        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = math.ceil(math.ceil(8 * d_model / 3) / 64) * 64
        self.W_1 = Linear(d_model, self.d_ff)
        self.W_2 = Linear(self.d_ff, self.d_model)
        self.W_3 = Linear(d_model, self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2.forward(silu(self.W_1.forward(x)) * self.W_3.forward(x))


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.theta = theta
        i_s = torch.arange(0, max_seq_len, device=device)[:, None]
        k = torch.arange(d_k // 2, device=device)[None, :]
        thetas = i_s / theta ** (2 * k / d_k)
        self.cos = torch.cos(thetas)
        self.sin = torch.sin(thetas)
        self.register_buffer("c", self.cos, persistent=False)
        self.register_buffer("s", self.sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        e = x[..., token_positions, ::2]
        o = x[..., token_positions, 1::2]

        c_table = self.cos[token_positions, ...]
        s_table = self.sin[token_positions, ...]

        x_out = torch.empty_like(x)
        x_out[..., token_positions, ::2] = c_table * e - s_table * o
        x_out[..., token_positions, 1::2] = s_table * e + c_table * o

        return x_out
