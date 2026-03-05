import math

import torch
from torch import nn
from einops import einsum, rearrange
from jaxtyping import Int


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
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            self.d_ff = math.ceil(math.ceil(8 * d_model / 3) / 64) * 64
        else:
            self.d_ff = d_ff

        self.W_1 = Linear(d_model, self.d_ff, device, dtype)
        self.W_2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.W_3 = Linear(d_model, self.d_ff, device, dtype)

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


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.unsqueeze(torch.max(x, dim=dim)[0], dim)
    y = torch.exp(x - x_max)
    s = torch.unsqueeze(torch.sum(y, dim), dim)
    return y / s


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    QTK = einsum(Q, K, "... i j, ... k j -> ... i k") / math.sqrt(d_k)
    if mask is not None:
        QTK[..., ~mask] -= torch.inf
    sQTK = softmax(QTK, -1)
    return einsum(sQTK, V, "... i j, ... j k -> ... i k")


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_QKV = Linear(d_model, 3 * d_model, device, dtype)
        self.W_O = Linear(d_model, d_model, device, dtype)

        if theta is not None:
            if max_seq_len is None:
                return ValueError
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)
            self.mask = torch.ones(
                max_seq_len, max_seq_len, device=device, dtype=torch.bool
            ).tril()
        else:
            self.rope = None
            self.mask = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        qkv_res = self.W_QKV.forward(x)

        if self.rope is not None:
            if token_positions is None:
                raise ValueError
            qkv_res_rearr = rearrange(
                qkv_res,
                "... (three num_heads k) -> three num_heads ... k",
                three=3,
                num_heads=self.num_heads,
                k=self.d_k,
            )
            q_block = self.rope.forward(qkv_res_rearr[0], token_positions)
            k_block = self.rope.forward(qkv_res_rearr[1], token_positions)
            qkv_res_rearr[0] = q_block
            qkv_res_rearr[1] = k_block

            qkv_res = rearrange(
                qkv_res_rearr,
                "three num_heads ... k -> ... (three num_heads k)",
                three=3,
                num_heads=self.num_heads,
                k=self.d_k,
            )

        qkv_block = rearrange(
            qkv_res,
            "... s (three num_heads k) -> three ... num_heads s k",
            three=3,
            num_heads=self.num_heads,
            k=self.d_k,
        )

        seq_len = qkv_res.shape[-2]
        if self.mask is None:
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        else:
            mask = self.mask[:seq_len, :seq_len]

        atn = scaled_dot_product_attention(
            qkv_block[0],
            qkv_block[1],
            qkv_block[2],
            mask,
        )

        atn_vec = rearrange(
            atn, "... num_heads s k -> ... s (num_heads k)", num_heads=self.num_heads
        )

        return self.W_O.forward(atn_vec)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.rms_norm_mha = RMSNorm(d_model, eps, device, dtype)
        self.multihead_attn = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device, dtype
        )

        self.rms_norm_swiglu = RMSNorm(d_model, eps, device, dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x += self.multihead_attn(self.rms_norm_mha(x), token_positions)
        return x + self.swiglu(self.rms_norm_swiglu(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.transformer_blocks = [
            TransformerBlock(
                d_model, num_heads, d_ff, eps, theta, context_length, device, dtype
            )
            for _ in range(num_layers)
        ]

        self.rms_norm = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, x: Int[torch.Tensor, "..."], token_positions: torch.Tensor | None = None
    ):
        embed = self.embedding.forward(x)

        transformed_embed = embed
        for t in self.transformer_blocks:
            transformed_embed = t.forward(transformed_embed, token_positions)

        logits = self.output_embedding.forward(self.rms_norm.forward(transformed_embed))

        return logits
