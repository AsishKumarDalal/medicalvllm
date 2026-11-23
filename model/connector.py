"""Connector modules: Perceiver resampler and a pooled-MLP baseline.

The connector maps frozen vision patch tokens into the decoder's embedding
space. See experiment/connector.md for the design and diagrams.
"""

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Map ``[B, P, dim]`` patch tokens to ``[B, K, dim]`` visual tokens.

    K is the number of learned queries (``num_queries``). Each block is a
    pre-norm residual cross-attention + FFN, as in Flamingo's Perceiver
    Resampler.
    """

    def __init__(
        self,
        dim: int = 768,
        num_queries: int = 64,
        depth: int = 2,
        heads: int = 12,
        ff_mult: int = 4,
        gate_init: float = 0.1,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} is not divisible by heads={heads}")
        self.dim = dim
        self.num_queries = num_queries

        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    dict(
                        ln_q=nn.LayerNorm(dim),
                        ln_kv=nn.LayerNorm(dim),
                        attn=nn.MultiheadAttention(
                            dim, heads, batch_first=True
                        ),
                        ln_ff=nn.LayerNorm(dim),
                        ff=nn.Sequential(
                            nn.Linear(dim, ff_mult * dim),
                            nn.GELU(),
                            nn.Linear(ff_mult * dim, dim),
                        ),
                    )
                )
                for _ in range(depth)
            ]
        )
        self.ln_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 3:
            raise ValueError(
                f"patches must be (batch, seq, dim), got "
                f"{tuple(patches.shape)}"
            )
        batch = patches.size(0)
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)
        for blk in self.blocks:
            kv = blk["ln_kv"](patches)
            q = q + blk["attn"](
                blk["ln_q"](q), kv, kv, need_weights=False
            )[0]
            q = q + blk["ff"](blk["ln_ff"](q))
        return self.gate * self.proj(self.ln_out(q))


class PooledMLPProjector(nn.Module):
    """Cheap baseline: 2x2 average pool -> 2-layer MLP.

    Assumes a square patch grid (196 = 14x14 for 224px / patch16).
    """

    def __init__(self, dim: int = 768, patch_grid: int = 14, pool: int = 2):
        super().__init__()
        if patch_grid % pool != 0:
            raise ValueError(
                f"patch_grid={patch_grid} is not divisible by pool={pool}"
            )
        self.patch_grid = patch_grid
        self.pool = pool
        self.num_queries = (patch_grid // pool) ** 2
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 3:
            raise ValueError(
                f"patches must be (batch, seq, dim), got "
                f"{tuple(patches.shape)}"
            )
        b, n, d = patches.shape
        if n != self.patch_grid**2:
            raise ValueError(
                f"expected {self.patch_grid ** 2} patches, got {n}"
            )
        x = patches.transpose(1, 2).reshape(
            b, d, self.patch_grid, self.patch_grid
        )
        x = nn.functional.avg_pool2d(x, self.pool)
        x = x.flatten(2).transpose(1, 2)
        return self.net(x)


def build_connector(cfg: dict, dim: int):
    """Build from configs/vision.yaml."""
    kind = str(cfg.get("connector", "perceiver")).lower()
    num_tokens = int(cfg.get("num_visual_tokens", 64))
    if kind == "perceiver":
        return PerceiverResampler(
            dim=dim,
            num_queries=num_tokens,
            depth=int(cfg.get("resampler_depth", 2)),
            heads=int(cfg.get("resampler_heads", 12)),
            ff_mult=int(cfg.get("resampler_ff_mult", 4)),
            gate_init=float(cfg.get("output_gate_init", 0.1)),
        )
    if kind in {"pooled_mlp", "pooled-mlp", "mlp"}:
        return PooledMLPProjector(
            dim=dim,
            patch_grid=int(cfg.get("patch_grid", 14)),
            pool=int(cfg.get("pool", 2)),
        )
    raise ValueError(f"unknown connector type: {kind!r}")
