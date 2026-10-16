"""A small Vision Transformer, written out so every tensor shape is visible.

Architecture (Dosovitskiy et al., ICLR 2021, "An Image is Worth 16x16 Words"),
at the sizes the brief fixes: embedding width 64, two encoder blocks, four
heads, feed-forward width 128, a learnable class token, learned position
embeddings, pre-norm blocks with residual connections, and a linear head to 10
logits. The encoder is deliberately hand-written rather than
``nn.TransformerEncoderLayer`` because the report needs the attention weights
themselves; ``tests/test_model.py`` checks the attention against
``nn.MultiheadAttention`` given the same weights, so the hand-written version
is verified rather than trusted.

Shape path for batch size B, patch size P, N = (32/P)^2, L = N + 1:

    images            (B, 3, 32, 32)
    patches           (B, N, 3*P*P)      non-overlapping, flattened
    projected tokens  (B, N, 64)         one shared linear layer
    + class token     (B, L, 64)
    + position emb.   (B, L, 64)
    encoder output    (B, L, 64)
    class repr.       (B, 64)            token 0 after the final LayerNorm
    logits            (B, 10)            raw; CrossEntropyLoss applies log-softmax

There is no attention mask anywhere: every patch attends to every patch.
"""

import math

import torch
from torch import nn

from ._config import IMAGE_SIZE


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*P*P), non-overlapping patches in raster order."""
    b, c, h, w = x.shape
    p = patch_size
    if h % p or w % p:
        raise ValueError(f"{h}x{w} image is not divisible by patch size {p}")
    x = x.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)               # (B, h/p, w/p, C, P, P)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


class Attention(nn.Module):
    """Multi-head self-attention with no mask.

    Per head: scores = Q K^T / sqrt(d_head), weights = softmax(scores) along
    the key axis, output = weights V. The heads are concatenated and mixed
    by an output projection.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("embedding width must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)      # each (B, H, L, d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B,H,L,L)
        weights = scores.softmax(dim=-1)
        out = self.attn_drop(weights) @ v         # (B, H, L, d_head)
        out = out.transpose(1, 2).reshape(b, length, dim)
        out = self.proj_drop(self.proj(out))
        return out, (weights if need_weights else None)


class Block(nn.Module):
    """Pre-norm encoder block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, need_weights: bool = False):
        a, weights = self.attn(self.norm1(x), need_weights)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, weights


class ViT(nn.Module):
    def __init__(self, patch_size: int, dim: int = 64, depth: int = 2,
                 heads: int = 4, mlp_dim: int = 128, dropout: float = 0.1,
                 num_classes: int = 10, image_size: int = IMAGE_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.tokens = self.num_patches + 1
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens, dim))
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_dim, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False,
                trace: "dict | None" = None):
        """Logits (B, 10). ``trace``, if given, is filled with every stage's shape.
        With ``return_attn`` returns (logits, [per-block (B, H, L, L) weights])."""
        def note(name, t):
            if trace is not None:
                trace[name] = tuple(t.shape)

        note("images", x)
        patches = patchify(x, self.patch_size)
        note("patches", patches)
        tokens = self.patch_proj(patches)
        note("projected_tokens", tokens)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        note("with_class_token", tokens)
        tokens = self.embed_drop(tokens + self.pos_embed)
        note("with_position_embedding", tokens)

        maps = []
        for block in self.blocks:
            tokens, w = block(tokens, need_weights=return_attn)
            if return_attn:
                maps.append(w)
        note("encoder_output", tokens)
        cls_repr = self.norm(tokens)[:, 0]
        note("class_representation", cls_repr)
        logits = self.head(cls_repr)
        note("logits", logits)
        return (logits, maps) if return_attn else logits

    @torch.no_grad()
    def trace_shapes(self, batch_size: int = 2) -> dict:
        """Shapes from a real forward pass (eval mode, CPU), not from arithmetic."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        trace: dict = {}
        self(torch.zeros(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device),
             trace=trace)
        self.train(was_training)
        return trace

    def parameter_breakdown(self) -> dict:
        """Parameters by component. Sums to ``count_parameters``."""
        n = lambda *mods: sum(p.numel() for m in mods for p in m.parameters())
        return {
            "patch_projection": n(self.patch_proj),
            "class_token": self.cls_token.numel(),
            "position_embedding": self.pos_embed.numel(),
            "encoder_blocks": n(self.blocks),
            "final_norm_and_head": n(self.norm, self.head),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def attention_entries(tokens: int, heads: int, depth: int) -> dict:
    """L^2 score entries per head per layer, and the whole-model total."""
    per = tokens * tokens
    return {"per_head_per_layer": per, "total": per * heads * depth}
