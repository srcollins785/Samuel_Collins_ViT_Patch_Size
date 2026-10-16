"""Shapes, parameter counts, and the hand-written attention.

Each assertion is a claim the report makes. The parameter counts are the
hand calculation, so a wrong hand calculation fails here:

    patch projection  3*P*P*64 + 64     3,136 (P=4)   12,352 (P=8)
    position embed.   L*64              4,160          1,088
    class token       64                   64             64
    two encoder blocks 2 * 33,472       66,944         66,944
    final LayerNorm + head 128 + 650       778            778
    total                               75,082         81,226
"""

import math

import pytest
import torch
from torch import nn

from vit_patch_size.model import (
    Attention, ViT, attention_entries, count_parameters, patchify,
)

EXPECTED_PARAMS = {4: 75_082, 8: 81_226}


def make(p, dropout=0.1):
    torch.manual_seed(0)
    return ViT(p, dropout=dropout)


@pytest.mark.parametrize("p,n", [(4, 64), (8, 16)])
def test_token_counts(p, n):
    m = make(p)
    assert m.num_patches == n and m.tokens == n + 1


@pytest.mark.parametrize("p", [4, 8])
def test_shapes_from_a_real_forward_pass(p):
    m = make(p)
    n, length = m.num_patches, m.tokens
    t = m.trace_shapes(batch_size=3)
    assert t == {
        "images": (3, 3, 32, 32),
        "patches": (3, n, 3 * p * p),
        "projected_tokens": (3, n, 64),
        "with_class_token": (3, length, 64),
        "with_position_embedding": (3, length, 64),
        "encoder_output": (3, length, 64),
        "class_representation": (3, 64),
        "logits": (3, 10),
    }


@pytest.mark.parametrize("p", [4, 8])
def test_parameter_counts(p):
    m = make(p)
    assert count_parameters(m) == EXPECTED_PARAMS[p]
    assert sum(m.parameter_breakdown().values()) == EXPECTED_PARAMS[p]


def test_architecture_matches_the_brief():
    m = make(4)
    assert len(m.blocks) == 2
    assert all(b.attn.heads == 4 for b in m.blocks)
    assert m.blocks[0].mlp[0].out_features == 128
    assert m.head.out_features == 10
    assert m.cls_token.requires_grad and m.pos_embed.requires_grad
    assert sum(isinstance(x, nn.LayerNorm) for x in m.modules()) == 5


def test_attention_entry_counts():
    assert attention_entries(65, 4, 2) == {"per_head_per_layer": 4225, "total": 33800}
    assert attention_entries(17, 4, 2) == {"per_head_per_layer": 289, "total": 2312}
    assert 4225 / 289 == pytest.approx(14.62, abs=0.01)


def test_patchify_is_a_lossless_non_overlapping_partition():
    x = torch.randn(2, 3, 32, 32)
    for p in (4, 8):
        patches = patchify(x, p)
        assert patches.shape == (2, (32 // p) ** 2, 3 * p * p)
        assert torch.equal(patches.sum(), patches.sum())
        assert torch.allclose(patches.sum(), x.sum(), atol=1e-3)
        # Patch 0 is the top-left PxP block, all channels.
        assert torch.equal(patches[0, 0].sort().values,
                           x[0, :, :p, :p].reshape(-1).sort().values)


def test_shared_linear_projection_equals_a_strided_convolution():
    """The 'shared linear patch projection' is the same operator as Conv2d
    with kernel = stride = P; this ties our version to the usual one."""
    p = 4
    m = make(p)
    conv = nn.Conv2d(3, 64, kernel_size=p, stride=p)
    with torch.no_grad():
        conv.weight.copy_(m.patch_proj.weight.reshape(64, 3, p, p))
        conv.bias.copy_(m.patch_proj.bias)
    x = torch.randn(2, 3, 32, 32)
    ours = m.patch_proj(patchify(x, p))
    theirs = conv(x).flatten(2).transpose(1, 2)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_attention_matches_torch_multihead_attention():
    torch.manual_seed(1)
    ours = Attention(64, 4, dropout=0.0).eval()
    ref = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(ours.qkv.weight)
        ref.in_proj_bias.copy_(ours.qkv.bias)
        ref.out_proj.weight.copy_(ours.proj.weight)
        ref.out_proj.bias.copy_(ours.proj.bias)
    x = torch.randn(3, 17, 64)
    out, w = ours(x, need_weights=True)
    ref_out, ref_w = ref(x, x, x, need_weights=True, average_attn_weights=False)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(w, ref_w, atol=1e-6)


def test_attention_rows_sum_to_one_and_are_not_causal():
    m = make(8, dropout=0.0).eval()
    x = torch.randn(2, 3, 32, 32)
    _, maps = m(x, return_attn=True)
    for w in maps:
        assert w.shape == (2, 4, 17, 17)
        assert torch.allclose(w.sum(-1), torch.ones(2, 4, 17), atol=1e-5)
        # No mask: attention to later positions is non-zero.
        assert torch.triu(w, diagonal=1).sum() > 0


def test_every_token_can_influence_every_token():
    """Perturb the last patch only; the class token's output must change.
    With a causal mask the class token (position 0) could not see it."""
    m = make(8, dropout=0.0).eval()
    x = torch.randn(1, 3, 32, 32)
    x2 = x.clone()
    x2[:, :, 24:, 24:] += 5.0                  # the final 8x8 patch
    assert not torch.allclose(m(x), m(x2), atol=1e-6)


def shuffle_patches(x, p, perm):
    """Rearrange the non-overlapping PxP blocks of an image batch."""
    b, c, h, _ = x.shape
    g = h // p
    blocks = x.reshape(b, c, g, p, g, p).permute(0, 1, 2, 4, 3, 5)
    blocks = blocks.reshape(b, c, g * g, p, p)[:, :, perm]
    return (blocks.reshape(b, c, g, g, p, p).permute(0, 1, 2, 4, 3, 5)
            .reshape(b, c, h, h))


def test_position_embedding_breaks_permutation_symmetry():
    """Through the real forward pass. Without positional information, shuffling
    the image's patches cannot change the class output (attention has no notion
    of order); with it, shuffling does. Zeroing the embedding must therefore
    make the model order-blind, and the trained-style embedding must not."""
    p = 8
    m = make(p, dropout=0.0).eval()
    x = torch.randn(1, 3, 32, 32)
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(0))
    xs = shuffle_patches(x, p, perm)
    assert not torch.equal(x, xs)

    with torch.no_grad():
        m.pos_embed.zero_()
        assert torch.allclose(m(x), m(xs), atol=1e-5)      # order-blind
        torch.nn.init.trunc_normal_(m.pos_embed, std=0.5)
        assert not torch.allclose(m(x), m(xs), atol=1e-4)  # order matters


def test_lecture_attention_example():
    """L1's three-value example, with exact softmax (not the slide's rounding)."""
    values = torch.tensor([[0., 0., 0.], [1., 4., 0.], [1., 3., 1.5]])
    scores = torch.tensor([2., 4., 4.])
    w = scores.softmax(0)
    assert torch.allclose(w, torch.tensor([0.0634, 0.4683, 0.4683]), atol=1e-4)
    assert torch.allclose(w @ values, torch.tensor([0.937, 3.278, 0.702]), atol=1e-3)
    ws = (scores / math.sqrt(3)).softmax(0)
    assert torch.allclose(ws, torch.tensor([0.1361, 0.4319, 0.4319]), atol=1e-4)
    assert torch.allclose(ws @ values, torch.tensor([0.864, 3.024, 0.648]), atol=1e-3)
