"""The SDPA joint-attention kernel must reproduce the eager kernel, including GQA and fully masked rows."""
import types
import pytest
import torch
from openpi.models_pytorch import gemma_pytorch
from transformers.models.gemma import modeling_gemma


def _module(groups):
    return types.SimpleNamespace(num_key_value_groups=groups, training=False)


def test_sdpa_matches_eager_with_additive_mask_and_gqa():
    torch.manual_seed(0); b, heads, kv, s, d = 2, 8, 1, 12, 16
    q = torch.randn(b, heads, s, d); k = torch.randn(b, kv, s, d); v = torch.randn(b, kv, s, d)
    allowed = torch.rand(b, 1, s, s) > 0.3; allowed[:, :, 0, :] = False
    mask = torch.where(allowed, 0.0, -2.3819763e38)
    m = _module(heads // kv)
    eager, _ = modeling_gemma.eager_attention_forward(m, q, k, v, mask, d ** -0.5)
    sdpa, _ = gemma_pytorch.sdpa_attention_forward(m, q, k, v, mask, d ** -0.5)
    assert eager.shape == sdpa.shape and torch.isfinite(sdpa).all()
    assert torch.allclose(eager, sdpa, atol=1e-5, rtol=1e-4)
    plain_e, _ = modeling_gemma.eager_attention_forward(m, q, k, v, None, d ** -0.5)
    plain_s, _ = gemma_pytorch.sdpa_attention_forward(m, q, k, v, None, d ** -0.5)
    assert torch.allclose(plain_e, plain_s, atol=1e-5, rtol=1e-4)


def test_dispatcher_selects_and_validates():
    torch.manual_seed(1); m = _module(2)
    q = torch.randn(1, 4, 5, 8); k = torch.randn(1, 2, 5, 8); v = torch.randn(1, 2, 5, 8)
    previous = gemma_pytorch.JOINT_ATTENTION
    try:
        gemma_pytorch.JOINT_ATTENTION = "eager"
        e, _ = gemma_pytorch.joint_attention_forward(m, q, k, v, None, 8 ** -0.5)
        gemma_pytorch.JOINT_ATTENTION = "sdpa"
        s, _ = gemma_pytorch.joint_attention_forward(m, q, k, v, None, 8 ** -0.5)
        assert torch.allclose(e, s, atol=1e-5, rtol=1e-4)
        gemma_pytorch.JOINT_ATTENTION = "flash"
        with pytest.raises(ValueError):
            gemma_pytorch.joint_attention_forward(m, q, k, v, None, 8 ** -0.5)
    finally:
        gemma_pytorch.JOINT_ATTENTION = previous
