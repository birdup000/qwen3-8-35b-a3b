import torch

from qwen3_8_moe import tiny_config
from qwen3_8_moe.layers import SparseMoeBlock, TopKRouter


def test_router_selects_top_k_and_renormalizes():
    config = tiny_config().text_config
    router = TopKRouter(config)
    hidden = torch.randn(6, config.hidden_size)
    logits, weights, indices = router(hidden)
    assert logits.shape == (6, config.num_experts)
    assert indices.shape == (6, config.num_experts_per_tok)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert indices.max() < config.num_experts


def test_shared_expert_always_runs():
    config = tiny_config().text_config
    block = SparseMoeBlock(config)
    hidden = torch.randn(2, 5, config.hidden_size)
    out, router_logits = block(hidden)
    assert out.shape == hidden.shape
    assert router_logits.shape[-1] == config.num_experts
    assert torch.isfinite(out).all()
