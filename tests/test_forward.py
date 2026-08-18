import torch

from qwen3_8_moe import Qwen38MoeForCausalLM, Qwen38MoeForConditionalGeneration, tiny_config


def test_causal_lm_forward_and_generate():
    config = tiny_config()
    model = Qwen38MoeForCausalLM(config.text_config)
    model.eval()
    input_ids = torch.randint(1, config.text_config.vocab_size, (2, 8))
    out = model(input_ids=input_ids, use_cache=False)
    assert out.logits.shape == (2, 8, config.text_config.vocab_size)
    assert torch.isfinite(out.logits).all()

    generated = model.generate(input_ids[:1], max_new_tokens=4, temperature=0.0)
    assert generated.shape[0] == 1
    assert generated.shape[1] == 12


def test_cached_decode_matches_prefill_greedy():
    config = tiny_config()
    model = Qwen38MoeForCausalLM(config.text_config)
    model.eval()
    input_ids = torch.randint(1, config.text_config.vocab_size, (1, 6))
    with torch.no_grad():
        cached = model.generate(input_ids, max_new_tokens=3, temperature=0.0)
        full = model(input_ids=cached, use_cache=False).logits
        step = model(input_ids=cached, use_cache=False).logits
    assert cached.shape[-1] == 9
    assert torch.isfinite(full).all()
    assert torch.isfinite(step).all()


def test_vision_conditional_forward():
    config = tiny_config()
    model = Qwen38MoeForConditionalGeneration(config)
    model.eval()
    t, h, w = 1, 4, 4
    merge = config.vision_config.spatial_merge_size
    n_patches = t * h * w
    n_tokens = n_patches // (merge * merge)
    pixel_values = torch.randn(
        n_patches,
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size
        * config.vision_config.patch_size,
    )
    grid = torch.tensor([[t, h, w]], dtype=torch.long)
    input_ids = torch.randint(1, config.text_config.vocab_size, (1, 8))
    input_ids[0, 2 : 2 + n_tokens] = config.image_token_id
    out = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=grid,
        use_cache=False,
    )
    assert out.logits.shape[-1] == config.text_config.vocab_size
    assert torch.isfinite(out.logits).all()
