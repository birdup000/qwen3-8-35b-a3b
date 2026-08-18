"""Closed-form parameter counts for Qwen3.8-35B-A3B (no weight allocation)."""

from __future__ import annotations

from .configuration import Qwen38MoeConfig, Qwen38MoeTextConfig, Qwen38MoeVisionConfig, qwen38_35b_a3b_config


def _full_attention_params(config: Qwen38MoeTextConfig) -> int:
    h, q, kv, d = config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q_out = q * d * (2 if config.attn_output_gate else 1)
    return h * q_out + h * (kv * d) + h * (kv * d) + (q * d) * h + 2 * d


def _linear_attention_params(config: Qwen38MoeTextConfig) -> int:
    h = config.hidden_size
    key_dim = config.linear_key_dim
    value_dim = config.linear_value_dim
    conv_dim = key_dim * 2 + value_dim
    conv = conv_dim * config.linear_conv_kernel_dim
    qkv = h * conv_dim
    z = h * value_dim
    ab = 2 * h * config.linear_num_value_heads
    out = value_dim * h
    scalars = 2 * config.linear_num_value_heads + config.linear_value_head_dim
    return conv + qkv + z + ab + out + scalars


def _moe_params(config: Qwen38MoeTextConfig) -> int:
    h, inter, n = config.hidden_size, config.moe_intermediate_size, config.num_experts
    experts = n * (2 * inter * h + h * inter)
    shared = 3 * h * config.shared_expert_intermediate_size
    shared_gate = h
    router = n * h
    return experts + shared + shared_gate + router


def text_parameter_count(config: Qwen38MoeTextConfig, include_mtp: bool = True) -> dict[str, int]:
    embed = config.vocab_size * config.hidden_size
    lm_head = 0 if config.tie_word_embeddings else config.vocab_size * config.hidden_size
    n_full = sum(kind == "full_attention" for kind in (config.layer_types or []))
    n_lin = sum(kind == "linear_attention" for kind in (config.layer_types or []))
    attn = n_full * _full_attention_params(config) + n_lin * _linear_attention_params(config)
    moe = config.num_hidden_layers * _moe_params(config)
    norms = (2 * config.num_hidden_layers + 1) * config.hidden_size
    mtp = 0
    if include_mtp and config.mtp_num_hidden_layers:
        # Approximate: reuse one decoder layer + fuse proj + norm.
        mtp = _full_attention_params(config) + _moe_params(config) + (2 * config.hidden_size) + (
            config.hidden_size * 2 * config.hidden_size
        )
    total = embed + lm_head + attn + moe + norms + mtp
    return {
        "embeddings": embed,
        "lm_head": lm_head,
        "attention": attn,
        "moe": moe,
        "norms": norms,
        "mtp": mtp,
        "total": total,
    }


def vision_parameter_count(config: Qwen38MoeVisionConfig) -> dict[str, int]:
    patch = config.in_channels * config.hidden_size * config.temporal_patch_size * (config.patch_size ** 2) + config.hidden_size
    pos = config.num_position_embeddings * config.hidden_size
    attn = (config.hidden_size * config.hidden_size * 3 + config.hidden_size) + (config.hidden_size * config.hidden_size + config.hidden_size)
    mlp = (config.hidden_size * config.intermediate_size + config.intermediate_size) + (
        config.intermediate_size * config.hidden_size + config.hidden_size
    )
    norms = 2 * (config.hidden_size + config.hidden_size)  # weight + bias per LN, two LNs
    # LayerNorm has weight+bias.
    norms = 2 * (2 * config.hidden_size)
    block = attn + mlp + norms
    merge_in = config.hidden_size * (config.spatial_merge_size ** 2)
    merger = config.hidden_size + config.hidden_size  # LN
    merger += merge_in * merge_in + merge_in
    merger += merge_in * config.out_hidden_size + config.out_hidden_size
    total = patch + pos + config.depth * block + merger
    return {
        "patch_embed": patch,
        "pos_embed": pos,
        "blocks": config.depth * block,
        "merger": merger,
        "rotary_table": 0,
        "total": total,
    }


def active_text_parameter_count(config: Qwen38MoeTextConfig) -> int:
    """Parameters touched for one token (9 experts + mixers + embed/head)."""
    embed = config.vocab_size * config.hidden_size  # table exists; lookup is one row
    lm_head = 0 if config.tie_word_embeddings else config.vocab_size * config.hidden_size
    n_full = sum(kind == "full_attention" for kind in (config.layer_types or []))
    n_lin = sum(kind == "linear_attention" for kind in (config.layer_types or []))
    attn = n_full * _full_attention_params(config) + n_lin * _linear_attention_params(config)
    expert = 3 * config.hidden_size * config.moe_intermediate_size
    shared = 3 * config.hidden_size * config.shared_expert_intermediate_size + config.hidden_size
    router = config.num_experts * config.hidden_size
    moe_active = config.num_hidden_layers * (config.num_experts_per_tok * expert + shared + router)
    norms = (2 * config.num_hidden_layers + 1) * config.hidden_size
    return embed + lm_head + attn + moe_active + norms


def total_parameter_count(config: Qwen38MoeConfig | None = None, include_vision: bool = True, include_mtp: bool = True) -> int:
    config = config or qwen38_35b_a3b_config()
    total = text_parameter_count(config.text_config, include_mtp=include_mtp)["total"]
    if include_vision:
        total += vision_parameter_count(config.vision_config)["total"]
    return total


def active_parameter_count(config: Qwen38MoeConfig | None = None) -> int:
    config = config or qwen38_35b_a3b_config()
    return active_text_parameter_count(config.text_config)


def parameter_report(config: Qwen38MoeConfig | None = None) -> dict[str, int]:
    config = config or qwen38_35b_a3b_config()
    text = text_parameter_count(config.text_config)
    vision = vision_parameter_count(config.vision_config)
    return {
        **{f"text_{key}": value for key, value in text.items()},
        **{f"vision_{key}": value for key, value in vision.items()},
        "total": text["total"] + vision["total"],
        "active": active_text_parameter_count(config.text_config),
        "num_hidden_layers": config.text_config.num_hidden_layers,
        "num_experts": config.text_config.num_experts,
        "activated_experts": config.text_config.num_experts_per_tok + 1,
    }
