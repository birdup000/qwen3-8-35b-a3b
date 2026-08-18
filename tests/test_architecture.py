from qwen3_8_moe import qwen38_35b_a3b_config, tiny_config
from qwen3_8_moe.configuration import build_layer_types
from qwen3_8_moe.params import parameter_report


def test_production_layout_matches_qwen36_runtime():
    config = qwen38_35b_a3b_config().text_config
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 40
    assert config.num_experts == 256
    assert config.num_experts_per_tok == 8
    assert config.moe_intermediate_size == 512
    assert config.shared_expert_intermediate_size == 512
    assert config.linear_num_value_heads == 32
    assert config.linear_num_key_heads == 16
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 2
    assert config.head_dim == 256
    assert config.rotary_dim == 64
    assert config.attn_output_gate is True
    assert config.vocab_size == 248320
    assert config.max_position_embeddings == 262144


def test_hybrid_3_to_1_layout():
    types = build_layer_types(40, 4)
    assert types.count("linear_attention") == 30
    assert types.count("full_attention") == 10
    assert types[3] == types[7] == types[39] == "full_attention"
    assert types[0] == types[1] == types[2] == "linear_attention"
    assert qwen38_35b_a3b_config().text_config.layer_types == types


def test_qwen38_intelligence_surface():
    config = qwen38_35b_a3b_config()
    assert config.enable_thinking is True
    assert config.preserve_thinking is True
    assert config.default_reasoning_effort == "xhigh"
    assert config.image_token_id == 248056
    assert config.vision_config.out_hidden_size == config.text_config.hidden_size


def test_parameter_counts_are_35b_a3b():
    report = parameter_report()
    assert 33e9 < report["text_total"] < 37e9
    assert 2.4e9 < report["active"] < 4.2e9
    assert report["text_moe"] > 30e9
    assert report["num_experts"] == 256
    assert report["activated_experts"] == 9


def test_hf_export_uses_official_runtime_names():
    payload = qwen38_35b_a3b_config().to_hf_dict()
    assert payload["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
    assert payload["model_type"] == "qwen3_5_moe"
    assert payload["text_config"]["num_experts"] == 256
    assert payload["vision_config"]["out_hidden_size"] == 2048


def test_tiny_config_keeps_hybrid_ratio():
    config = tiny_config()
    types = config.text_config.layer_types
    assert len(types) == 4
    assert types == ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
