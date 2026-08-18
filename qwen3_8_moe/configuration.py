"""Architecture configs for Qwen3.8-35B-A3B.

The text/vision topology matches Qwen3.6-35B-A3B so the checkpoint serves
through the official `qwen3_5_moe` stack (Transformers 5.x, vLLM, SGLang).
The intelligence surface matches Qwen3.8-27B: hybrid 3:1 Gated DeltaNet /
Gated Attention, fused attention output gate, MTP, vision, and
`reasoning_effort` / `preserve_thinking`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable


def build_layer_types(num_hidden_layers: int, full_attention_interval: int = 4) -> list[str]:
    """Official 3:1 layout: three Gated DeltaNet layers, then one gated attention."""
    return [
        "linear_attention" if (i + 1) % full_attention_interval else "full_attention"
        for i in range(num_hidden_layers)
    ]


@dataclass
class Qwen38MoeTextConfig:
    """Language-tower config. Defaults are Qwen3.6-35B-A3B / Qwen3.8-35B-A3B."""

    model_type: str = "qwen3_5_moe_text"
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    full_attention_interval: int = 4
    layer_types: list[str] | None = None
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 8
    num_experts: int = 256
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    pad_token_id: int | None = None
    bos_token_id: int = 248044
    eos_token_id: int = 248044
    partial_rotary_factor: float = 0.25
    mamba_ssm_dtype: str = "float32"
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    rope_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default",
        }
    )

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = build_layer_types(self.num_hidden_layers, self.full_attention_interval)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        if self.hidden_size % self.num_attention_heads != 0 and self.head_dim <= 0:
            raise ValueError("head_dim or hidden_size/num_attention_heads must be valid")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rope_parameters.get("partial_rotary_factor", self.partial_rotary_factor))

    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer_types"] = list(self.layer_types or [])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen38MoeTextConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class Qwen38MoeVisionConfig:
    """Vision tower. Defaults match Qwen3.6-35B-A3B (`out_hidden_size` = text hidden)."""

    model_type: str = "qwen3_5_moe_vision"
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    deepstack_visual_indexes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen38MoeVisionConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class Qwen38MoeConfig:
    """Top-level multimodal config, HF-compatible with `qwen3_5_moe` serving."""

    model_type: str = "qwen3_5_moe"
    architectures: list[str] = field(default_factory=lambda: ["Qwen3_5MoeForConditionalGeneration"])
    text_config: Qwen38MoeTextConfig = field(default_factory=Qwen38MoeTextConfig)
    vision_config: Qwen38MoeVisionConfig = field(default_factory=Qwen38MoeVisionConfig)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False
    # Qwen3.8 intelligence surface (chat / sampling), not a compute-graph change.
    default_reasoning_effort: str = "xhigh"
    enable_thinking: bool = True
    preserve_thinking: bool = True
    native_context_length: int = 262144
    yarn_context_length: int = 1_000_000

    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = Qwen38MoeTextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen38MoeVisionConfig.from_dict(self.vision_config)
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            self.vision_config.out_hidden_size = self.text_config.hidden_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "architectures": list(self.architectures),
            "model_type": self.model_type,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "default_reasoning_effort": self.default_reasoning_effort,
            "enable_thinking": self.enable_thinking,
            "preserve_thinking": self.preserve_thinking,
            "native_context_length": self.native_context_length,
            "yarn_context_length": self.yarn_context_length,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
        }

    def to_hf_dict(self) -> dict[str, Any]:
        """Config accepted by official Qwen3.6 / `qwen3_5_moe` loaders."""
        payload = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
        }
        payload["text_config"]["model_type"] = "qwen3_5_moe_text"
        payload["vision_config"]["model_type"] = "qwen3_5_moe_vision"
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen38MoeConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in allowed}
        return cls(**kwargs)


def qwen38_35b_a3b_config() -> Qwen38MoeConfig:
    """Production Qwen3.8-35B-A3B configuration."""
    return Qwen38MoeConfig()


def tiny_config(
    vocab_size: int = 128,
    hidden_size: int = 64,
    num_hidden_layers: int = 4,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
) -> Qwen38MoeConfig:
    """CPU-sized clone of the 3:1 MoE layout for tests."""
    text = Qwen38MoeTextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        bos_token_id=0,
        eos_token_id=0,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 2, 0],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000.0,
            "rope_type": "default",
        },
    )
    vision = Qwen38MoeVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=hidden_size,
        num_position_embeddings=64,
    )
    return Qwen38MoeConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=2,
        video_token_id=3,
        vision_start_token_id=4,
        vision_end_token_id=5,
    )


def yarn_rope_parameters(factor: float = 4.0, original_max: int = 262144) -> dict[str, Any]:
    """Qwen3.8 YaRN override used to extend context to ~1M tokens."""
    return {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "yarn",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
        "factor": factor,
        "original_max_position_embeddings": original_max,
    }


def iter_full_attention_indices(layer_types: Iterable[str]) -> list[int]:
    return [index for index, kind in enumerate(layer_types) if kind == "full_attention"]


def iter_linear_attention_indices(layer_types: Iterable[str]) -> list[int]:
    return [index for index, kind in enumerate(layer_types) if kind == "linear_attention"]
