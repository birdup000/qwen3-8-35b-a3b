"""Qwen3.8-35B-A3B: Qwen3.8 architecture on the Qwen3.6 35B-A3B MoE runtime."""

from .chat import Qwen38ChatFormatter, ReasoningEffort, SamplingPreset
from .configuration import (
    Qwen38MoeConfig,
    Qwen38MoeTextConfig,
    Qwen38MoeVisionConfig,
    build_layer_types,
    qwen38_35b_a3b_config,
    tiny_config,
)
from .modeling import (
    Qwen38MoeForCausalLM,
    Qwen38MoeForConditionalGeneration,
    Qwen38MoeTextModel,
)
from .params import active_parameter_count, parameter_report, total_parameter_count

__version__ = "0.1.0"
__all__ = [
    "Qwen38ChatFormatter",
    "Qwen38MoeConfig",
    "Qwen38MoeForCausalLM",
    "Qwen38MoeForConditionalGeneration",
    "Qwen38MoeTextConfig",
    "Qwen38MoeTextModel",
    "Qwen38MoeVisionConfig",
    "ReasoningEffort",
    "SamplingPreset",
    "active_parameter_count",
    "build_layer_types",
    "parameter_report",
    "qwen38_35b_a3b_config",
    "tiny_config",
    "total_parameter_count",
]
