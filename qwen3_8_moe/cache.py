"""Hybrid cache: KV for gated attention, conv + recurrent state for DeltaNet."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .configuration import Qwen38MoeTextConfig


@dataclass
class LayerCache:
    conv_states: torch.Tensor | None = None
    recurrent_states: torch.Tensor | None = None
    key_states: torch.Tensor | None = None
    value_states: torch.Tensor | None = None


@dataclass
class HybridCache:
    config: Qwen38MoeTextConfig
    layers: list[LayerCache] = field(default_factory=list)
    seen_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = [LayerCache() for _ in range(self.config.num_hidden_layers)]

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        if layer_idx is None:
            return any(layer.conv_states is not None or layer.recurrent_states is not None or layer.key_states is not None for layer in self.layers)
        layer = self.layers[layer_idx]
        return layer.conv_states is not None or layer.recurrent_states is not None or layer.key_states is not None

    def update_conv_state(self, state: torch.Tensor, layer_idx: int) -> None:
        self.layers[layer_idx].conv_states = state

    def update_recurrent_state(self, state: torch.Tensor | None, layer_idx: int) -> None:
        self.layers[layer_idx].recurrent_states = state

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        if layer.key_states is None:
            layer.key_states = key_states
            layer.value_states = value_states
        else:
            layer.key_states = torch.cat([layer.key_states, key_states], dim=2)
            layer.value_states = torch.cat([layer.value_states, value_states], dim=2)
        return layer.key_states, layer.value_states
