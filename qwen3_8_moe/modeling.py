"""Qwen3.8-35B-A3B reference model.

The compute graph is the official Qwen3.6-35B-A3B / `qwen3_5_moe` runtime:
40 hybrid layers, 256 experts, 8 routed + 1 shared, gated attention + Gated
DeltaNet. The intelligence surface (tokenizer, thinking, MTP, vision tokens)
matches Qwen3.8-27B.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .cache import HybridCache
from .configuration import Qwen38MoeConfig, Qwen38MoeTextConfig
from .layers import DecoderLayer, RMSNorm, TextRotaryEmbedding, make_causal_mask
from .vision import Qwen38MoeVisionModel


@dataclass
class ModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: HybridCache | None = None
    router_logits: tuple[torch.Tensor, ...] | None = None


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    past_key_values: HybridCache | None = None
    hidden_states: torch.Tensor | None = None
    mtp_logits: torch.Tensor | None = None


class Qwen38MoeTextModel(nn.Module):
    def __init__(self, config: Qwen38MoeTextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HybridCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool = False,
    ) -> ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = HybridCache(self.config)

        batch, seq_len, _ = inputs_embeds.shape
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen
            position_ids = position_ids.view(1, 1, -1).expand(3, batch, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = make_causal_mask(batch, seq_len, seq_len + past_seen, inputs_embeds.device, inputs_embeds.dtype)
        if attention_mask is not None and attention_mask.dim() == 2:
            pad = (1.0 - attention_mask[:, None, None, :].to(causal_mask.dtype)) * torch.finfo(causal_mask.dtype).min
            if pad.shape[-1] == causal_mask.shape[-1]:
                causal_mask = causal_mask + pad

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        router_logits: list[torch.Tensor] = []
        for layer, layer_type in zip(self.layers, self.config.layer_types):
            layer_mask: torch.Tensor | None
            if layer_type == "linear_attention":
                layer_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
            else:
                layer_mask = causal_mask
            hidden_states, layer_router = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
            )
            if output_router_logits:
                router_logits.append(layer_router)

        if past_key_values is not None:
            past_key_values.seen_tokens = past_seen + seq_len
        return ModelOutput(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values,
            router_logits=tuple(router_logits) if output_router_logits else None,
        )


class MultiTokenPredictionHead(nn.Module):
    """One-step MTP head used by Qwen3.6/3.8 (`mtp_num_hidden_layers=1`)."""

    def __init__(self, config: Qwen38MoeTextConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(config, config.num_hidden_layers - 1) for _ in range(max(config.mtp_num_hidden_layers, 1))]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fused = self.hidden_proj(torch.cat([hidden_states, next_embeds], dim=-1))
        for layer in self.layers:
            fused, _ = layer(fused, position_embeddings=position_embeddings)
        return self.norm(fused)


class Qwen38MoeForCausalLM(nn.Module):
    def __init__(self, config: Qwen38MoeTextConfig | Qwen38MoeConfig) -> None:
        super().__init__()
        if isinstance(config, Qwen38MoeConfig):
            config = config.text_config
        self.config = config
        self.model = Qwen38MoeTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if not config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        self.mtp = MultiTokenPredictionHead(config) if config.mtp_num_hidden_layers else None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HybridCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int = 0,
        compute_mtp: bool = False,
    ) -> CausalLMOutput:
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        mtp_logits = None
        if compute_mtp and self.mtp is not None and input_ids is not None and hidden_states.size(1) > 1:
            next_embeds = self.model.embed_tokens(input_ids[:, 1:])
            mtp_hidden = self.mtp(
                hidden_states[:, :-1],
                next_embeds,
                self.model.rotary_emb(hidden_states[:, :-1], torch.arange(hidden_states.size(1) - 1, device=hidden_states.device).view(1, 1, -1).expand(3, hidden_states.size(0), -1)),
            )
            mtp_logits = self.lm_head(mtp_hidden)

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            mtp_logits=mtp_logits,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        eos_token_id: int | None = None,
    ) -> torch.LongTensor:
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        generated = input_ids
        cache: HybridCache | None = None
        seen: set[int] = set(int(t) for t in input_ids[0].tolist())
        for _ in range(max_new_tokens):
            step_ids = generated if cache is None else generated[:, -1:]
            outputs = self.forward(input_ids=step_ids, attention_mask=attention_mask if cache is None else None, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            if repetition_penalty != 1.0:
                for token_id in seen:
                    logits[:, token_id] /= repetition_penalty
            if presence_penalty != 0.0:
                for token_id in seen:
                    logits[:, token_id] -= presence_penalty
            next_token = _sample_next_token(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
            generated = torch.cat([generated, next_token], dim=-1)
            seen.add(int(next_token.item()))
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        return generated


class Qwen38MoeForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen38MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen38MoeVisionModel(config.vision_config)
        self.language_model = Qwen38MoeForCausalLM(config.text_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HybridCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int = 0,
    ) -> CausalLMOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            if pixel_values is not None and image_grid_thw is not None:
                image_embeds = self.visual(pixel_values, image_grid_thw)
                image_mask = input_ids == self.config.image_token_id
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.dtype)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, **kwargs) -> torch.LongTensor:
        return self.language_model.generate(input_ids=input_ids, **kwargs)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> torch.LongTensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, -1, None], torch.finfo(logits.dtype).min)
    if min_p > 0:
        probs = torch.softmax(logits, dim=-1)
        logits = logits.masked_fill(probs < min_p * probs.max(dim=-1, keepdim=True).values, torch.finfo(logits.dtype).min)
    if 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cdf = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cdf > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, torch.finfo(logits.dtype).min)
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min).scatter(1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
