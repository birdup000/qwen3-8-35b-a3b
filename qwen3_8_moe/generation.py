"""Qwen3.8 generation helpers (thinking vs instruct presets)."""

from __future__ import annotations

from typing import Any

from .chat import Qwen38ChatFormatter, ReasoningEffort, SamplingPreset
from .modeling import Qwen38MoeForCausalLM, Qwen38MoeForConditionalGeneration


def generate_chat(
    model: Qwen38MoeForCausalLM | Qwen38MoeForConditionalGeneration,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = True,
    preserve_thinking: bool = True,
    reasoning_effort: ReasoningEffort = "xhigh",
    max_new_tokens: int = 256,
    **sample_overrides: Any,
) -> dict[str, str]:
    formatter = Qwen38ChatFormatter(
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
        reasoning_effort=reasoning_effort,
    )
    prompt = formatter.format_messages(messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    preset: SamplingPreset = formatter.sampling()
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "min_p": preset.min_p,
        "presence_penalty": preset.presence_penalty,
        "repetition_penalty": preset.repetition_penalty,
        "eos_token_id": getattr(tokenizer, "eos_token_id", model.language_model.config.eos_token_id)
        if isinstance(model, Qwen38MoeForConditionalGeneration)
        else getattr(tokenizer, "eos_token_id", model.config.eos_token_id),
    }
    kwargs.update(sample_overrides)
    output_ids = model.generate(input_ids, **kwargs)
    new_tokens = output_ids[0, input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    thinking, answer = formatter.split_thinking(text)
    return {"prompt": prompt, "raw": text, "thinking": thinking, "answer": answer}
