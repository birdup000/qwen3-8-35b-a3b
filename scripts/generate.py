#!/usr/bin/env python3
"""Generate with the Qwen3.8 thinking / reasoning_effort API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_8_moe import Qwen38MoeForCausalLM, tiny_config
from qwen3_8_moe.chat import Qwen38ChatFormatter
from qwen3_8_moe.generation import generate_chat


class DummyTokenizer:
    """Byte-fallback tokenizer for architecture smoke tests (not 248k vocab)."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = [(ord(ch) % (self.vocab_size - 1)) + 1 for ch in text[:64]] or [1]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(chr(33 + (int(i) % 94)) for i in values)


def load_model(path: str | None, device: str):
    if path:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(path, device_map=device, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            return model, tokenizer
        except Exception as exc:
            raise SystemExit(f"Failed to load {path}: {exc}") from exc
    config = tiny_config()
    model = Qwen38MoeForCausalLM(config.text_config).to(device)
    return model, DummyTokenizer(config.text_config.vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF snapshot. Default: random tiny model")
    parser.add_argument("--prompt", default="Explain mixture-of-experts in one paragraph.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"), default="xhigh")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    formatter = Qwen38ChatFormatter(
        enable_thinking=args.enable_thinking,
        preserve_thinking=args.preserve_thinking,
        reasoning_effort=args.reasoning_effort,
    )
    messages = [{"role": "user", "content": args.prompt}]
    if args.model is None:
        prompt = formatter.format_messages(messages)
        print(prompt)
        print("--- tiny random model (architecture smoke test) ---")
    result = generate_chat(
        model if not hasattr(model, "generate") else model,
        tokenizer,
        messages,
        enable_thinking=args.enable_thinking,
        preserve_thinking=args.preserve_thinking,
        reasoning_effort=args.reasoning_effort,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({key: result[key] for key in ("thinking", "answer", "raw")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
