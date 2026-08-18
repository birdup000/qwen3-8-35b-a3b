"""Qwen3.8 chat surface: thinking, preserve_thinking, reasoning_effort."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

ReasoningEffort = Literal["xhigh", "medium", "low"]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

EFFORT_HINTS: dict[str, str] = {
    "xhigh": "",
    "medium": "Keep the analysis focused. Prefer a medium-length reasoning trace.",
    "low": "Reason briefly. Prefer a short analysis and move to the answer quickly.",
}


@dataclass(frozen=True)
class SamplingPreset:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float

    @classmethod
    def thinking(cls) -> "SamplingPreset":
        return cls(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)

    @classmethod
    def instruct(cls) -> "SamplingPreset":
        return cls(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0)

    @classmethod
    def for_mode(cls, enable_thinking: bool) -> "SamplingPreset":
        return cls.thinking() if enable_thinking else cls.instruct()


@dataclass
class ChatMessage:
    role: str
    content: str


class Qwen38ChatFormatter:
    """Matches the Qwen3.8-27B request surface on the 35B-A3B runtime."""

    def __init__(
        self,
        enable_thinking: bool = True,
        preserve_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        default_system: str = "You are Qwen3.8-35B-A3B, a helpful multimodal assistant.",
    ) -> None:
        if reasoning_effort not in EFFORT_HINTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(EFFORT_HINTS)}")
        self.enable_thinking = enable_thinking
        self.preserve_thinking = preserve_thinking
        self.reasoning_effort = reasoning_effort
        self.default_system = default_system

    def sampling(self) -> SamplingPreset:
        return SamplingPreset.for_mode(self.enable_thinking)

    def format_messages(self, messages: Iterable[dict[str, str] | ChatMessage], add_generation_prompt: bool = True) -> str:
        normalized: list[ChatMessage] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                normalized.append(message)
            else:
                normalized.append(ChatMessage(role=str(message["role"]), content=str(message.get("content", ""))))
        if not normalized or normalized[0].role != "system":
            normalized = [ChatMessage(role="system", content=self._system_text(None))] + normalized
        else:
            normalized[0] = ChatMessage(role="system", content=self._system_text(normalized[0].content))

        chunks: list[str] = []
        for message in normalized:
            content = message.content
            if message.role == "assistant" and not self.preserve_thinking:
                content = THINK_RE.sub("", content).strip()
            chunks.append(f"{IM_START}{message.role}\n{content}{IM_END}\n")
        if add_generation_prompt:
            prefix = f"{IM_START}assistant\n"
            if self.enable_thinking:
                prefix += f"{THINK_OPEN}\n"
            chunks.append(prefix)
        return "".join(chunks)

    def _system_text(self, user_system: str | None) -> str:
        parts = [user_system.strip() if user_system else self.default_system]
        hint = EFFORT_HINTS[self.reasoning_effort]
        if hint:
            parts.append(hint)
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def split_thinking(text: str) -> tuple[str, str]:
        match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
        if not match:
            if text.startswith(THINK_OPEN):
                return text[len(THINK_OPEN) :].lstrip("\n"), ""
            return "", text
        return match.group(1).strip(), match.group(2).lstrip("\n")
