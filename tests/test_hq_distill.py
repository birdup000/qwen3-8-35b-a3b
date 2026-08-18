from pathlib import Path

import torch

from qwen3_8_moe.chat import Qwen38ChatFormatter
from scripts.hq_distill import (
    LORA_TARGET_CANDIDATES,
    append_jsonl,
    existing_trace_ids,
    extract_messages,
    format_sft_row,
    looks_contaminated,
    pack_topk_logits,
    resolve_lora_targets,
    synthetic_sft_rows,
    unpack_topk_kd_loss,
)


def test_format_sft_row_masks_only_assistant():
    formatter = Qwen38ChatFormatter(enable_thinking=True, reasoning_effort="xhigh")
    row = format_sft_row(
        [
            {"role": "user", "content": "What is MoE?"},
            {"role": "assistant", "content": "<think>\nplan\n</think>\nSparse routing."},
        ],
        formatter=formatter,
    )
    assert row["text"].startswith("<|im_start|>system")
    assert row["prompt"].endswith("<|im_start|>assistant\n")
    assert row["text"].startswith(row["prompt"])
    assert "Sparse routing." in row["text"]
    assert "Sparse routing." not in row["prompt"]
    assert row["text"].count("<|im_start|>assistant") == 1


def test_synthetic_rows_and_extractors():
    rows = synthetic_sft_rows(3)
    assert len(rows) == 3
    assert rows[0]["messages"][-1]["role"] == "assistant"
    rendered = format_sft_row(rows[0]["messages"])
    assert "<think>" in rendered["text"]

    chat = extract_messages(
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        "messages",
    )
    conv = extract_messages(
        {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
        "conversations",
    )
    instr = extract_messages({"instruction": "Add", "output": "def add(a, b): return a+b"}, "instruction")
    assert chat[-1]["content"] == "Hello"
    assert conv[0]["role"] == "user"
    assert instr is not None and "return a+b" in instr[-1]["content"]


def test_resume_skips_existing_trace_ids(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    append_jsonl(path, {"id": "trace-a", "messages": []})
    append_jsonl(path, {"id": "trace-b", "messages": []})
    have = existing_trace_ids(path)
    assert have == {"trace-a", "trace-b"}
    append_jsonl(path, {"id": "trace-c", "messages": []})
    assert "trace-c" in existing_trace_ids(path)


def test_topk_pack_unpack_kd_finite():
    torch.manual_seed(0)
    teacher = torch.randn(6, 32)
    student = teacher + 0.1 * torch.randn(6, 32)
    packed = pack_topk_logits(teacher, k=8)
    assert packed["values"].shape == (6, 8)
    assert packed["indices"].dtype == torch.int32
    loss = unpack_topk_kd_loss(student.unsqueeze(0), packed, temperature=2.0)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_lora_candidates_cover_hybrid_and_moe():
    for name in (
        "q_proj",
        "in_proj_qkv",
        "out_proj",
        "gate_proj",
        "down_proj",
        "shared_expert_gate",
        "gate",
    ):
        assert name in LORA_TARGET_CANDIDATES


def test_resolve_lora_targets_skips_non_linear():
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.gate_proj = torch.nn.Linear(4, 8, bias=False)
            self.experts_down = torch.nn.Parameter(torch.zeros(2, 4, 4))

    found = resolve_lora_targets(Toy())
    assert "q_proj" in found
    assert "gate_proj" in found
    assert "experts_down" not in found


def test_contamination_filter():
    assert looks_contaminated("solve this", source="gsm8k")
    assert looks_contaminated("OpenAI HumanEval prompt")
    assert not looks_contaminated("Write a FastAPI endpoint", source="ultrachat")


def test_collect_stage_a_resumes(tmp_path: Path):
    extra = tmp_path / "extra.jsonl"
    for row in synthetic_sft_rows(4):
        append_jsonl(extra, row)
    from scripts.hq_distill import collect_stage_a_jsonl

    out = collect_stage_a_jsonl(tmp_path / "stage_a.jsonl", max_rows=2, extra_jsonl=extra)
    first = existing_trace_ids(out)
    assert len(first) == 2
    collect_stage_a_jsonl(tmp_path / "stage_a.jsonl", max_rows=2, extra_jsonl=extra)
    assert existing_trace_ids(tmp_path / "stage_a.jsonl") == first


def test_hq_progress_reads_drive_layout(tmp_path: Path):
    from scripts.hq_distill import hq_progress, save_state

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_state(adapter / "state.json", {"stage": "B", "a_step": 200, "c_step": 0})
    append_jsonl(tmp_path / "stage_a.jsonl", {"id": "row-1"})
    append_jsonl(tmp_path / "traces.jsonl", {"id": "trace-1"})
    progress = hq_progress(tmp_path, max_rows=10, max_traces=5, stage_c_steps=4000)
    assert progress["adapter"] is True
    assert progress["stage"] == "B"
    assert progress["stage_a_rows"] == 1
    assert progress["traces"] == 1
    assert progress["a_step"] == 200
