#!/usr/bin/env python3
"""High-quality Colab distill: 3.8 thinking on the 35B-A3B runtime.

Stage A: SFT on an open instruct/reasoning mix (no teacher GPU).
Stage B: Qwen3.8-27B generates thinking traces to jsonl (resumable).
Stage C: response-only CE + compact top-k KL on those traces.

This will not match 27B dense quality on a 40GB A100. It is the strongest
Drive-resumable recipe that still decodes as qwen3_5_moe (~3B active).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen3_8_moe.chat import Qwen38ChatFormatter

DEFAULT_TEACHER = "Qwen/Qwen3.8-27B"
DEFAULT_STUDENT = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_SEQ_LEN = 2048
FALLBACK_SEQ_LEN = 1024
DEFAULT_TOP_K = 128
CHECKPOINT_EVERY = 200

# Leaf Linear names. PEFT matches these anywhere in the graph. 3D expert
# Parameter tensors (gate_up_proj / Experts.down_proj) are not Linear and
# are skipped — shared expert + router + attn + DeltaNet still train.
LORA_TARGET_CANDIDATES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert_gate",
    "gate",
)

# Streamed, permissive-leaning public mixes. Skip GSM8K/MATH/HumanEval dumps.
# license comments are for operators, not a grant of rights.
STAGE_A_SOURCES: tuple[dict[str, str], ...] = (
    {
        "id": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "kind": "messages",
        "license": "MIT",
    },
    {
        "id": "Open-Orca/SlimOrca",
        "split": "train",
        "kind": "conversations",
        "license": "MIT",
    },
    {
        "id": "m-a-p/CodeFeedback-Filtered-Instruction",
        "split": "train",
        "kind": "instruction",
        "license": "Apache-2.0",
    },
)

CONTAMINATION_MARKERS = (
    "humaneval",
    "openai_humaneval",
    "gsm8k",
    "competition_math",
    "hendrycks_math",
    "mbpp test",
)


def default_formatter() -> Qwen38ChatFormatter:
    return Qwen38ChatFormatter(enable_thinking=True, preserve_thinking=True, reasoning_effort="xhigh")


def format_sft_row(
    messages: list[dict[str, str]],
    *,
    formatter: Qwen38ChatFormatter | None = None,
) -> dict[str, str]:
    """Render a complete conversation and the prompt prefix for response-only loss."""
    formatter = formatter or default_formatter()
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT row must end with an assistant message")
    text = formatter.format_messages(messages, add_generation_prompt=False)
    marker = "<|im_start|>assistant\n"
    idx = text.rfind(marker)
    if idx < 0:
        raise ValueError("Rendered SFT text has no assistant turn")
    prompt = text[: idx + len(marker)]
    return {"text": text, "prompt": prompt}


def synthetic_sft_rows(count: int = 8) -> list[dict[str, Any]]:
    from scripts.colab_pipeline import SAMPLE_TEXTS

    rows: list[dict[str, Any]] = []
    for index, user in enumerate(SAMPLE_TEXTS[:count]):
        assistant = (
            f"<think>\nBreak the request into steps and answer clearly. Item {index}.\n</think>\n"
            f"Here is a direct answer for: {user[:80]}"
        )
        rows.append(
            {
                "id": f"synthetic-{index:04d}",
                "source": "synthetic",
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    return rows


def looks_contaminated(text: str, source: str = "") -> bool:
    blob = f"{source}\n{text}".lower()
    return any(marker in blob for marker in CONTAMINATION_MARKERS)


def extract_messages(row: dict[str, Any], kind: str) -> list[dict[str, str]] | None:
    if kind == "messages":
        raw = row.get("messages") or row.get("conversation")
    elif kind == "conversations":
        raw = row.get("conversations") or row.get("conversation")
    elif kind == "instruction":
        instruction = (row.get("instruction") or row.get("query") or "").strip()
        output = (row.get("output") or row.get("response") or row.get("answer") or "").strip()
        if not instruction or not output:
            return None
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    else:
        raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("from") or "").lower()
        content = str(item.get("content") or item.get("value") or "").strip()
        if role in {"human", "user", "prompt"}:
            role = "user"
        elif role in {"gpt", "assistant", "model"}:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        if content:
            messages.append({"role": role, "content": content})
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    return messages


def existing_trace_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.is_file():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                ids.add(str(row["id"]))
    return ids


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def pack_topk_logits(logits: torch.Tensor, k: int = DEFAULT_TOP_K) -> dict[str, torch.Tensor]:
    """Compact teacher distribution: [seq, vocab] -> top-k values + indices."""
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    k = min(k, logits.size(-1))
    values, indices = torch.topk(logits.float(), k, dim=-1)
    return {
        "values": values.half().contiguous(),
        "indices": indices.to(torch.int32).contiguous(),
        "k": torch.tensor(k, dtype=torch.int32),
    }


def unpack_topk_kd_loss(
    student_logits: torch.Tensor,
    packed: dict[str, torch.Tensor],
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(teacher_topk || student) using only the stored teacher mass."""
    values = packed["values"].to(device=student_logits.device, dtype=torch.float32)
    indices = packed["indices"].to(device=student_logits.device, dtype=torch.long)
    if student_logits.dim() == 3:
        student_logits = student_logits[0]
    seq = min(student_logits.size(0), values.size(0))
    student = student_logits[:seq]
    values = values[:seq]
    indices = indices[:seq]
    t = temperature
    teacher = torch.softmax(values / t, dim=-1)
    student_sel = student.gather(-1, indices)
    student_log = F.log_softmax(student_sel.float() / t, dim=-1)
    return F.kl_div(student_log, teacher, reduction="batchmean") * (t * t)


def save_topk(path: Path, packed: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in packed.items()}, path)


def load_topk(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_lora_linear(module: torch.nn.Module) -> bool:
    if isinstance(module, torch.nn.Linear):
        return True
    return type(module).__name__.lower() in {"linear4bit", "linear8bitlt", "linear8bit"}


def _weight_is_meta(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return weight is not None and getattr(weight, "device", None) is not None and weight.device.type == "meta"


def resolve_lora_targets(model: torch.nn.Module, candidates: Iterable[str] = LORA_TARGET_CANDIDATES) -> list[str]:
    wanted = set(candidates)
    found: list[str] = []
    skipped_3d: list[str] = []
    skipped_meta: list[str] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in wanted:
            continue
        if not _is_lora_linear(module):
            skipped_3d.append(name)
            continue
        if _weight_is_meta(module):
            skipped_meta.append(name)
            continue
        found.append(name)
    if skipped_3d:
        print(f"LoRA skipped non-Linear modules ({len(skipped_3d)}), e.g. {skipped_3d[:3]}")
    if skipped_meta:
        print(f"LoRA skipped meta-weight modules ({len(skipped_meta)}), e.g. {skipped_meta[:3]}")
    if not found:
        found = [name for name in LORA_TARGET_CANDIDATES if name in {"q_proj", "k_proj", "v_proj", "o_proj"}]
        print("LoRA: no candidate Linear names matched; falling back to attention projections")
        print(f"LoRA targets: {found}")
        return found
    print(f"LoRA targets: {len(found)} modules  e.g. {found[:4]}")
    return found


def tokenize_sft(
    tokenizer,
    row: dict[str, Any],
    seq_len: int,
    formatter: Qwen38ChatFormatter | None = None,
) -> dict[str, torch.Tensor] | None:
    rendered = format_sft_row(row["messages"], formatter=formatter)
    full = tokenizer(rendered["text"], truncation=True, max_length=seq_len, add_special_tokens=False)
    prompt = tokenizer(rendered["prompt"], truncation=True, max_length=seq_len, add_special_tokens=False)
    input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
    if input_ids.numel() < 16:
        return None
    prompt_len = min(len(prompt["input_ids"]), input_ids.numel() - 1)
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    if (labels != -100).sum() < 4:
        return None
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def pack_batch(examples: list[dict[str, torch.Tensor]], seq_len: int) -> dict[str, torch.Tensor]:
    """Left-to-right pack short examples into one seq_len sequence."""
    ids: list[int] = []
    labels: list[int] = []
    for example in examples:
        piece_ids = example["input_ids"].tolist()
        piece_labels = example["labels"].tolist()
        if len(ids) + len(piece_ids) > seq_len:
            break
        ids.extend(piece_ids)
        labels.extend(piece_labels)
    if len(ids) < 16:
        example = examples[0]
        ids = example["input_ids"][:seq_len].tolist()
        labels = example["labels"][:seq_len].tolist()
    pad = seq_len - len(ids)
    if pad > 0:
        ids = ids + [0] * pad
        labels = labels + [-100] * pad
        mask = [1] * (seq_len - pad) + [0] * pad
    else:
        ids = ids[:seq_len]
        labels = labels[:seq_len]
        mask = [1] * seq_len
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(mask, dtype=torch.long).unsqueeze(0),
        "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
    }


def load_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"step": 0}


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_state(path)
    current.update(payload)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def hq_progress(
    work_dir: Path,
    *,
    max_rows: int = 10000,
    max_traces: int = 5000,
    stage_c_steps: int = 4000,
) -> dict[str, Any]:
    """Drive-resume snapshot for Colab: rows, traces, adapter, last stage."""
    work_dir = Path(work_dir)
    adapter = work_dir / "adapter"
    state = load_state(adapter / "state.json")
    rows = existing_trace_ids(work_dir / "stage_a.jsonl")
    traces = existing_trace_ids(work_dir / "traces.jsonl")
    return {
        "dir": str(work_dir),
        "stage": state.get("stage"),
        "adapter": (adapter / "adapter_config.json").is_file(),
        "stage_a_rows": len(rows),
        "stage_a_target": max_rows,
        "a_step": int(state.get("a_step", 0)),
        "traces": len(traces),
        "traces_target": max_traces,
        "c_step": int(state.get("c_step", 0)),
        "c_target": stage_c_steps,
    }


def iter_stage_a_rows(max_rows: int, extra_jsonl: Path | None = None) -> Iterator[dict[str, Any]]:
    emitted = 0
    if extra_jsonl is not None and extra_jsonl.is_file():
        for row in iter_jsonl(extra_jsonl):
            yield row
            emitted += 1
            if emitted >= max_rows:
                return
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; using synthetic SFT rows")
        for row in synthetic_sft_rows(min(max_rows, 16)):
            yield row
            emitted += 1
            if emitted >= max_rows:
                return
        return

    per_source = max((max_rows + len(STAGE_A_SOURCES) - 1) // max(len(STAGE_A_SOURCES), 1), 1)
    for spec in STAGE_A_SOURCES:
        if emitted >= max_rows:
            return
        try:
            stream = load_dataset(spec["id"], split=spec["split"], streaming=True)
        except Exception as exc:
            print(f"Skipping {spec['id']}: {exc}")
            continue
        taken = 0
        for raw in stream:
            if emitted >= max_rows or taken >= per_source:
                break
            messages = extract_messages(raw, spec["kind"])
            if messages is None:
                continue
            user_blob = " ".join(m["content"] for m in messages if m["role"] == "user")
            if looks_contaminated(user_blob, spec["id"]):
                continue
            yield {
                "id": f"{spec['id']}:{taken}",
                "source": spec["id"],
                "license": spec["license"],
                "messages": messages,
            }
            taken += 1
            emitted += 1
    if emitted == 0:
        print("No streamed rows; falling back to synthetic SFT")
        yield from synthetic_sft_rows(min(max_rows, 16))


def collect_stage_a_jsonl(path: Path, max_rows: int, extra_jsonl: Path | None = None) -> Path:
    """Write (or resume) Stage A rows. Existing ids are kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    have = existing_trace_ids(path)
    if len(have) >= max_rows or (max_rows >= 1000 and len(have) >= max_rows - 3):
        print(f"Reusing {len(have)} Stage A rows in {path}")
        return path
    for row in iter_stage_a_rows(max_rows, extra_jsonl=extra_jsonl):
        if row["id"] in have:
            continue
        append_jsonl(path, row)
        have.add(row["id"])
        if len(have) % 200 == 0:
            print(f"Stage A rows {len(have)}/{max_rows}")
        if len(have) >= max_rows:
            break
    print(f"Stage A jsonl {path}  n={len(have)}")
    return path


def _enable_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def attach_hq_lora(model, r: int = 32, alpha: int = 64):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    from scripts.colab_pipeline import disable_generation_cache

    disable_generation_cache(model)
    has_cpu = any(str(device) in {"cpu", "disk"} for device in (getattr(model, "hf_device_map", None) or {}).values())
    if has_cpu:
        print("Skipping prepare_model_for_kbit_training (CPU expert banks; PEFT would serialize 4-bit meta state)")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        _enable_checkpointing(model)
    else:
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except RuntimeError as exc:
            if "meta" not in str(exc).lower():
                raise
            print("prepare_model_for_kbit_training hit meta tensors; enabling input grads only")
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        else:
            _enable_checkpointing(model)
    targets = resolve_lora_targets(model)
    try:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.05,
                target_modules=targets,
            ),
        )
    except RuntimeError as exc:
        if "meta" not in str(exc).lower():
            raise
        print("get_peft_model hit meta tensors; retrying with attention projections only")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
    model.print_trainable_parameters()
    return model


def _maybe_load_adapter(model, adapter_dir: Path):
    if not (adapter_dir / "adapter_config.json").is_file():
        return model
    from peft import PeftModel

    print(f"Resuming adapter {adapter_dir}")
    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)


def _save_adapter(model, adapter_dir: Path, extra: dict[str, Any]) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    save_state(adapter_dir / "TRAIN.json", extra)


def _module_device(model) -> torch.device:
    embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embed is not None and getattr(embed, "weight", None) is not None:
        return embed.weight.device
    return next(p for p in model.parameters() if p.device.type != "meta").device


def stage_a_sft(
    *,
    student_id: str,
    work_dir: Path,
    max_rows: int = 10000,
    steps: int | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    lr: float = 1e-4,
    lora_r: int = 32,
    fourbit: bool = True,
    extra_jsonl: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    from transformers import AutoTokenizer

    from scripts.colab_pipeline import free_model, load_causal_lm, overlay_qwen38_configs

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows_path = collect_stage_a_jsonl(work_dir / "stage_a.jsonl", max_rows, extra_jsonl=extra_jsonl)
    adapter_dir = work_dir / "adapter"
    overlay_qwen38_configs(adapter_dir, repo_root)
    state_path = adapter_dir / "state.json"
    state = load_state(state_path)
    start_step = int(state.get("a_step", 0))

    tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatter = default_formatter()
    rows = [row for row in iter_jsonl(rows_path) if row.get("messages")]
    if not rows:
        raise RuntimeError(f"No Stage A rows in {rows_path}")
    tokenized: list[dict[str, torch.Tensor]] = []
    for row in rows:
        item = tokenize_sft(tokenizer, row, seq_len, formatter)
        if item is not None:
            tokenized.append(item)
    if not tokenized:
        if seq_len > FALLBACK_SEQ_LEN:
            print(f"No rows fit seq_len={seq_len}; retrying {FALLBACK_SEQ_LEN}")
            return stage_a_sft(
                student_id=student_id,
                work_dir=work_dir,
                max_rows=max_rows,
                steps=steps,
                seq_len=FALLBACK_SEQ_LEN,
                lr=lr,
                lora_r=lora_r,
                fourbit=fourbit,
                extra_jsonl=extra_jsonl,
                repo_root=repo_root,
            )
        raise RuntimeError("No tokenized Stage A examples")

    n_steps = steps if steps is not None else max(len(tokenized), 1)
    if start_step >= n_steps and (adapter_dir / "adapter_config.json").is_file():
        print(f"Stage A already complete at step {start_step}; skipping train")
        return {"adapter": adapter_dir, "rows": rows_path}

    student = load_causal_lm(student_id, fourbit=fourbit)
    try:
        if (adapter_dir / "adapter_config.json").is_file():
            student = _maybe_load_adapter(student, adapter_dir)
            _enable_checkpointing(student)
        else:
            student = attach_hq_lora(student, r=lora_r)
        student.train()
        device = _module_device(student)
        optimizer = torch.optim.AdamW(
            (p for p in student.parameters() if p.requires_grad and p.device.type != "meta"),
            lr=lr,
        )

        step = start_step
        while step < n_steps:
            example = tokenized[step % len(tokenized)]
            batch = pack_batch([example, tokenized[(step + 1) % len(tokenized)]], seq_len)
            batch = {key: value.to(device) for key, value in batch.items()}
            out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.device.type != "meta"), 1.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0:
                print(f"stage A step {step}/{n_steps}  ce={loss.item():.4f}")
            if step % CHECKPOINT_EVERY == 0 or step == n_steps:
                _save_adapter(student, adapter_dir, {"stage": "A", "a_step": step, "seq_len": seq_len})
                save_state(state_path, {"a_step": step, "step": step, "stage": "A"})
                print(f"Checkpointed adapter at step {step}")
    except torch.cuda.OutOfMemoryError:
        free_model(student)
        if seq_len <= FALLBACK_SEQ_LEN:
            raise
        print(f"OOM during Stage A at seq_len={seq_len}; retrying {FALLBACK_SEQ_LEN}")
        return stage_a_sft(
            student_id=student_id,
            work_dir=work_dir,
            max_rows=max_rows,
            steps=steps,
            seq_len=FALLBACK_SEQ_LEN,
            lr=lr,
            lora_r=lora_r,
            fourbit=fourbit,
            extra_jsonl=extra_jsonl,
            repo_root=repo_root,
        )

    free_model(student)
    return {"adapter": adapter_dir, "rows": rows_path}


def _prompt_pool_from_stage_a(rows_path: Path, limit: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    if rows_path.is_file():
        for row in iter_jsonl(rows_path):
            messages = [m for m in row.get("messages", []) if m["role"] != "assistant"]
            if not messages:
                continue
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            if not user:
                continue
            prompts.append({"id": f"trace-{row['id']}", "messages": messages, "user": user})
            if len(prompts) >= limit:
                break
    if len(prompts) < limit:
        for row in synthetic_sft_rows(limit):
            user_msgs = [m for m in row["messages"] if m["role"] == "user"]
            prompts.append({"id": f"trace-{row['id']}", "messages": user_msgs, "user": user_msgs[0]["content"]})
            if len(prompts) >= limit:
                break
    return prompts[:limit]


def stage_b_traces(
    *,
    teacher_id: str,
    work_dir: Path,
    max_traces: int = 5000,
    max_new_tokens: int = 1024,
    seq_len: int = DEFAULT_SEQ_LEN,
    top_k: int = DEFAULT_TOP_K,
    fourbit: bool = True,
) -> dict[str, Path]:
    from transformers import AutoTokenizer

    from scripts.colab_pipeline import free_model, load_causal_lm

    work_dir = Path(work_dir)
    traces_path = work_dir / "traces.jsonl"
    topk_dir = work_dir / "topk"
    have = existing_trace_ids(traces_path)
    if len(have) >= max_traces:
        print(f"Reusing {len(have)} traces in {traces_path}")
        return {"traces": traces_path, "topk": topk_dir}

    formatter = default_formatter()
    sampling = formatter.sampling()
    prompts = _prompt_pool_from_stage_a(work_dir / "stage_a.jsonl", max_traces)
    tokenizer = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = load_causal_lm(teacher_id, fourbit=fourbit)
    teacher.eval()
    device = _module_device(teacher)

    for prompt_row in prompts:
        if prompt_row["id"] in have:
            continue
        if len(have) >= max_traces:
            break
        prompt_text = formatter.format_messages(prompt_row["messages"], add_generation_prompt=True)
        encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=seq_len)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        try:
            with torch.no_grad():
                generated = teacher.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            print("OOM during generate; cutting max_new_tokens in half")
            max_new_tokens = max(256, max_new_tokens // 2)
            torch.cuda.empty_cache()
            continue
        completion_ids = generated[0, encoded["input_ids"].size(-1) :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
        if not completion:
            continue
        if "<think>" not in completion:
            completion = f"<think>\n{completion}"
        messages = list(prompt_row["messages"]) + [{"role": "assistant", "content": completion}]
        full_text = formatter.format_messages(messages, add_generation_prompt=False)
        full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=seq_len)
        full_enc = {key: value.to(device) for key, value in full_enc.items()}
        with torch.no_grad():
            logits = teacher(**full_enc).logits[0].float().cpu()
        packed = pack_topk_logits(logits, k=top_k)
        save_topk(topk_dir / f"{prompt_row['id']}.pt", packed)
        append_jsonl(
            traces_path,
            {
                "id": prompt_row["id"],
                "messages": messages,
                "completion_tokens": int(completion_ids.numel()),
                "teacher": teacher_id,
            },
        )
        have.add(prompt_row["id"])
        print(f"trace {len(have)}/{max_traces}  new_tokens={completion_ids.numel()}")

    free_model(teacher)
    return {"traces": traces_path, "topk": topk_dir}


def stage_c_align(
    *,
    student_id: str,
    work_dir: Path,
    steps: int = 4000,
    seq_len: int = DEFAULT_SEQ_LEN,
    lr: float = 5e-5,
    temperature: float = 2.0,
    kd_weight: float = 0.5,
    stage_a_mix: float = 0.3,
    lora_r: int = 32,
    fourbit: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    from transformers import AutoTokenizer

    from scripts.colab_pipeline import free_model, load_causal_lm, overlay_qwen38_configs

    work_dir = Path(work_dir)
    traces_path = work_dir / "traces.jsonl"
    topk_dir = work_dir / "topk"
    rows_path = work_dir / "stage_a.jsonl"
    adapter_dir = work_dir / "adapter"
    overlay_qwen38_configs(adapter_dir, repo_root)
    if not traces_path.is_file():
        raise FileNotFoundError(f"Run Stage B first: missing {traces_path}")

    tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatter = default_formatter()
    traces = list(iter_jsonl(traces_path))
    if not traces:
        raise RuntimeError(f"No traces in {traces_path}")
    stage_a = list(iter_jsonl(rows_path)) if rows_path.is_file() else []
    state_path = adapter_dir / "state.json"
    state = load_state(state_path)
    c_step = int(state.get("c_step", 0))
    if c_step >= steps and (adapter_dir / "adapter_config.json").is_file():
        print(f"Stage C already complete at c_step {c_step}; skipping train")
        return {"adapter": adapter_dir, "traces": traces_path}

    student = load_causal_lm(student_id, fourbit=fourbit)
    try:
        if (adapter_dir / "adapter_config.json").is_file():
            student = _maybe_load_adapter(student, adapter_dir)
            _enable_checkpointing(student)
        else:
            student = attach_hq_lora(student, r=lora_r)
        student.train()
        device = _module_device(student)
        optimizer = torch.optim.AdamW(
            (p for p in student.parameters() if p.requires_grad and p.device.type != "meta"),
            lr=lr,
        )

        step = c_step
        while step < steps:
            use_a = bool(stage_a) and (step % 10 < int(10 * stage_a_mix))
            pool = stage_a if use_a else traces
            row = pool[step % len(pool)]
            tokenized = tokenize_sft(tokenizer, row, seq_len, formatter)
            if tokenized is None:
                step += 1
                continue
            batch = {key: value.unsqueeze(0).to(device) for key, value in tokenized.items()}
            out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = out.loss
            topk_path = topk_dir / f"{row.get('id')}.pt"
            if (not use_a) and topk_path.is_file() and kd_weight > 0:
                packed = load_topk(topk_path)
                kd = unpack_topk_kd_loss(out.logits, packed, temperature=temperature)
                loss = loss + kd_weight * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.device.type != "meta"), 1.0)
            optimizer.step()
            step += 1
            if step == c_step + 1 or step % 10 == 0:
                print(f"stage C step {step}/{steps}  loss={loss.item():.4f}")
            if step % CHECKPOINT_EVERY == 0 or step == steps:
                _save_adapter(student, adapter_dir, {"stage": "C", "c_step": step, "seq_len": seq_len})
                save_state(state_path, {"step": step, "c_step": step, "stage": "C"})
                print(f"Checkpointed adapter at C step {step}")
    except torch.cuda.OutOfMemoryError:
        free_model(student)
        if seq_len <= FALLBACK_SEQ_LEN:
            raise
        print(f"OOM during Stage C at seq_len={seq_len}; retrying {FALLBACK_SEQ_LEN}")
        return stage_c_align(
            student_id=student_id,
            work_dir=work_dir,
            steps=steps,
            seq_len=FALLBACK_SEQ_LEN,
            lr=lr,
            temperature=temperature,
            kd_weight=kd_weight,
            stage_a_mix=stage_a_mix,
            lora_r=lora_r,
            fourbit=fourbit,
            repo_root=repo_root,
        )

    free_model(student)
    return {"adapter": adapter_dir, "traces": traces_path}


def run_hq_pipeline(
    *,
    work_dir: Path,
    teacher_id: str = DEFAULT_TEACHER,
    student_id: str = DEFAULT_STUDENT,
    max_rows: int = 10000,
    max_traces: int = 5000,
    stage_a_steps: int | None = None,
    stage_c_steps: int = 4000,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_new_tokens: int = 1024,
    fourbit: bool = True,
    extra_jsonl: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print("HQ distill: reload this process after disconnect; Drive/work_dir checkpoints resume automatically.")
    a = stage_a_sft(
        student_id=student_id,
        work_dir=work_dir,
        max_rows=max_rows,
        steps=stage_a_steps,
        seq_len=seq_len,
        fourbit=fourbit,
        extra_jsonl=extra_jsonl,
        repo_root=repo_root,
    )
    b = stage_b_traces(
        teacher_id=teacher_id,
        work_dir=work_dir,
        max_traces=max_traces,
        max_new_tokens=max_new_tokens,
        seq_len=seq_len,
        fourbit=fourbit,
    )
    c = stage_c_align(
        student_id=student_id,
        work_dir=work_dir,
        steps=stage_c_steps,
        seq_len=seq_len,
        fourbit=fourbit,
        repo_root=repo_root,
    )
    return {"adapter": c["adapter"], "stage_a": a["rows"], "traces": b["traces"], "topk": b["topk"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b", "c", "all"), default="all")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--stage-a-steps", type=int, default=None)
    parser.add_argument("--stage-c-steps", type=int, default=4000)
    parser.add_argument("--data", help="Optional extra jsonl of {messages:[...]}")
    parser.add_argument("--no-fourbit", action="store_true")
    args = parser.parse_args()
    work = Path(args.work_dir)
    extra = Path(args.data) if args.data else None
    fourbit = not args.no_fourbit
    if args.stage == "a":
        print(stage_a_sft(student_id=args.student, work_dir=work, max_rows=args.max_rows, steps=args.stage_a_steps, seq_len=args.seq_len, extra_jsonl=extra, fourbit=fourbit))
    elif args.stage == "b":
        print(stage_b_traces(teacher_id=args.teacher, work_dir=work, max_traces=args.max_traces, max_new_tokens=args.max_new_tokens, seq_len=args.seq_len, fourbit=fourbit))
    elif args.stage == "c":
        print(stage_c_align(student_id=args.student, work_dir=work, steps=args.stage_c_steps, seq_len=args.seq_len, fourbit=fourbit))
    else:
        print(
            run_hq_pipeline(
                work_dir=work,
                teacher_id=args.teacher,
                student_id=args.student,
                max_rows=args.max_rows,
                max_traces=args.max_traces,
                stage_a_steps=args.stage_a_steps,
                stage_c_steps=args.stage_c_steps,
                seq_len=args.seq_len,
                max_new_tokens=args.max_new_tokens,
                fourbit=fourbit,
                extra_jsonl=extra,
            )
        )


if __name__ == "__main__":
    main()
