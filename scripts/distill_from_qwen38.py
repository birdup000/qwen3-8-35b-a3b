#!/usr/bin/env python3
"""Distill Qwen3.8-27B intelligence onto the Qwen3.8-35B-A3B student.

The 27B teacher is dense (hidden 5120, 64 layers). The student is the
35B-A3B MoE (hidden 2048, 40 layers) and must keep that graph so it
keeps running like Qwen3.6-35B-A3B.

Both models share vocab_size=248320, so token-level KL is aligned.
Layer mapping: 10 student hybrid blocks <- 16 teacher hybrid blocks
via nearest-block assignment (3:1 DeltaNet/Attention preserved).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TEACHER_LAYERS = 64
STUDENT_LAYERS = 40
BLOCK = 4  # 3 linear + 1 full


def map_student_layer_to_teacher(student_idx: int) -> int:
    """Map student layer i in [0, 40) onto teacher layer in [0, 64)."""
    student_block, offset = divmod(student_idx, BLOCK)
    teacher_block = round(student_block * (TEACHER_LAYERS // BLOCK - 1) / (STUDENT_LAYERS // BLOCK - 1))
    return teacher_block * BLOCK + offset


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = temperature
    student = F.log_softmax(student_logits.float() / t, dim=-1)
    teacher = F.softmax(teacher_logits.float() / t, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * (t * t)


def load_teacher(model_id: str, device: str, dtype: torch.dtype):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("transformers is required to load Qwen3.8-27B") from exc
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )


def load_student(path: str, device: str, dtype: torch.dtype):
    """Prefer official qwen3_5_moe weights; fall back to this repo's module."""
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )
    except Exception:
        from qwen3_8_moe import Qwen38MoeForCausalLM, qwen38_35b_a3b_config

        model = Qwen38MoeForCausalLM(qwen38_35b_a3b_config().text_config)
        model.to(device=device, dtype=dtype)
        return model


def iter_texts(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield row.get("text") or row.get("content") or json.dumps(row)
        return
    text = path.read_text(encoding="utf-8")
    for chunk in text.split("\n\n"):
        if chunk.strip():
            yield chunk.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--student", required=True, help="Qwen3.8-35B-A3B or Qwen3.6-35B-A3B snapshot")
    parser.add_argument("--data", required=True, help="UTF-8 text or JSONL with a 'text' field")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print("Layer map (student -> teacher):")
    for student in range(STUDENT_LAYERS):
        print(f"  {student:02d} -> {map_student_layer_to_teacher(student):02d}")

    teacher = load_teacher(args.teacher, args.device, dtype)
    student = load_student(args.student, args.device, dtype)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    except Exception as exc:
        raise SystemExit(f"Could not load tokenizer from teacher: {exc}") from exc

    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.lr)
    texts = list(iter_texts(Path(args.data)))
    if not texts:
        raise SystemExit("No distillation texts found")

    student.train()
    step = 0
    while step < args.steps:
        text = texts[step % len(texts)]
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        encoded = {key: value.to(args.device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
        if encoded["input_ids"].size(-1) < 8:
            step += 1
            continue
        with torch.no_grad():
            teacher_out = teacher(**encoded)
            teacher_logits = teacher_out.logits
        student_out = student(**encoded)
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out["logits"]
        length = min(student_logits.size(1), teacher_logits.size(1))
        loss = kd_loss(student_logits[:, :length], teacher_logits[:, :length], args.temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps}  kd={loss.item():.4f}  ppl~{math.exp(min(20.0, loss.item())):.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(student, "save_pretrained"):
        student.save_pretrained(out)
        tokenizer.save_pretrained(out)
    else:
        torch.save(student.state_dict(), out / "pytorch_model.bin")
    (out / "DISTILL.json").write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "student": args.student,
                "steps": args.steps,
                "temperature": args.temperature,
                "layer_map": {str(i): map_student_layer_to_teacher(i) for i in range(STUDENT_LAYERS)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved distilled student to {out}")


if __name__ == "__main__":
    main()
