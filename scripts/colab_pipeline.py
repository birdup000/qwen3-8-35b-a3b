"""Google Colab full pipeline for Qwen3.8-35B-A3B.

4-bit two-phase KD from Qwen3.8-27B onto Qwen3.6-35B-A3B, then a LoRA
adapter that still serves as qwen3_5_moe. Teacher and student are never
resident at the same time.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_from_qwen38 import kd_loss, map_student_layer_to_teacher

DEFAULT_TEACHER = "Qwen/Qwen3.8-27B"
DEFAULT_STUDENT = "Qwen/Qwen3.6-35B-A3B"

SAMPLE_TEXTS = [
    "Explain mixture-of-experts routing with a shared expert in one paragraph.",
    "Write a Python function that merges two sorted lists in linear time.",
    "Reason step by step: a bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
    "You are debugging a React app that re-renders in a loop. List the three most likely causes.",
    "Summarize how Gated DeltaNet differs from softmax attention for 256K context.",
    "Given a git rebase conflict in main, write the exact commands to abort and then rebase onto origin/main.",
    "Translate this into a SQL query: customers who spent more than $500 in March 2026.",
    "What is the difference between temperature, top_p, and reasoning_effort when sampling a thinking model?",
    "Write a FastAPI endpoint that streams tokens from an OpenAI-compatible client.",
    "Explain YaRN rope scaling and when not to apply it to short prompts.",
    "Given a flaky pytest that only fails under xdist, list a systematic debug order.",
    "Convert this into a typed TypeScript function: function add(a, b) { return a + b }",
    "Plan a 4-step agent that edits a repo, runs tests, and opens a pull request.",
    "Why does GQA with 16 query heads and 2 KV heads shrink the KV cache?",
    "Write a bash one-liner that finds the largest directories under /content.",
    "Critique this SQL: SELECT * FROM users WHERE created_at = '2026-08-17'.",
]


def detect_runtime() -> dict:
    gpu_name = None
    vram_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024**3)
    free_t4 = vram_gb < 20
    return {
        "gpu": gpu_name or "cpu",
        "vram_gb": round(vram_gb, 2),
        "cuda": torch.cuda.is_available(),
        "recommend_full": vram_gb >= 22,
        "note": (
            "Full 4-bit two-phase KD needs ~22GB+ VRAM. This GPU is tight; expect OOM on T4."
            if free_t4 or not torch.cuda.is_available()
            else "This GPU can run the 4-bit two-phase full pipeline."
        ),
    }


def ensure_repo(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root or REPO_ROOT).resolve()
    if not (root / "qwen3_8_moe" / "configuration.py").is_file():
        raise FileNotFoundError(
            f"Repo not found at {root}. Upload the project zip or mount Drive first."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    return root


def write_sample_data(path: Path, extra: list[str] | None = None, min_rows: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = SAMPLE_TEXTS + (extra or [])
    rows: list[str] = []
    index = 0
    while len(rows) < max(min_rows, len(seed)):
        base = seed[index % len(seed)]
        cycle = index // len(seed)
        rows.append(base if cycle == 0 else f"{base} Expand the answer with one extra example. [v{cycle + 1}]")
        index += 1
    with path.open("w", encoding="utf-8") as handle:
        for text in rows:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return path


def overlay_qwen38_configs(out_dir: Path, repo_root: Path | None = None) -> Path:
    """Write the Qwen3.8 chat/generation surface onto a student snapshot dir."""
    repo_root = Path(repo_root or REPO_ROOT)
    src = repo_root / "configs" / "Qwen3.8-35B-A3B"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json"):
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, out_dir / name)
    return out_dir


def gpu_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / (1024**3)


def bitsandbytes_config(dtype: str = "bfloat16", cpu_offload: bool = False):
    from transformers import BitsAndBytesConfig

    compute = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute,
    }
    if cpu_offload:
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    return BitsAndBytesConfig(**kwargs)


def load_causal_lm(model_id: str, fourbit: bool, dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if fourbit:
        free = gpu_free_gb()
        print(f"Loading {model_id} in NF4  free_gpu={free:.1f}GiB  cpu_offload=True")
        kwargs["quantization_config"] = bitsandbytes_config(dtype, cpu_offload=True)
        kwargs["device_map"] = "auto"
        if torch.cuda.is_available():
            budget = max(free - 2.0, 8.0)
            kwargs["max_memory"] = {0: f"{budget:.1f}GiB", "cpu": "80GiB"}
    else:
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError as exc:
        message = str(exc)
        if "CPU or the disk" not in message and "cpu or the disk" not in message.lower():
            raise
        print("Retrying 4-bit load with explicit CPU offload after dispatch error")
        kwargs["quantization_config"] = bitsandbytes_config(dtype, cpu_offload=True)
        kwargs["device_map"] = "auto"
        if torch.cuda.is_available():
            kwargs["max_memory"] = {0: f"{max(gpu_free_gb() - 2.0, 6.0):.1f}GiB", "cpu": "80GiB"}
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def free_model(model=None) -> None:
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del model
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()
    print(f"GPU free after unload: {gpu_free_gb():.1f} GiB")


def run_tiny_pipeline(steps: int = 20, seq_len: int = 64, device: str | None = None) -> dict:
    """Always-on Colab path: distill a tiny student onto itself + chat format."""
    from qwen3_8_moe import Qwen38MoeForCausalLM, tiny_config
    from qwen3_8_moe.chat import Qwen38ChatFormatter

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = tiny_config()
    teacher = Qwen38MoeForCausalLM(config.text_config).to(device).eval()
    student = Qwen38MoeForCausalLM(config.text_config).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    vocab = config.text_config.vocab_size
    losses: list[float] = []
    for step in range(steps):
        ids = torch.randint(1, vocab, (2, seq_len), device=device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits
        student_logits = student(input_ids=ids, use_cache=False).logits
        loss = kd_loss(student_logits, teacher_logits, temperature=2.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step == 0 or (step + 1) % 5 == 0:
            print(f"tiny step {step + 1}/{steps}  kd={loss.item():.4f}")

    formatter = Qwen38ChatFormatter(enable_thinking=True, reasoning_effort="medium")
    prompt = formatter.format_messages([{"role": "user", "content": "What is MoE?"}])
    generated = student.generate(ids[:1, :8], max_new_tokens=8, temperature=0.0)
    return {
        "device": device,
        "final_loss": losses[-1],
        "losses": losses,
        "prompt_preview": prompt,
        "generated_shape": list(generated.shape),
        "layer_map_example": {0: map_student_layer_to_teacher(0), 39: map_student_layer_to_teacher(39)},
    }


def collect_teacher_logits(
    teacher_id: str,
    data_path: Path,
    out_path: Path,
    *,
    tokenizer_id: str | None = None,
    fourbit: bool = True,
    seq_len: int = 256,
    max_samples: int = 32,
) -> Path:
    from transformers import AutoTokenizer

    from scripts.distill_from_qwen38 import iter_texts

    out_dir = out_path if out_path.suffix == "" else out_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.pt"))
    if existing:
        print(f"Reusing {len(existing)} teacher batches in {out_dir}")
        return out_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or teacher_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = load_causal_lm(teacher_id, fourbit=fourbit)
    teacher.eval()
    device = next(p for p in teacher.parameters() if p.device.type != "meta").device

    texts = [text for text in iter_texts(data_path) if text and text.strip()]
    if not texts:
        raise RuntimeError(f"No distillation texts in {data_path}")

    n = 0
    cursor = 0
    attempts = 0
    while n < max_samples and attempts < max_samples * 4:
        text = texts[cursor % len(texts)]
        cursor += 1
        attempts += 1
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len, padding=False)
        if encoded["input_ids"].size(-1) < 8:
            continue
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = teacher(**encoded).logits.float().cpu()
        torch.save(
            {
                "input_ids": encoded["input_ids"].cpu(),
                "attention_mask": encoded["attention_mask"].cpu(),
                "logits": logits,
            },
            out_dir / f"{n:04d}.pt",
        )
        n += 1
        print(f"teacher logits {n}/{max_samples}  seq={encoded['input_ids'].size(-1)}")
    if n == 0:
        raise RuntimeError("No usable distillation samples")
    free_model(teacher)
    meta = {"samples": n, "teacher": teacher_id, "seq_len": seq_len}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {n} teacher batches to {out_dir}")
    return out_dir


def _iter_saved_batches(path: Path):
    path = Path(path)
    files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
    for file in files:
        yield torch.load(file, map_location="cpu", weights_only=False)


def distill_student_from_logits(
    student_id: str,
    logits_path: Path,
    out_dir: Path,
    *,
    fourbit: bool = True,
    lora: bool = True,
    steps: int = 50,
    lr: float = 1e-4,
    temperature: float = 2.0,
    repo_root: Path | None = None,
) -> Path:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_qwen38_configs(out_dir, repo_root)
    free_model()
    student = load_causal_lm(student_id, fourbit=fourbit)
    if lora:
        student = prepare_model_for_kbit_training(student)
        student = get_peft_model(
            student,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        student.print_trainable_parameters()
    student.train()
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=lr)
    embed = student.get_input_embeddings()
    device = embed.weight.device if embed is not None else next(p for p in student.parameters() if p.device.type != "meta").device

    batches = list(_iter_saved_batches(logits_path))
    if not batches:
        raise RuntimeError(f"No batches in {logits_path}")

    step = 0
    while step < steps:
        batch = batches[step % len(batches)]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        teacher_logits = batch["logits"].to(device)
        out = student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = out.logits
        length = min(student_logits.size(1), teacher_logits.size(1))
        loss = kd_loss(student_logits[:, :length], teacher_logits[:, :length], temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step == 1 or step % 10 == 0:
            print(f"student step {step}/{steps}  kd={loss.item():.4f}")

    student.save_pretrained(out_dir)
    (out_dir / "DISTILL.json").write_text(
        json.dumps(
            {
                "teacher_logits": str(logits_path),
                "student": student_id,
                "steps": steps,
                "lora": lora,
                "fourbit": fourbit,
                "layer_map": {str(i): map_student_layer_to_teacher(i) for i in range(40)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved distilled student / LoRA adapter to {out_dir}")
    free_model(student)
    return out_dir


def run_full_pipeline(
    *,
    teacher_id: str = DEFAULT_TEACHER,
    student_id: str = DEFAULT_STUDENT,
    work_dir: Path,
    data_path: Path | None = None,
    extra_texts: list[str] | None = None,
    fourbit: bool = True,
    seq_len: int = 512,
    max_samples: int = 64,
    steps: int = 100,
    lr: float = 1e-4,
    temperature: float = 2.0,
    yarn: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Overlay 3.8 configs, dump 27B teacher logits, LoRA-distill 35B-A3B, export HF."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if data_path is None:
        data_path = write_sample_data(work_dir / "distill.jsonl", extra=extra_texts)
    student_dir = overlay_qwen38_configs(work_dir / "Qwen3.8-35B-A3B", repo_root)
    logits_dir = collect_teacher_logits(
        teacher_id,
        Path(data_path),
        work_dir / "teacher_logits",
        fourbit=fourbit,
        seq_len=seq_len,
        max_samples=max_samples,
    )
    adapter_dir = distill_student_from_logits(
        student_id,
        logits_dir,
        work_dir / "Qwen3.8-35B-A3B-lora",
        fourbit=fourbit,
        lora=True,
        steps=steps,
        lr=lr,
        temperature=temperature,
        repo_root=repo_root,
    )
    hf_dir = export_hf_snapshot(student_dir, work_dir / "Qwen3.8-35B-A3B-hf", yarn=yarn)
    return {"student": student_dir, "logits": logits_dir, "lora": adapter_dir, "hf": hf_dir}


def export_unsloth_xl_gguf(
    *,
    work_dir: Path,
    lora_dir: Path | None = None,
    hf_dir: Path | None = None,
    out_dir: Path | None = None,
    quants: list[str] | tuple[str, ...] = ("Q4_K_XL",),
    base_id: str = DEFAULT_STUDENT,
    free_teacher: bool = True,
    imatrix: bool = True,
    keep_bf16: bool = False,
) -> dict[str, Path]:
    """After distill: merge LoRA and write Unsloth-style UD-Q4_K_XL / UD-Q3_K_XL GGUFs."""
    from scripts.export_gguf import export_gguf

    return export_gguf(
        work_dir=work_dir,
        quants=quants,
        lora_dir=lora_dir,
        hf_dir=hf_dir,
        base_id=base_id,
        out_dir=out_dir,
        imatrix=imatrix,
        keep_bf16=keep_bf16,
        free_teacher=free_teacher,
    )


def export_hf_snapshot(src: Path, out: Path, yarn: bool = False) -> Path:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "export_hf.py"), "--src", str(src), "--out", str(out)]
    if yarn:
        cmd.append("--yarn")
    subprocess.check_call(cmd)
    return Path(out)
