#!/usr/bin/env python3
"""Chase Qwen3.8-27B on the 35B-A3B runtime (no git commit from this file).

Research this implements
------------------------
Qwen3 / GKD / MiniLLM / Lightning-OPD: SFT warmup, then *the same teacher you
want to match* writes traces, then on-policy top-k distillation. Extra
Max-Preview SFT dumps do not close a ~9pt LiveCodeBench gap.

Capacity: 27B is dense; the student is ~3B active with frozen expert banks.
Mode-seeking truncated reverse KL on the teacher's top-256 is the right
divergence for that gap (MiniLLM / Fu et al. truncated R-KL).

Colab 40GB cannot co-locate teacher and student. Stages never load both.

Curriculum
----------
A  already running: Max/GLM/Kimi SFT warmup (hq_maxmix)
P  collect hard *prompts only* (OpenCodeReasoning contests, OpenThoughts3
   questions, Magicoder problems, Stage A users). Drop QwQ/R1/Max answers
   so 27B writes them. Do not train on Arena-Hard (that is a benchmark).
B  Qwen3.8-27B generate + assistant-span top-256 packs
C  reverse-KL + response CE on those 27B traces (resume maxmix adapter)
D  on-policy: student rollouts → 27B scores logits → more reverse-KL (GKD λ→1)

80GB: raise --max-traces 20000 --seq-len 4096 --max-new-tokens 4096.
Even then, near-27B on LiveCodeBench likely still needs expert training.

License: 27B is Apache-2.0. OT3 is Apache-2.0. Do not mix Max dumps into B/C.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hq_distill import (
    DEFAULT_SEQ_LEN,
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    append_jsonl,
    existing_trace_ids,
    generate_rollouts,
    iter_jsonl,
    score_traces_topk,
    stage_b_traces,
    stage_c_align,
)

CHASE_TOP_K = 256
CHASE_MAX_NEW = 1536
CHASE_C_STEPS = 6000
CHASE_D_STEPS = 3000  # additional on-policy steps (own d_step counter)
CHASE_KD_WEIGHT = 0.85
CHASE_A_MIX = 0.1
CHASE_LR = 5e-5

# Prompt-only sources. Answers are discarded so Qwen3.8-27B is the teacher.
# Competitive-code is oversampled because LiveCodeBench is the largest clean
# gap vs 27B (~90 vs ~80). Arena-Hard is a benchmark — do not train on it.
OT3_ID = "open-thoughts/OpenThoughts3-1.2M"
MAGICODER_ID = "ise-uiuc/Magicoder-OSS-Instruct-75K"
OCR_ID = "nvidia/OpenCodeReasoning"

EVAL_LEAK_MARKERS = (
    "humaneval",
    "openai_humaneval",
    "livecodebench",
    "lcbv",
    "gpqa diamond",
    "gpqa",
    "aime 2024",
    "aime 2025",
    "aime 2026",
    "gsm8k",
    "competition_math",
    "arena-hard",
    "swe-bench",
    "nl2repo",
    "mbpp",
)


def looks_eval_leak(text: str, source: str = "") -> bool:
    blob = f"{source}\n{text}".lower()
    return any(marker in blob for marker in EVAL_LEAK_MARKERS)


def extract_prompt_only(row: dict[str, Any], kind: str) -> list[dict[str, str]] | None:
    """Keep the question; drop every teacher/QwQ/solution turn."""
    if kind == "ot3":
        raw = row.get("conversations") or []
        if not isinstance(raw, list):
            return None
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("from") or item.get("role") or "").lower()
            content = str(item.get("value") or item.get("content") or "").strip()
            if role in {"human", "user"} and content:
                return [{"role": "user", "content": content}]
        return None
    if kind == "magicoder":
        problem = str(row.get("problem") or row.get("instruction") or row.get("query") or "").strip()
        if not problem:
            return None
        return [{"role": "user", "content": problem}]
    if kind == "arena_hard":
        # Extractor kept for tests; collector does not pull this benchmark.
        turns = row.get("turns") or row.get("prompt") or row.get("messages")
        if isinstance(turns, str):
            turns = [{"content": turns}]
        if not isinstance(turns, list) or not turns:
            return None
        first = turns[0]
        if isinstance(first, dict):
            content = str(first.get("content") or first.get("value") or "").strip()
        else:
            content = str(first).strip()
        if not content:
            return None
        return [{"role": "user", "content": content}]
    if kind == "ocr":
        splits = [
            part.strip()
            for part in str(row.get("split") or "").lower().replace(";", ",").split(",")
            if part.strip()
        ]
        if any(part in {"test", "valid", "validation"} for part in splits):
            return None
        problem = str(row.get("input") or "").strip()
        if not problem:
            return None
        return [{"role": "user", "content": problem}]
    return None


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _stream_hf(repo_id: str, split: str = "train", name: str | None = None):
    from datasets import load_dataset

    if name is None:
        return load_dataset(repo_id, split=split, streaming=True)
    return load_dataset(repo_id, name, split=split, streaming=True)


def collect_chase_prompts(
    path: Path,
    *,
    max_prompts: int = 8000,
    stage_a_jsonl: Path | None = None,
    ocr_quota: int = 2500,
    magicoder_quota: int = 1000,
    code_quota: int = 2000,
    math_quota: int = 1600,
    science_quota: int = 400,
    stage_a_quota: int = 1500,
) -> Path:
    """Write prompt-only jsonl. Existing ids are kept (Drive resume).

    Order is competitive-code first so LiveCodeBench-like prompts fill before
    math/chat when max_prompts caps the pool.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    have = existing_trace_ids(path)
    if len(have) >= max_prompts:
        print(f"Reusing {len(have)} chase prompts in {path}")
        return path

    def emit(row_id: str, messages: list[dict[str, str]], source: str, domain: str = "", extra: str = "") -> bool:
        if row_id in have or len(have) >= max_prompts:
            return False
        user = " ".join(m["content"] for m in messages if m["role"] == "user")
        if looks_eval_leak(user, f"{source} {extra}"):
            return False
        append_jsonl(
            path,
            {"id": row_id, "source": source, "domain": domain, "messages": messages},
        )
        have.add(row_id)
        if len(have) % 200 == 0:
            print(f"chase prompts {len(have)}/{max_prompts}")
        return True

    try:
        from datasets import load_dataset  # noqa: F401
        have_datasets = True
    except ImportError:
        have_datasets = False
        print("datasets not installed; chase prompts fall back to Stage A users")

    if have_datasets:
        try:
            taken = 0
            stream = _stream_hf(OCR_ID, split="split_0", name="split_0")
            for raw in stream:
                if taken >= ocr_quota or len(have) >= max_prompts:
                    break
                messages = extract_prompt_only(raw, "ocr")
                if messages is None:
                    continue
                extra = f"{raw.get('dataset') or ''} {raw.get('source') or ''} {raw.get('split') or ''}"
                if emit(_stable_id("ocr", messages[0]["content"]), messages, OCR_ID, "code", extra=extra):
                    taken += 1
            print(f"OpenCodeReasoning prompts {taken}")
        except Exception as exc:
            print(f"Skipping {OCR_ID}: {exc}")

        try:
            taken = 0
            for raw in _stream_hf(MAGICODER_ID):
                if taken >= magicoder_quota or len(have) >= max_prompts:
                    break
                messages = extract_prompt_only(raw, "magicoder")
                if messages is None:
                    continue
                if emit(_stable_id("magic", messages[0]["content"]), messages, MAGICODER_ID, "code"):
                    taken += 1
            print(f"Magicoder prompts {taken}")
        except Exception as exc:
            print(f"Skipping {MAGICODER_ID}: {exc}")

        domain_taken = {"code": 0, "math": 0, "science": 0}
        domain_quota = {"code": code_quota, "math": math_quota, "science": science_quota}
        try:
            stream = _stream_hf(OT3_ID)
            try:
                stream = stream.shuffle(seed=42, buffer_size=4096)
            except Exception:
                pass
            for raw in stream:
                if len(have) >= max_prompts:
                    break
                if all(domain_taken[d] >= domain_quota[d] for d in domain_quota):
                    break
                domain = str(raw.get("domain") or "").lower()
                if domain not in domain_quota or domain_taken[domain] >= domain_quota[domain]:
                    continue
                messages = extract_prompt_only(raw, "ot3")
                if messages is None:
                    continue
                extra = str(raw.get("source") or "")
                if emit(_stable_id("ot3", messages[0]["content"]), messages, OT3_ID, domain, extra=extra):
                    domain_taken[domain] += 1
            print(f"OT3 quotas filled {domain_taken}")
        except Exception as exc:
            print(f"Skipping {OT3_ID}: {exc}")

    if stage_a_jsonl is not None and stage_a_jsonl.is_file():
        taken = 0
        for row in iter_jsonl(stage_a_jsonl):
            if taken >= stage_a_quota or len(have) >= max_prompts:
                break
            messages = [m for m in row.get("messages", []) if m.get("role") != "assistant"]
            if not messages:
                continue
            extra = str(row.get("source") or "stage_a")
            if emit(f"sfta-{row.get('id')}", messages, extra, str(row.get("domain") or ""), extra=extra):
                taken += 1
        print(f"Stage A user prompts {taken}")

    print(f"Chase prompts {path} n={len(have)}")
    if not have:
        raise RuntimeError("No chase prompts collected. Check HF login / dataset ids.")
    return path


def stage_a_steps_done(adapter_dir: Path) -> int:
    """Read hq_maxmix a_step from state.json (TRAIN.json fallback)."""
    from scripts.hq_distill import load_state

    adapter_dir = Path(adapter_dir)
    state = load_state(adapter_dir / "state.json")
    step = int(state.get("a_step") or 0)
    if step <= 0:
        train = load_state(adapter_dir / "TRAIN.json")
        step = int(train.get("a_step") or 0)
    return step


def require_stage_a_done(adapter_dir: Path, min_a_step: int = 4000) -> int:
    """Block chase until Maxmix Stage A has actually finished.

    Keep the running A job on hq_maxmix. Chase B/C/D (and prompt collection on
    the same Colab) wait so they do not fight the A100 / copy a mid-run LoRA.
    """
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Stage A adapter not found at {adapter_dir}. Let hq_maxmix Stage A finish first."
        )
    step = stage_a_steps_done(adapter_dir)
    if step < min_a_step:
        raise RuntimeError(
            f"Stage A is still at a_step {step}/{min_a_step} in {adapter_dir}. "
            "Keep that job running. Start chase only after it hits the target."
        )
    print(f"Stage A finished at a_step {step}; chase may seed from {adapter_dir}")
    return step


def seed_adapter(src: Path, dest: Path) -> Path:
    """Copy the Stage A LoRA into the chase work dir once. Never overwrite a newer C/D adapter."""
    src = Path(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / "adapter_config.json").is_file():
        print(f"Keeping existing chase adapter {dest}")
        return dest
    if not (src / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Stage A adapter not found at {src}. Finish hq_maxmix Stage A first.")
    shutil.copytree(src, dest, dirs_exist_ok=True)
    print(f"Seeded chase adapter from {src} -> {dest}")
    return dest


def seed_stage_a_jsonl(src: Path | None, dest: Path) -> None:
    """Copy Maxmix Stage A rows into hq_27b so C can mix 10% CE."""
    dest = Path(dest)
    if dest.is_file():
        return
    if src is None or not Path(src).is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Copied Stage A mix {src} -> {dest}")


def chase_progress(work_dir: Path) -> dict[str, Any]:
    from scripts.hq_distill import load_state

    work_dir = Path(work_dir)
    state = load_state(work_dir / "adapter" / "state.json")
    return {
        "dir": str(work_dir),
        "prompts": len(existing_trace_ids(work_dir / "prompts.jsonl")),
        "traces": len(existing_trace_ids(work_dir / "traces.jsonl")),
        "opd": len(existing_trace_ids(work_dir / "opd_traces.jsonl")),
        "mix": len(existing_trace_ids(work_dir / "mix_traces.jsonl")),
        "adapter": (work_dir / "adapter" / "adapter_config.json").is_file(),
        "c_step": int(state.get("c_step", 0)),
        "d_step": int(state.get("d_step", 0)),
        "stage": state.get("stage"),
    }


def merge_trace_jsonl(out: Path, *sources: Path) -> Path:
    have = existing_trace_ids(out)
    for src in sources:
        if not src.is_file():
            continue
        for row in iter_jsonl(src):
            row_id = str(row.get("id") or "")
            if not row_id or row_id in have:
                continue
            append_jsonl(out, row)
            have.add(row_id)
    print(f"Merged traces {out} n={len(have)}")
    return out


def run_chase(
    *,
    work_dir: Path,
    seed_adapter_dir: Path,
    stage: str = "all",
    teacher_id: str = DEFAULT_TEACHER,
    student_id: str = DEFAULT_STUDENT,
    max_prompts: int = 8000,
    max_traces: int = 4000,
    max_new_tokens: int = CHASE_MAX_NEW,
    seq_len: int = DEFAULT_SEQ_LEN,
    top_k: int = CHASE_TOP_K,
    stage_c_steps: int = CHASE_C_STEPS,
    stage_d_steps: int = CHASE_D_STEPS,
    fourbit: bool = True,
    stage_a_jsonl: Path | None = None,
    min_a_step: int = 4000,
) -> dict[str, Path]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(
        "CHASE 27B: teacher is Qwen3.8-27B (not Max-Preview). "
        "Keep hq_maxmix Stage A running until it finishes; this path starts after. "
        "40GB A100: B generate is slow; D is a second teacher load. "
        "This will not match 27B dense if experts stay frozen."
    )
    require_stage_a_done(seed_adapter_dir, min_a_step=min_a_step)
    seed_adapter(seed_adapter_dir, work_dir / "adapter")
    src_a = Path(stage_a_jsonl) if stage_a_jsonl else Path(seed_adapter_dir).parent / "stage_a.jsonl"
    seed_stage_a_jsonl(src_a, work_dir / "stage_a.jsonl")
    result: dict[str, Path] = {"adapter": work_dir / "adapter"}
    print("chase progress", chase_progress(work_dir))

    if stage in {"prompts", "p", "all"}:
        result["prompts"] = collect_chase_prompts(
            work_dir / "prompts.jsonl",
            max_prompts=max_prompts,
            stage_a_jsonl=src_a if src_a.is_file() else stage_a_jsonl,
        )
        if stage in {"prompts", "p"}:
            return result

    if stage in {"b", "all"}:
        if not (work_dir / "prompts.jsonl").is_file():
            collect_chase_prompts(
                work_dir / "prompts.jsonl",
                max_prompts=max_prompts,
                stage_a_jsonl=src_a,
            )
        result.update(
            stage_b_traces(
                teacher_id=teacher_id,
                work_dir=work_dir,
                max_traces=max_traces,
                max_new_tokens=max_new_tokens,
                seq_len=seq_len,
                top_k=top_k,
                fourbit=fourbit,
            )
        )
        if stage == "b":
            return result

    if stage in {"c", "all"}:
        result.update(
            stage_c_align(
                student_id=student_id,
                work_dir=work_dir,
                steps=stage_c_steps,
                seq_len=seq_len,
                lr=CHASE_LR,
                temperature=2.0,
                kd_weight=CHASE_KD_WEIGHT,
                stage_a_mix=CHASE_A_MIX,
                fourbit=fourbit,
                reverse_kl=True,
                traces_name="traces.jsonl",
            )
        )
        if stage == "c":
            return result

    if stage in {"d", "all"}:
        if not (work_dir / "traces.jsonl").is_file():
            raise FileNotFoundError(f"Run chase stage b first: missing {work_dir / 'traces.jsonl'}")
        generate_rollouts(
            model_id=student_id,
            work_dir=work_dir,
            traces_name="opd_traces.jsonl",
            max_traces=min(max_traces, 2000),
            max_new_tokens=min(max_new_tokens, 1024),
            seq_len=seq_len,
            fourbit=fourbit,
            adapter_dir=work_dir / "adapter",
            id_prefix="opd",
        )
        score_traces_topk(
            teacher_id=teacher_id,
            work_dir=work_dir,
            traces_name="opd_traces.jsonl",
            seq_len=seq_len,
            top_k=top_k,
            fourbit=fourbit,
        )
        merge_trace_jsonl(work_dir / "mix_traces.jsonl", work_dir / "traces.jsonl", work_dir / "opd_traces.jsonl")
        result.update(
            stage_c_align(
                student_id=student_id,
                work_dir=work_dir,
                steps=stage_d_steps,
                seq_len=seq_len,
                lr=CHASE_LR,
                temperature=2.0,
                kd_weight=CHASE_KD_WEIGHT,
                stage_a_mix=0.05,
                fourbit=fourbit,
                reverse_kl=True,
                traces_name="mix_traces.jsonl",
                step_key="d_step",
            )
        )
    print("chase progress", chase_progress(work_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed-adapter", required=True, help="hq_maxmix/adapter after Stage A")
    parser.add_argument("--stage", choices=("prompts", "p", "b", "c", "d", "all"), default="prompts")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--max-prompts", type=int, default=8000)
    parser.add_argument("--max-traces", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=CHASE_MAX_NEW)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--top-k", type=int, default=CHASE_TOP_K)
    parser.add_argument("--stage-c-steps", type=int, default=CHASE_C_STEPS)
    parser.add_argument("--stage-d-steps", type=int, default=CHASE_D_STEPS)
    parser.add_argument("--stage-a-jsonl", default="")
    parser.add_argument("--min-a-step", type=int, default=4000, help="Refuse chase until hq_maxmix a_step reaches this")
    parser.add_argument("--no-fourbit", action="store_true")
    args = parser.parse_args()
    print(
        run_chase(
            work_dir=Path(args.work_dir),
            seed_adapter_dir=Path(args.seed_adapter),
            stage=args.stage,
            teacher_id=args.teacher,
            student_id=args.student,
            max_prompts=args.max_prompts,
            max_traces=args.max_traces,
            max_new_tokens=args.max_new_tokens,
            seq_len=args.seq_len,
            top_k=args.top_k,
            stage_c_steps=args.stage_c_steps,
            stage_d_steps=args.stage_d_steps,
            fourbit=not args.no_fourbit,
            stage_a_jsonl=Path(args.stage_a_jsonl) if args.stage_a_jsonl else None,
            min_a_step=args.min_a_step,
        )
    )


if __name__ == "__main__":
    main()
