#!/usr/bin/env python3
"""Merge the distilled LoRA and export Unsloth-style XL GGUFs.

Produces files named like:
  Qwen3.8-35B-A3B-UD-Q4_K_XL.gguf
  Qwen3.8-35B-A3B-UD-Q3_K_XL.gguf

These are not llama.cpp native ftypes. They follow the public Unsloth /
Bartowski XL recipe on top of Q3_K_M / Q4_K_M:

* token embeddings + output kept at Q8_0
* Gated DeltaNet / SSM out tensors kept at Q8_0 (Unsloth warns against
  quantizing ssm_out on Qwen3.5/3.6 hybrids)
* attn_v / attn_output / ffn_down (including MoE ffn_down_exps) bumped
  one step above the base mix
* optional llama.cpp importance matrix ("dynamic" bit allocation)

This is the exportable XL recipe, not Unsloth's unpublished Dynamic 2.0
per-layer table. llama.cpp must be new enough to convert qwen3_5_moe
(switch_mlp experts + Gated DeltaNet).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PREFIX = "Qwen3.8-35B-A3B"
DEFAULT_BASE = "Qwen/Qwen3.6-35B-A3B"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_QUANTS = ("Q4_K_XL",)

# Unsloth / Bartowski XL: native K-quant mix + Q8 embed/output, plus
# Qwen3.5/3.6 hybrid tensors Unsloth says not to crush (ssm_out, ffn_down).
XL_RECIPES = {
    "Q3_K_XL": {
        "base": "Q3_K_M",
        "token_embedding": "q8_0",
        "output": "q8_0",
        "tensor_types": (
            "ssm_out=q8_0",
            "attn_v=q5_k",
            "attn_output=q5_k",
            "ffn_down=q5_k",
        ),
    },
    "Q4_K_XL": {
        "base": "Q4_K_M",
        "token_embedding": "q8_0",
        "output": "q8_0",
        "tensor_types": (
            "ssm_out=q8_0",
            "attn_v=q6_k",
            "attn_output=q6_k",
            "ffn_down=q6_k",
        ),
    },
}

HF_CACHE_TO_FREE = (
    "models--Qwen--Qwen3.8-27B",
)


@dataclass(frozen=True)
class LlamaCppTools:
    root: Path
    convert: Path
    quantize: Path
    imatrix: Path | None


def normalize_quant(name: str) -> str:
    cleaned = name.strip().upper().replace("-", "_")
    if cleaned.startswith("UD_"):
        cleaned = cleaned[3:]
    aliases = {
        "Q3_K": "Q3_K_XL",
        "Q3_XL": "Q3_K_XL",
        "Q4_K": "Q4_K_XL",
        "Q4_XL": "Q4_K_XL",
        "Q4_K_L": "Q4_K_XL",
        "Q3_K_L": "Q3_K_XL",
    }
    return aliases.get(cleaned, cleaned)


def gguf_filename(quant: str, prefix: str = MODEL_PREFIX) -> str:
    return f"{prefix}-UD-{normalize_quant(quant)}.gguf"


def recipe_for(quant: str) -> dict:
    key = normalize_quant(quant)
    if key not in XL_RECIPES:
        supported = ", ".join(sorted(XL_RECIPES))
        raise ValueError(f"Unsupported XL quant {quant!r}. Choose {supported}.")
    return XL_RECIPES[key]


def quantize_flags(quant: str) -> list[str]:
    recipe = recipe_for(quant)
    flags = [
        "--token-embedding-type",
        recipe["token_embedding"],
        "--output-tensor-type",
        recipe["output"],
    ]
    for mapping in recipe["tensor_types"]:
        flags.extend(["--tensor-type", mapping])
    return flags


def disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def write_calib_text(path: Path, texts: list[str] | None = None) -> Path:
    from scripts.colab_pipeline import SAMPLE_TEXTS

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [text.strip() for text in (texts or SAMPLE_TEXTS) if text and text.strip()]
    if not rows:
        raise ValueError("No calibration texts")
    # Repeat so llama-imatrix has enough tokens without a huge extra download.
    body = "\n\n".join(rows * 8)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def is_lora_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file()


def is_hf_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def free_teacher_cache() -> list[str]:
    """Drop the 27B teacher snapshot so Colab has room for merge + BF16 GGUF."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    removed: list[str] = []
    for name in HF_CACHE_TO_FREE:
        target = hub / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
            print(f"Removed HF cache {target}")
    return removed


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def find_tool(root: Path, name: str) -> Path | None:
    candidates = [
        root / "build" / "bin" / name,
        root / "bin" / name,
        root / name,
    ]
    which = shutil.which(name)
    if which:
        candidates.append(Path(which))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def find_convert_script(root: Path) -> Path | None:
    for path in (
        root / "convert_hf_to_gguf.py",
        root / "convert" / "convert_hf_to_gguf.py",
    ):
        if path.is_file():
            return path
    matches = list(root.rglob("convert_hf_to_gguf.py"))
    return matches[0] if matches else None


def ensure_llama_cpp(llama_dir: Path, *, with_cuda: bool | None = None) -> LlamaCppTools:
    llama_dir = Path(llama_dir)
    if not (llama_dir / ".git").is_dir() and not find_convert_script(llama_dir):
        llama_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", LLAMA_CPP_URL, str(llama_dir)])

    convert = find_convert_script(llama_dir)
    if convert is None:
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {llama_dir}")
    req = llama_dir / "requirements.txt"
    if req.is_file():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])

    quantize = find_tool(llama_dir, "llama-quantize")
    imatrix = find_tool(llama_dir, "llama-imatrix")
    if quantize is None:
        if with_cuda is None:
            try:
                import torch

                with_cuda = bool(torch.cuda.is_available())
            except Exception:
                with_cuda = False
        cmake = ["cmake", "-S", str(llama_dir), "-B", str(llama_dir / "build"), "-DBUILD_SHARED_LIBS=OFF"]
        if with_cuda:
            cmake.append("-DGGML_CUDA=ON")
        try:
            _run(cmake)
            _run(
                [
                    "cmake",
                    "--build",
                    str(llama_dir / "build"),
                    "--config",
                    "Release",
                    "-j",
                    str(os.cpu_count() or 4),
                    "--target",
                    "llama-quantize",
                    "llama-imatrix",
                ]
            )
        except subprocess.CalledProcessError:
            if with_cuda:
                print("CUDA llama.cpp build failed; retrying CPU-only")
                _run(
                    [
                        "cmake",
                        "-S",
                        str(llama_dir),
                        "-B",
                        str(llama_dir / "build"),
                        "-DBUILD_SHARED_LIBS=OFF",
                        "-DGGML_CUDA=OFF",
                    ]
                )
                _run(
                    [
                        "cmake",
                        "--build",
                        str(llama_dir / "build"),
                        "--config",
                        "Release",
                        "-j",
                        str(os.cpu_count() or 4),
                        "--target",
                        "llama-quantize",
                        "llama-imatrix",
                    ]
                )
            else:
                raise
        quantize = find_tool(llama_dir, "llama-quantize")
        imatrix = find_tool(llama_dir, "llama-imatrix")
    if quantize is None:
        raise FileNotFoundError("llama-quantize binary not found after build")
    return LlamaCppTools(root=llama_dir, convert=convert, quantize=quantize, imatrix=imatrix)


def merge_lora(
    base_id: str,
    lora_dir: Path,
    out_dir: Path,
    *,
    dtype: str = "bfloat16",
    repo_root: Path | None = None,
) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.colab_pipeline import free_model, overlay_qwen38_configs

    out_dir = Path(out_dir)
    if is_hf_checkpoint(out_dir) and not is_lora_dir(out_dir):
        print(f"Reusing merged checkpoint {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging LoRA {lora_dir} into {base_id} → {out_dir}")
    print(f"Disk free before merge: {disk_free_gb(out_dir):.1f} GiB")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "12GiB", "cpu": "70GiB"}
    else:
        kwargs["device_map"] = "cpu"

    try:
        base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    except Exception:
        from transformers import AutoModel

        base = AutoModel.from_pretrained(base_id, **kwargs)
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        tokenizer.save_pretrained(out_dir)
    except Exception as exc:
        print(f"Tokenizer copy skipped: {exc}")
    overlay_qwen38_configs(out_dir, repo_root or REPO_ROOT)
    (out_dir / "MERGE.json").write_text(
        json.dumps({"base": base_id, "lora": str(lora_dir), "dtype": dtype}, indent=2) + "\n",
        encoding="utf-8",
    )
    free_model(model)
    print(f"Merged HF snapshot at {out_dir}")
    return out_dir


def convert_hf_to_gguf(tools: LlamaCppTools, hf_dir: Path, outfile: Path, outtype: str = "bf16") -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.is_file() and outfile.stat().st_size > 1_000_000:
        print(f"Reusing {outfile}")
        return outfile
    cmd = [
        sys.executable,
        str(tools.convert),
        str(hf_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "HF → GGUF convert failed. Need a recent llama.cpp with qwen3_5_moe "
            "/ switch_mlp support (ggml-org/llama.cpp#24502)."
        ) from exc
    if not outfile.is_file():
        raise FileNotFoundError(f"Converter did not write {outfile}")
    return outfile


def compute_imatrix(tools: LlamaCppTools, model_gguf: Path, calib: Path, outfile: Path) -> Path | None:
    if tools.imatrix is None:
        print("llama-imatrix missing; quantizing without an importance matrix")
        return None
    if outfile.is_file() and outfile.stat().st_size > 0:
        print(f"Reusing imatrix {outfile}")
        return outfile
    cmd = [
        str(tools.imatrix),
        "-m",
        str(model_gguf),
        "-f",
        str(calib),
        "-o",
        str(outfile),
        "--no-ppl",
        "-ngl",
        "99",
        "-ot",
        ".ffn_.*_exps.=CPU",
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        print("GPU imatrix failed; retrying CPU-only")
        cmd = [
            str(tools.imatrix),
            "-m",
            str(model_gguf),
            "-f",
            str(calib),
            "-o",
            str(outfile),
            "--no-ppl",
            "-ngl",
            "0",
        ]
        try:
            _run(cmd)
        except subprocess.CalledProcessError:
            print("imatrix failed; continuing with the XL recipe only")
            return None
    return outfile if outfile.is_file() else None


def quantize_gguf(
    tools: LlamaCppTools,
    src: Path,
    out: Path,
    quant: str,
    *,
    imatrix: Path | None = None,
    n_threads: int | None = None,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 1_000_000:
        print(f"Reusing {out}")
        return out
    recipe = recipe_for(quant)
    cmd = [str(tools.quantize)]
    if imatrix is not None:
        cmd.extend(["--imatrix", str(imatrix)])
    cmd.extend(quantize_flags(quant))
    cmd.extend([str(src), str(out), recipe["base"], str(n_threads or os.cpu_count() or 8)])
    print(f"Quantizing {quant} as {recipe['base']} + XL overrides")
    _run(cmd)
    if not out.is_file():
        raise FileNotFoundError(f"llama-quantize did not write {out}")
    return out


def resolve_hf_dir(
    *,
    hf_dir: Path | None,
    lora_dir: Path | None,
    base_id: str,
    work_dir: Path,
    dtype: str,
    repo_root: Path | None,
) -> Path:
    if hf_dir is not None and is_hf_checkpoint(Path(hf_dir)):
        return Path(hf_dir)
    if lora_dir is None:
        raise ValueError("Pass --hf (merged snapshot) or --lora (adapter to merge)")
    lora_dir = Path(lora_dir)
    if is_hf_checkpoint(lora_dir) and not is_lora_dir(lora_dir):
        return lora_dir
    if not is_lora_dir(lora_dir):
        raise FileNotFoundError(f"No adapter_config.json in {lora_dir}")
    return merge_lora(base_id, lora_dir, work_dir / "Qwen3.8-35B-A3B-merged", dtype=dtype, repo_root=repo_root)


def export_gguf(
    *,
    work_dir: Path,
    quants: list[str] | tuple[str, ...] = DEFAULT_QUANTS,
    lora_dir: Path | None = None,
    hf_dir: Path | None = None,
    base_id: str = DEFAULT_BASE,
    out_dir: Path | None = None,
    llama_dir: Path | None = None,
    dtype: str = "bfloat16",
    outtype: str = "bf16",
    imatrix: bool = True,
    keep_bf16: bool = False,
    free_teacher: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir or work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [normalize_quant(q) for q in quants]
    for quant in wanted:
        recipe_for(quant)

    print(f"Disk free: {disk_free_gb(work_dir):.1f} GiB (need ~90GiB peak for merge + BF16 GGUF)")
    if free_teacher:
        free_teacher_cache()

    merged = resolve_hf_dir(
        hf_dir=hf_dir,
        lora_dir=lora_dir,
        base_id=base_id,
        work_dir=work_dir,
        dtype=dtype,
        repo_root=repo_root,
    )
    tools = ensure_llama_cpp(Path(llama_dir or work_dir / "llama.cpp"))
    bf16_path = work_dir / f"{MODEL_PREFIX}-BF16.gguf"
    convert_hf_to_gguf(tools, merged, bf16_path, outtype=outtype)

    imatrix_path = None
    if imatrix:
        calib = write_calib_text(work_dir / "calib.txt")
        imatrix_path = compute_imatrix(tools, bf16_path, calib, work_dir / "imatrix.gguf")

    written: dict[str, Path] = {"hf": merged, "bf16": bf16_path}
    for quant in wanted:
        dest = out_dir / gguf_filename(quant)
        quantize_gguf(tools, bf16_path, dest, quant, imatrix=imatrix_path)
        written[quant] = dest
        (out_dir / f"{dest.stem}.QUANT.json").write_text(
            json.dumps(
                {
                    "file": dest.name,
                    "quant": quant,
                    "recipe": recipe_for(quant),
                    "imatrix": str(imatrix_path) if imatrix_path else None,
                    "source_hf": str(merged),
                    "note": (
                        "Unsloth-style XL (Q8 embed/output + protected hybrid tensors). "
                        "Not bit-identical to Unsloth Dynamic 2.0 Hub uploads."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {dest}  ({dest.stat().st_size / 1024**3:.2f} GiB)")

    if not keep_bf16 and bf16_path.is_file() and any(q in written for q in wanted):
        bf16_path.unlink()
        print(f"Deleted intermediate {bf16_path.name}")
        written.pop("bf16", None)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", help="Distilled PEFT adapter directory")
    parser.add_argument("--hf", help="Already-merged HF snapshot (skips LoRA merge)")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base weights to merge LoRA into")
    parser.add_argument("--work-dir", required=True, help="Scratch dir for merge / llama.cpp / BF16")
    parser.add_argument("--out-dir", help="Where to write the UD-*.gguf files")
    parser.add_argument(
        "--quant",
        action="append",
        dest="quants",
        help="Q4_K_XL (default) and/or Q3_K_XL. Repeat to export both.",
    )
    parser.add_argument("--llama-dir", help="Existing llama.cpp checkout")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--outtype", default="bf16", choices=("bf16", "f16"))
    parser.add_argument("--no-imatrix", action="store_true")
    parser.add_argument("--keep-bf16", action="store_true")
    parser.add_argument("--free-teacher-cache", action="store_true")
    args = parser.parse_args()

    paths = export_gguf(
        work_dir=Path(args.work_dir),
        quants=args.quants or list(DEFAULT_QUANTS),
        lora_dir=Path(args.lora) if args.lora else None,
        hf_dir=Path(args.hf) if args.hf else None,
        base_id=args.base,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        llama_dir=Path(args.llama_dir) if args.llama_dir else None,
        dtype=args.dtype,
        outtype=args.outtype,
        imatrix=not args.no_imatrix,
        keep_bf16=args.keep_bf16,
        free_teacher=args.free_teacher_cache,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
