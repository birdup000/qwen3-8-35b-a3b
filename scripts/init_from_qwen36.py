#!/usr/bin/env python3
"""Initialize Qwen3.8-35B-A3B from Qwen3.6-35B-A3B weights.

The two checkpoints share the official `qwen3_5_moe` compute graph
(hidden 2048, 40 layers, 256 experts, 8+1 active). Copying 3.6 weights
makes the model *run* like Qwen3.6-35B-A3B on day one. Overlay the
Qwen3.8 chat / generation surface, then distill from Qwen3.8-27B.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "Qwen3.8-35B-A3B"

WEIGHT_GLOBS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)


def _copy_glob(src: Path, dst: Path, pattern: str) -> list[Path]:
    copied: list[Path] = []
    for path in src.glob(pattern):
        target = dst / path.name
        if path.is_file():
            shutil.copy2(path, target)
            copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Local Qwen3.6-35B-A3B snapshot (or HF cache dir)")
    parser.add_argument("--out", required=True, help="Output directory for Qwen3.8-35B-A3B")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-weights", action="store_true", help="Copy tokenizer/config only")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")
    out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    if not args.skip_weights:
        for pattern in WEIGHT_GLOBS:
            copied.extend(str(path.name) for path in _copy_glob(src, out, pattern))
        if not copied:
            print("Warning: no weight shards found. Use --skip-weights or pass a full snapshot.")

    for name in TOKENIZER_NAMES:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, out / name)
            copied.append(name)

    for name in ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json"):
        src_cfg = Path(args.config_dir) / name
        if src_cfg.is_file():
            shutil.copy2(src_cfg, out / name)

    readme = {
        "model": "Qwen3.8-35B-A3B",
        "initialized_from": str(src),
        "runtime": "identical to Qwen3.6-35B-A3B / qwen3_5_moe",
        "next_step": "python scripts/distill_from_qwen38.py --student {out} --teacher Qwen/Qwen3.8-27B".format(out=out),
    }
    (out / "INIT.json").write_text(json.dumps(readme, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Qwen3.8-35B-A3B skeleton to {out}")
    print(f"Copied {len(copied)} artifacts")
    print("Serve like Qwen3.6:")
    print(f'  vllm serve "{out}" --trust-remote-code')


if __name__ == "__main__":
    main()
