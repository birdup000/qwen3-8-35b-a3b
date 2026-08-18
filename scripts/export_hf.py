#!/usr/bin/env python3
"""Export a snapshot that official Qwen3.6 / qwen3_5_moe servers will load."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_8_moe import qwen38_35b_a3b_config
from qwen3_8_moe.configuration import yarn_rope_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "Qwen3.8-35B-A3B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Initialized or distilled Qwen3.8-35B-A3B dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--yarn", action="store_true", help="Write 1M-context YaRN rope override")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for path in src.iterdir():
        if path.name in {"config.json", "INIT.json", "DISTILL.json"}:
            continue
        target = out / path.name
        if path.is_file():
            shutil.copy2(path, target)
        elif path.is_dir() and path.name.startswith("model"):
            shutil.copytree(path, target, dirs_exist_ok=True)

    config = qwen38_35b_a3b_config()
    payload = config.to_hf_dict()
    if args.yarn:
        payload["text_config"]["rope_parameters"] = yarn_rope_parameters()
        payload["text_config"]["max_position_embeddings"] = 1_000_000
    (out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    gen_src = DEFAULT_CONFIG / "generation_config.json"
    if gen_src.is_file():
        shutil.copy2(gen_src, out / "generation_config.json")
    print(f"Exported official qwen3_5_moe snapshot to {out}")
    print("Serve like Qwen3.6-35B-A3B:")
    print(f"  vllm serve {out}")
    print(f"  python -m sglang.launch_server --model-path {out}")


if __name__ == "__main__":
    main()
