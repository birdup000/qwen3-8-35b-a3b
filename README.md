# Qwen3.8-35B-A3B

Qwen3.8 intelligence on the Qwen3.6-35B-A3B runtime.

Qwen3.8-27B is a dense hybrid model (`qwen3_5`): 64 layers, hidden 5120, Gated DeltaNet + gated full attention, native vision, MTP, and a `reasoning_effort` / `preserve_thinking` chat surface. Qwen3.6-35B-A3B is the sparse sibling (`qwen3_5_moe`): 35B total / ~3B active, 40 layers, hidden 2048, 256 experts with 8 routed + 1 shared.

This repo is the missing combination: **the Qwen3.8 architecture and request API, served as a 35B-A3B MoE that loads like Qwen3.6-35B-A3B.**

## What is matched

| | Qwen3.8-27B (teacher) | Qwen3.6-35B-A3B (runtime) | **Qwen3.8-35B-A3B (this repo)** |
| --- | --- | --- | --- |
| Family | `qwen3_5` dense | `qwen3_5_moe` | `qwen3_5_moe` |
| Total / active | 27B dense | 35B / 3B | 35B / 3B |
| Hidden | 5120 | 2048 | 2048 |
| Layers | 64 | 40 | 40 |
| Layout | 16 × (3×DeltaNet + 1×Attn) | 10 × (3×DeltaNet + 1×Attn) | 10 × (3×DeltaNet + 1×Attn) |
| FFN | dense SwiGLU 17408 | MoE 256 / top-8 + shared | MoE 256 / top-8 + shared |
| Gated attention | 24Q / 4KV, dim 256, **output gate fused into `q_proj`** | 16Q / 2KV, dim 256, same gate | same as 3.6 |
| Gated DeltaNet | 48V / 16QK, dim 128 | 32V / 16QK, dim 128 | same as 3.6 |
| Vocab | 248320 | 248320 | 248320 (aligned for KD) |
| Context | 262K, YaRN to 1M | 262K, YaRN to 1.01M | 262K, YaRN to 1M |
| Thinking API | `reasoning_effort`, `preserve_thinking` | thinking preservation | Qwen3.8 defaults (`xhigh`) |

The compute graph is intentionally identical to Qwen3.6-35B-A3B so Transformers 5.x, vLLM, and SGLang can serve the checkpoint as `Qwen3_5MoeForConditionalGeneration`.

## Weights

This repository ships the architecture, configs, and training/serving path. It does **not** include 70GB of trained weights.

1. **Run like 3.6 immediately** — copy `Qwen/Qwen3.6-35B-A3B` shards (same tensor shapes).
2. **Match 3.8-27B intelligence** — distill logits from `Qwen/Qwen3.8-27B` (same 248k vocab, hybrid-block layer map 64→40).

```bash
python scripts/init_from_qwen36.py \
  --src /path/to/Qwen3.6-35B-A3B \
  --out checkpoints/Qwen3.8-35B-A3B

python scripts/distill_from_qwen38.py \
  --teacher Qwen/Qwen3.8-27B \
  --student checkpoints/Qwen3.8-35B-A3B \
  --data data/distill.jsonl \
  --out checkpoints/Qwen3.8-35B-A3B-distilled

python scripts/export_hf.py \
  --src checkpoints/Qwen3.8-35B-A3B-distilled \
  --out checkpoints/Qwen3.8-35B-A3B-hf

python scripts/export_gguf.py \
  --lora checkpoints/Qwen3.8-35B-A3B-lora \
  --work-dir /tmp/qwen38_gguf \
  --out-dir checkpoints \
  --quant Q4_K_XL \
  --quant Q3_K_XL
```

That writes Unsloth-style XL GGUFs:

* `Qwen3.8-35B-A3B-UD-Q4_K_XL.gguf` (~20GB, default)
* `Qwen3.8-35B-A3B-UD-Q3_K_XL.gguf` (~16GB)

The recipe is Q3_K_M / Q4_K_M plus Q8_0 embeddings and output, Q8_0 Gated DeltaNet `ssm_out`, and a higher-bit `ffn_down` / attention bump — the public Unsloth/Bartowski XL pattern, not bit-identical to Unsloth Dynamic 2.0 Hub uploads. Needs a recent [llama.cpp](https://github.com/ggml-org/llama.cpp) with `qwen3_5_moe` / `switch_mlp` conversion. Peak disk is large: merged BF16 ~70GB + BF16 GGUF ~67GB before the XL file is written.

Serve the HF export like Qwen3.6-35B-A3B:

```bash
vllm serve checkpoints/Qwen3.8-35B-A3B-hf
```

Qwen3.8 request surface:

```python
extra_body = {
    "chat_template_kwargs": {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",  # or medium / low
    }
}
```

Thinking-mode sampling (default): `temperature=1.0`, `top_p=0.95`, `top_k=20`.  
Instruct mode: `temperature=0.7`, `top_p=0.80`, `presence_penalty=1.5`.

## Reference model

`qwen3_8_moe` is a standalone PyTorch implementation of the official hybrid + MoE graph (Gated DeltaNet torch kernels, fused attention gate, top-8 + shared expert, vision tower, MTP head). Use it for tests, distillation plumbing, and architecture checks. Production decode should go through the HF/vLLM export.

```python
from qwen3_8_moe import Qwen38MoeForCausalLM, qwen38_35b_a3b_config, parameter_report

print(parameter_report())  # ~35B total, ~3B active
# Do not instantiate the full 35B model on a laptop; use tiny_config() for tests.
```

```bash
python scripts/count_params.py
python -m pytest tests/
```

## Clone and setup

```bash
git clone https://github.com/birdup000/qwen3-8-35b-a3b.git
cd qwen3-8-35b-a3b
python -m pip install -e ".[colab]"
```

Or one shot:

```bash
REPO_URL=https://github.com/birdup000/qwen3-8-35b-a3b.git DEST=./qwen3-8-35b-a3b bash scripts/colab_setup.sh
```

## Google Colab

[Open in Colab](https://colab.research.google.com/github/birdup000/qwen3-8-35b-a3b/blob/main/notebooks/Qwen3.8-35B-A3B_Colab.ipynb)

Runtime → **A100 40GB+**, High-RAM if you also export GGUF. The notebook clones this repo and `pip install -e ".[colab]"`. Hugging Face login uses a Colab secret named `HF_TOKEN` if you added one; otherwise it opens the Hugging Face widget. Do not snapshot both models in BF16. After distill, section 8 merges the LoRA and writes `UD-Q4_K_XL` / `UD-Q3_K_XL`.

## License

Apache-2.0. Architecture and serving compatibility follow the Qwen3.5 / Qwen3.6 / Qwen3.8 open releases.
