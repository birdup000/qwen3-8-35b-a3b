from pathlib import Path

from scripts.colab_bootstrap import clone_repo


def test_clone_repo_keeps_existing_checkout(tmp_path):
    dest = tmp_path / "repo"
    (dest / "qwen3_8_moe").mkdir(parents=True)
    (dest / "qwen3_8_moe" / "configuration.py").write_text("# present\n", encoding="utf-8")
    out = clone_repo(dest=dest)
    assert out == dest.resolve()
    assert (dest / "qwen3_8_moe" / "configuration.py").is_file()


def test_colab_notebook_is_hq_distill_and_gguf_only():
    from pathlib import Path

    text = Path("notebooks/Qwen3.8-35B-A3B_Colab.ipynb").read_text(encoding="utf-8")
    assert "run_hq_pipeline" in text
    assert "export_unsloth_xl_gguf" in text
    assert "UD-Q4_K_XL" in text
    assert "hq_maxmix" in text
    assert "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation" in text
    assert "sft_balanced" in text
    assert "run_full_pipeline" not in text
    assert "Smoke distill" not in text
    assert "64 short prompts" not in text
    assert "section 5b" not in text
