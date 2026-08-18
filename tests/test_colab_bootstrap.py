from pathlib import Path

from scripts.colab_bootstrap import clone_repo


def test_clone_repo_keeps_existing_checkout(tmp_path):
    dest = tmp_path / "repo"
    (dest / "qwen3_8_moe").mkdir(parents=True)
    (dest / "qwen3_8_moe" / "configuration.py").write_text("# present\n", encoding="utf-8")
    out = clone_repo(dest=dest)
    assert out == dest.resolve()
    assert (dest / "qwen3_8_moe" / "configuration.py").is_file()
