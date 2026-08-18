#!/usr/bin/env python3
"""Self-contained Colab clone + install. Safe to paste or run before the package exists."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/birdup000/qwen3-8-35b-a3b.git"
DEFAULT_BRANCH = "main"
DEFAULT_DEST = "/content/Qwen3.8-35B-A3B"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def clone_repo(
    repo_url: str = DEFAULT_REPO_URL,
    dest: str | Path = DEFAULT_DEST,
    branch: str = DEFAULT_BRANCH,
    token: str | None = None,
) -> Path:
    dest = Path(dest)
    clone_url = repo_url
    if token and repo_url.startswith("https://") and "@" not in repo_url[8:]:
        clone_url = repo_url.replace("https://", f"https://x-access-token:{token}@", 1)

    if (dest / ".git").is_dir():
        run(["git", "fetch", "--depth", "1", "origin", branch], cwd=dest)
        run(["git", "checkout", branch], cwd=dest)
        run(["git", "reset", "--hard", f"origin/{branch}"], cwd=dest)
    elif (dest / "qwen3_8_moe" / "configuration.py").is_file():
        print(f"Repo files already present at {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", branch, clone_url, str(dest)])
    return dest.resolve()


def install_repo(dest: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"])
    extras = dest / "pyproject.toml"
    req = ".[colab]" if extras.is_file() else "."
    run([sys.executable, "-m", "pip", "install", "-q", "-e", req], cwd=dest)


def verify_import(dest: Path) -> None:
    if str(dest) not in sys.path:
        sys.path.insert(0, str(dest))
    os.chdir(dest)
    from qwen3_8_moe import parameter_report, qwen38_35b_a3b_config

    text = qwen38_35b_a3b_config().text_config
    assert (text.hidden_size, text.num_hidden_layers, text.num_experts) == (2048, 40, 256)
    print(f"Setup OK  {text.num_hidden_layers}L/{text.num_experts}E  active~{parameter_report()['active'] / 1e9:.2f}B")


def bootstrap(
    repo_url: str = DEFAULT_REPO_URL,
    dest: str | Path = DEFAULT_DEST,
    branch: str = DEFAULT_BRANCH,
    token: str | None = None,
) -> Path:
    dest_path = clone_repo(repo_url=repo_url, dest=dest, branch=branch, token=token)
    install_repo(dest_path)
    verify_import(dest_path)
    return dest_path


def main() -> None:
    dest = bootstrap(
        repo_url=os.environ.get("REPO_URL", DEFAULT_REPO_URL),
        dest=os.environ.get("DEST", DEFAULT_DEST),
        branch=os.environ.get("REPO_BRANCH", DEFAULT_BRANCH),
        token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
    )
    print("WORKDIR", dest)


if __name__ == "__main__":
    main()
