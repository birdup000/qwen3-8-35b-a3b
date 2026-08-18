#!/usr/bin/env bash
# Clone (if needed) and install Qwen3.8-35B-A3B for Google Colab.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/birdup000/qwen3-8-35b-a3b.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
DEST="${DEST:-/content/Qwen3.8-35B-A3B}"

if [[ -d "${DEST}/.git" ]]; then
  echo "Updating ${DEST} (${REPO_BRANCH})"
  git -C "${DEST}" fetch --depth 1 origin "${REPO_BRANCH}"
  git -C "${DEST}" checkout "${REPO_BRANCH}"
  git -C "${DEST}" reset --hard "origin/${REPO_BRANCH}"
elif [[ -f "${DEST}/qwen3_8_moe/configuration.py" ]]; then
  echo "Repo files already present at ${DEST}"
else
  echo "Cloning ${REPO_URL} → ${DEST}"
  mkdir -p "$(dirname "${DEST}")"
  git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${DEST}"
fi

cd "${DEST}"
python3 -m pip install -q -U pip
python3 -m pip install -q -e ".[colab]"
python3 - <<'PY'
import subprocess, sys
for package in ("causal-conv1d", "flash-linear-attention"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print("optional", package, "installed")
    except Exception as exc:
        print("optional", package, "skipped:", exc)
PY
python3 - <<'PY'
from qwen3_8_moe import qwen38_35b_a3b_config, parameter_report
cfg = qwen38_35b_a3b_config().text_config
assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_experts) == (2048, 40, 256)
print("Setup OK: qwen3_8_moe imports")
print(f"Graph {cfg.num_hidden_layers}L / {cfg.num_experts}E  active~{parameter_report()['active']/1e9:.2f}B")
PY

echo "WORKDIR=${DEST}"
