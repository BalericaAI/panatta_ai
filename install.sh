#!/usr/bin/env bash
#
# install.sh — set up the Python environment for train_tiny_llm.py / generate.py.
#
# MLX ships a Linux x86_64 CPU backend (the "cpu" extra), so this project runs
# on Linux as well as on Apple Silicon. This script is idempotent: re-running it
# just re-checks/upgrades the dependencies.
#
# Usage:
#   bash install.sh
#   python train_tiny_llm.py           # tiny smoke run
#   python generate.py --ckpt checkpoints/ckpt_000200.safetensors --prompt "ROMEO:"
set -euo pipefail

PY="${PYTHON:-python3}"

echo "Using interpreter: $("$PY" --version 2>&1) ($(command -v "$PY"))"

"$PY" -m pip install --user --upgrade pip
"$PY" -m pip install --user -r "$(dirname "$0")/requirements.txt"

echo
echo "Verifying MLX import..."
"$PY" - <<'PY'
import mlx.core as mx
print("mlx OK, default device:", mx.default_device())
PY

echo "Install complete."
