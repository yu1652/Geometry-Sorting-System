#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OPENBLAS_CORETYPE=ARMV8
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH:-}"
exec .venv_jp4/bin/python side_model_manager.py "$@"
