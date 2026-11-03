#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Missing Jetson environment. Follow README.md after checking JetPack.'
  exit 1
fi
exec .venv/bin/python app.py --fullscreen "$@"
