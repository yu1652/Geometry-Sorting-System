#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Missing Jetson environment. Follow README.md after checking JetPack.'
  exit 1
fi
entry=live_console.py
arguments=()
for argument in "$@"; do
  case "$argument" in
    --offline) entry=app.py; continue ;;
    --file|--file=*|--preprocessed|--raw) entry=app.py ;;
  esac
  arguments+=("$argument")
done
exec .venv/bin/python "$entry" --fullscreen "${arguments[@]}"
