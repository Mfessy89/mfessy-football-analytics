#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_directory/.." && pwd)"
cd "$project_root"

port="${PORT:-5000}"
base_path="${BASE_PATH:-/}"
args=(
  run python_app/app.py
  --server.port "$port"
  --server.address 0.0.0.0
  --server.headless true
)

if [[ -n "$base_path" && "$base_path" != "/" ]]; then
  args+=(--server.baseUrlPath "$base_path")
fi

exec streamlit "${args[@]}"
