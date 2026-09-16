#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HUGO="$ROOT/.local/bin/hugo"
if [[ ! -x "$HUGO" ]]; then
  echo "缺少项目本地 Hugo，请按 README.md 中的 Linux / WSL 安装说明操作。" >&2
  exit 1
fi
export HUGO_CACHEDIR="$ROOT/.hugo_cache"
exec "$HUGO" "$@"
