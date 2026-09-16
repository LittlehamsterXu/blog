#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HUGO_BIN="${HUGO_BIN:-hugo}"
if [[ -x "$ROOT/.local/bin/hugo" ]]; then HUGO_BIN="$ROOT/.local/bin/hugo"; fi
DESTINATION="${1:-public}"
BASE_URL="https://blog.little-hamster-xu.workers.dev/"
"$HUGO_BIN" --gc --minify --baseURL "$BASE_URL" --destination "$DESTINATION" --cacheDir "$ROOT/.hugo_cache"
python3 scripts/check-site.py "$DESTINATION" --base-url "$BASE_URL"
