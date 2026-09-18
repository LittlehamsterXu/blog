#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HUGO_BIN="${HUGO_BIN:-hugo}"
if [[ -x "$ROOT/.local/bin/hugo" ]]; then HUGO_BIN="$ROOT/.local/bin/hugo"; fi
DESTINATION="${1:-public}"
BASE_URL="https://blog.little-hamster-xu.workers.dev/"
# --cleanDestinationDir：删掉上次构建留下的过期文件。
# 不带它的话，public/assets/css/ 会按哈希累积历史样式表，wrangler 会把它们一起上传，
# 而且 check-site.py 会在那些陈旧文件上报错，让这道校验形同虚设。
"$HUGO_BIN" --gc --minify --cleanDestinationDir --baseURL "$BASE_URL" --destination "$DESTINATION" --cacheDir "$ROOT/.hugo_cache"
python3 scripts/check-site.py "$DESTINATION" --base-url "$BASE_URL"
