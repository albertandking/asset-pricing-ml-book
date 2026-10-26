#!/usr/bin/env sh
# 在 http://localhost:8000 提供构建好的网页
set -e
cd "$(dirname "$0")/../book/_build/html"
uv run python -m http.server 8000
