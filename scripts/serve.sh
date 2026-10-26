#!/usr/bin/env sh
# 在 http://localhost:8000 提供构建好的网页，并尝试自动打开浏览器；按 Ctrl+C 停止
set -e
cd "$(dirname "$0")/../book/_build/html"
URL="http://localhost:8000"
echo "本书已在 $URL 启动，按 Ctrl+C 停止"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) (sleep 1; cmd.exe //c start "" "$URL") & ;;
  Darwin) (sleep 1; open "$URL") & ;;
esac
uv run python -u -m http.server 8000
