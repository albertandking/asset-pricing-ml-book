#!/usr/bin/env sh
# 构建全书：执行全部代码单元，输出静态网页到 book/_build/html
set -e
cd "$(dirname "$0")/../book"
uv run jupyter book build --html --execute
