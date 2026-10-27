#!/usr/bin/env sh
# 构建全书：执行全部代码单元，输出静态网页到 book/_build/html。
# 用法：sh scripts/build.sh [--clean]
#   --clean  先删除执行缓存 book/_build/execute 再构建（冷构建，验收时使用）；
#            不删除 book/_build/templates（其中缓存了网页主题模板）。
# --strict 使任何警告或错误（包括代码单元中的 assert 失败）都令构建以非零状态退出。
set -e
cd "$(dirname "$0")/../book"
for arg in "$@"; do
  case "$arg" in
    --clean) rm -rf _build/execute ;;
    *) echo "未知参数：$arg（可用参数：--clean）" >&2; exit 2 ;;
  esac
done
uv run jupyter book build --html --execute --strict
sh ../scripts/postprocess.sh
