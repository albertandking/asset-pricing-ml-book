#!/usr/bin/env sh
# 构建后处理 book/_build/html：
# 1. 把页面语言从主题默认的 en 改为 zh-CN（影响读屏软件、浏览器翻译提示与断词）；
# 2. 传入网站地址时，把 sitemap.xml、robots.txt 中 mystmd 静态构建写死的 http://localhost:3000 换成该地址。
# 用法：sh scripts/postprocess.sh [网站地址]，例如 sh scripts/postprocess.sh https://albertandking.github.io/asset-pricing-ml-book
set -e
cd "$(dirname "$0")/../book/_build/html"
find . -name '*.html' -exec sed -i 's/<html lang="en"/<html lang="zh-CN"/' {} +
if [ -n "$1" ]; then
  for f in sitemap.xml robots.txt; do
    [ -f "$f" ] && sed -i "s#http://localhost:3000#$1#g" "$f"
  done
fi
