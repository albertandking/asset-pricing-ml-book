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

# 3. 已移动页面的旧网址：生成跳转页，避免外部链接失效（格式：旧路径 新路径，相对网站根目录）
while read -r old new; do
  [ -z "$old" ] && continue
  mkdir -p "$old"
  cat > "$old/index.html" <<HTML
<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>页面已移动</title>
<link rel="canonical" href="../$new/"><meta http-equiv="refresh" content="0; url=../$new/">
</head><body><p>本页已移动到 <a href="../$new/">新地址</a>。</p></body></html>
HTML
done <<'MOVED'
ch00-regressions primer-regressions
MOVED
