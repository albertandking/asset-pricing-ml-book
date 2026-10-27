"""前向引用台账：找出正文中指向尚未写出的章节或附录的引用，生成 docs/promises.md。

用途
----
书是逐章写作的，前面的章节常常写“见第 10 章”“第 16、17 章详细讨论”“附录 A”，
这些都是对后文的承诺。本脚本扫描 ``book/**/*.md`` 的正文，把指向尚未写出的
章节或附录的引用逐条列出，按被引用的章号分组，写入 ``docs/promises.md``。
写每一章之前先运行本脚本，逐条核对指向本章的承诺：在本章兑现，或改写原处的
说法；本章写完后重新生成台账（设计文档第 7 节第 16 条）。

运行方式（在项目根目录下）::

    uv run python scripts/promises.py

规则
----
- 跳过页首元数据（frontmatter）与围栏代码块（````` ``` ````` 或 ``~~~`` 开头，含
  ``code-cell`` 与 ``math``）；``:::`` 指令（习题、解答等）中的文字照常扫描。
- 识别的写法：“第 10 章”“第 16、17 章”“第 1～4 章”（按范围展开）“第 6 和 13 章”，
  以及“附录 A”“附录 A、B”。
- “第 N 章　标题”（章号后紧跟全角空格）是章节标题的罗列（如前言的写作进度表），
  不是承诺，不列入台账。
- 已写出的章节以 ``book/myst.yml`` 目录中存在的文件为准：读取每个文件的一级标题，
  “第 N 章”“附录 X”开头的记为已写出；标题中没有时，按文件名 ``chNN-`` 或
  ``appendix-x`` 判断。

只用标准库。
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
OUTPUT = ROOT / "docs" / "promises.md"

# 章号列表：数字之间用顿号、逗号、和/与/及/或连接，或用浪纹线等表示范围
_SEP = r"\s*(?:[、，,]|和|与|及|或|至|到|[～~–—-])\s*(?:第\s*)?"
CHAPTER_RE = re.compile(r"第\s*(\d+(?:" + _SEP + r"\d+)*)\s*章(?!　)")
APPENDIX_RE = re.compile(r"附录\s*([A-Z](?:\s*[、，,和与及或～~–—-]\s*[A-Z])*)(?![A-Za-z　])")
_TOKEN_RE = re.compile(r"\d+|[A-Z]|[～~–—-]|至|到")
_RANGE = set("～~–—-") | {"至", "到"}
_SENTENCE_END = "。！？；"
CONTEXT_WIDTH = 80


def _expand(tokens: list[str], convert) -> list:
    """把 ['1', '～', '4', '6'] 展开为 [1, 2, 3, 4, 6]；convert 把单个记号转为整数序号。"""
    values: list[int] = []
    pending_range = False
    for tok in tokens:
        if tok in _RANGE:
            pending_range = True
            continue
        v = convert(tok)
        if pending_range and values:
            values.extend(range(values[-1] + 1, v + 1))
        else:
            values.append(v)
        pending_range = False
    return values


def parse_refs(text: str) -> list[tuple[str, int | str, int, int]]:
    """解析一段文字中的章节与附录引用。

    返回 ``(kind, key, start, end)`` 的列表：``kind`` 为 ``"chapter"`` 或 ``"appendix"``，
    ``key`` 为章号（int）或附录字母（str），``start``、``end`` 为匹配在 ``text`` 中的位置。
    例如“第 16、17 章”给出 16 与 17，“第 1～4 章”给出 1、2、3、4，“附录 A”给出 ``"A"``。
    """
    refs: list[tuple[str, int | str, int, int]] = []
    for m in CHAPTER_RE.finditer(text):
        tokens = _TOKEN_RE.findall(m.group(1))
        for n in _expand(tokens, int):
            refs.append(("chapter", n, m.start(), m.end()))
    for m in APPENDIX_RE.finditer(text):
        tokens = _TOKEN_RE.findall(m.group(1))
        for n in _expand(tokens, ord):
            refs.append(("appendix", chr(n), m.start(), m.end()))
    refs.sort(key=lambda r: (r[2], sort_key((r[0], r[1]))))
    return refs


def sort_key(ref: tuple[str, int | str]) -> tuple[int, int | str]:
    """章在前（按章号），附录在后（按字母）。"""
    kind, key = ref
    return (0, key) if kind == "chapter" else (1, key)


def label(ref: tuple[str, int | str]) -> str:
    kind, key = ref
    return f"第 {key} 章" if kind == "chapter" else f"附录 {key}"


def prose_lines(text: str):
    """逐行给出 (行号, 行内容)，跳过 frontmatter 与围栏代码块。"""
    lines = text.split("\n")
    i = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                i = j + 1
                break
    fence: str | None = None
    for lineno in range(i, len(lines)):
        line = lines[lineno]
        stripped = line.lstrip()
        if fence is None:
            m = re.match(r"(`{3,}|~{3,})", stripped)
            if m:
                fence = m.group(1)
                continue
            if stripped.startswith("%"):  # MyST 注释
                continue
            yield lineno + 1, line
        elif stripped.startswith(fence) and stripped.strip(fence[0]).strip() == "":
            fence = None


def _clean(s: str) -> str:
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # 链接只留文字
    s = s.replace("**", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lstrip("-|> ").strip()


def context(line: str, start: int, end: int, width: int = CONTEXT_WIDTH) -> str:
    """截取匹配所在的句子；句子过长时以匹配为中心截到约 width 个字符。"""
    left = max(line.rfind(c, 0, start) for c in _SENTENCE_END) + 1
    rights = [p for p in (line.find(c, end) for c in _SENTENCE_END) if p != -1]
    right = min(rights) + 1 if rights else len(line)
    if "|" in line[left:right]:  # 表格行：只取匹配所在的单元格
        left = max(left, line.rfind("|", 0, start) + 1)
        nxt = line.find("|", end)
        right = min(right, nxt) if nxt != -1 else right
    sentence = line[left:right]
    s, e = start - left, end - left
    if len(sentence) > width:
        # 引用多在句末，窗口偏向匹配之前的文字：匹配之后最多保留 15 个字符
        b = min(len(sentence), e + 15)
        a = max(0, b - width)
        b = min(len(sentence), a + width)
        sentence = ("……" if a > 0 else "") + sentence[a:b] + ("……" if b < len(sentence) else "")
    return _clean(sentence)


def toc_files(myst_yml: Path) -> list[Path]:
    """myst.yml 目录中列出的文件（不解析 YAML，只取 file: 条目）。"""
    text = myst_yml.read_text(encoding="utf-8")
    return [myst_yml.parent / m.group(1).strip().strip("'\"")
            for m in re.finditer(r"^\s*-?\s*file:\s*(\S+)", text, flags=re.M)]


def written(files: list[Path]) -> set[tuple[str, int | str]]:
    """已写出的章节：目录中存在的文件，按一级标题或文件名判断章号。"""
    done: set[tuple[str, int | str]] = set()
    for f in files:
        if not f.exists():
            continue
        h1 = re.search(r"^#\s+(.+)$", f.read_text(encoding="utf-8"), flags=re.M)
        title = h1.group(1) if h1 else ""
        if m := re.match(r"第\s*(\d+)\s*章", title):
            done.add(("chapter", int(m.group(1))))
        elif m := re.match(r"附录\s*([A-Z])", title):
            done.add(("appendix", m.group(1)))
        elif m := re.match(r"ch(\d+)-", f.name):
            done.add(("chapter", int(m.group(1))))
        elif m := re.match(r"appendix-?([a-z])", f.name, flags=re.I):
            done.add(("appendix", m.group(1).upper()))
    return done


def collect(book: Path = BOOK) -> dict[tuple[str, int | str], list[tuple[str, int, str]]]:
    """扫描 book/**/*.md，返回 {被引用的章节: [(文件, 行号, 上下文), ...]}。"""
    done = written(toc_files(book / "myst.yml"))
    promises: dict[tuple[str, int | str], list[tuple[str, int, str]]] = defaultdict(list)
    for path in sorted(book.rglob("*.md")):
        rel = path.relative_to(book)
        if rel.parts[0].startswith("_"):  # _build、_static 等
            continue
        name = path.relative_to(ROOT).as_posix()
        for lineno, line in prose_lines(path.read_text(encoding="utf-8")):
            seen: set[tuple[str, int | str]] = set()
            for kind, key, s, e in parse_refs(line):
                ref = (kind, key)
                # “进入第 N 章”是阅读导航，不是要在第 N 章兑现的承诺
                if line[max(0, s - 2):s] == "进入":
                    continue
                if ref in done or ref in seen:
                    continue
                seen.add(ref)
                promises[ref].append((name, lineno, context(line, s, e)))
    return dict(promises)


def render(promises: dict[tuple[str, int | str], list[tuple[str, int, str]]]) -> str:
    total = sum(len(v) for v in promises.values())
    out = [
        "# 前向引用台账",
        "",
        "本文件由 `scripts/promises.py` 生成，请勿手工修改。重新生成：`uv run python scripts/promises.py`。",
        "",
        "正文中指向尚未写出的章节或附录的引用，按被引用的章号分组，每条给出文件与行号（如 `book/glossary.md:30`），破折号后是所在句子的上下文。"
        "写每一章之前，逐条核对指向本章的承诺：在本章兑现，或改写原处的说法；本章写完后重新生成本文件。",
        "",
        f"共 {total} 条，涉及 {len(promises)} 个章节或附录。",
        "",
    ]
    for ref in sorted(promises, key=sort_key):
        items = promises[ref]
        out.append(f"## {label(ref)}（{len(items)} 条）")
        out.append("")
        for name, lineno, ctx in items:
            out.append(f"- `{name}:{lineno}` — {ctx}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    promises = collect()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(promises), encoding="utf-8", newline="\n")
    total = sum(len(v) for v in promises.values())
    top = sorted(promises.items(), key=lambda kv: (-len(kv[1]), sort_key(kv[0])))[:5]
    print(f"已写入 {OUTPUT.relative_to(ROOT).as_posix()}：共 {total} 条")
    print("被引用最多：" + "，".join(f"{label(r)} {len(v)} 条" for r, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
