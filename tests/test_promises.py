"""scripts/promises.py 的章号解析与正文扫描。"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "promises.py"
_spec = importlib.util.spec_from_file_location("promises", _PATH)
promises = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(promises)


def keys(text):
    return [(kind, key) for kind, key, _, _ in promises.parse_refs(text)]


def test_single_chapter():
    assert keys("详见第 10 章。") == [("chapter", 10)]


def test_chapter_list_split():
    assert keys("见第 16、17 章") == [("chapter", 16), ("chapter", 17)]
    assert keys("第 1、4、10 章") == [("chapter", 1), ("chapter", 4), ("chapter", 10)]


def test_chapter_range_expanded():
    assert keys("第 1～4 章") == [("chapter", n) for n in (1, 2, 3, 4)]


def test_appendix():
    assert keys("数学工具见附录 A。") == [("appendix", "A")]
    assert keys("附录 A、B") == [("appendix", "A"), ("appendix", "B")]


def test_title_listing_is_not_a_promise():
    # “第 N 章　标题”是章节标题的罗列，不是承诺
    assert keys("| 第 1 章　消费、欧拉方程与随机贴现因子 | 未开始 |") == []
    assert keys("附录 A　数学工具") == []


def test_prose_lines_skip_frontmatter_and_code():
    text = "---\ntitle: 第 3 章\n---\n正文见第 5 章。\n```{code-cell} python\n# 第 6 章\n```\n又见第 7 章。\n"
    lines = [(n, line) for n, line in promises.prose_lines(text) if line]
    assert [n for n, _ in lines] == [4, 8]
    assert [k for _, line in lines for k in keys(line)] == [("chapter", 5), ("chapter", 7)]


def test_context_is_the_sentence():
    line = "前一句。Shanken 修正见第 10 章，这里不展开。后一句。"
    (_, _, s, e), = promises.parse_refs(line)
    assert promises.context(line, s, e) == "Shanken 修正见第 10 章，这里不展开。"
