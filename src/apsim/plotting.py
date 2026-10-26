"""全书统一的 matplotlib 样式：英文和数字用 Arial，中文用系统可用的黑体类字体。"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimHei",
    "Arial Unicode MS",
)

# dataviz 技能参考调色板（浅色模式）前五个分类色槽，按固定顺序使用：
# 蓝、橙、青绿、黄、洋红。
_COLOR_CYCLE: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def _available_cjk_font() -> str | None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in CJK_FONT_CANDIDATES if name in installed), None)


def setup() -> str | None:
    """设置全局作图样式，返回所选中文字体名。"""
    chosen = _available_cjk_font()
    if chosen is None:
        warnings.warn("未找到中文字体，图中的中文可能无法显示", UserWarning, stacklevel=2)
    sans = ["Arial"] + ([chosen] if chosen else []) + ["DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "axes.unicode_minus": False,
            "axes.prop_cycle": plt.cycler(color=list(_COLOR_CYCLE)),
            "lines.linewidth": 2.0,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.figsize": (7.0, 4.0),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return chosen
