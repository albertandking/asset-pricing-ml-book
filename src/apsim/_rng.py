"""随机子流的统一派生规则（market 与 consumption 共用）。"""

from __future__ import annotations

import numpy as np


def substreams(seed: int | np.random.Generator, n: int) -> list[np.random.Generator]:
    """由 seed 派生 n 个相互独立的子流；新增子流只在末尾追加，不扰动已有子流。"""
    if isinstance(seed, np.random.Generator):
        # 每次调用只从父生成器取 1 个子键，再由它派生全部子流。若直接 seed.spawn(n)，
        # 子流数改变时，父生成器被取走的键数随之改变，同一生成器后续调用的结果就会全部改变。
        return list(seed.spawn(1)[0].spawn(n))
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        return [np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(n)]
    raise TypeError(
        "seed 须为非负整数或 np.random.Generator；"
        "确实需要不可复现的随机数时，显式传入 seed=np.random.default_rng()"
    )
