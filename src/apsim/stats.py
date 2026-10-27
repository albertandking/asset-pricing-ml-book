"""第 0 章用到的估计工具：最小二乘、交互项设计矩阵与 Fama–MacBeth 程序。

记号与数组约定见 book/notation.md 的“公式与代码的对应”：数据数组“观测在行”，一维数组代表公式中的列向量。
本书的写法规则（设计文档第 7 节第 15 条）：一个估计工具第一次在书中出现时，在页面中内联实现并讲解，
随后用 assert 断言内联实现与这里的同名函数结果一致；之后的页面直接调用本模块。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


def ols(X, y) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘系数与同方差标准误。

    对应书中第 0 章“模拟实验：用交互回归恢复 Γ”一节（`ch00-experiments.md`，标签 sec-ch0-interact）：
    系数 (X'X)⁻¹X'y 用 `lstsq` 求得，条件协方差矩阵 σ̂²(X'X)⁻¹ 中的 (X'X)⁻¹ 用 `solve(X'X, I)` 求得，
    σ̂² = 残差平方和 / (n − p)。

    参数
    - X：(n, p) 设计矩阵，每行是一个观测的解释变量（公式中列向量的转置）。
    - y：(n,) 被解释变量。

    返回 `(coef, se)`：系数 (p,) 与同方差标准误 (p,)。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)                               # (p,)
    resid = y - X @ coef                                                       # (n, p) @ (p,) -> (n,)
    s2 = resid @ resid / (len(y) - X.shape[1])                                 # σ̂_ε²
    XtX_inv = np.linalg.solve(X.T @ X, np.eye(X.shape[1]))                     # (X'X)⁻¹：(p, p)
    return coef, np.sqrt(np.diag(s2 * XtX_inv))


def interaction_design(market) -> tuple[np.ndarray, np.ndarray]:
    """“特征 × 因子”交互项设计矩阵 X 与期初特征矩阵 Z。

    对应书中的 Kronecker 积回归式（标签 eq-ch0-kron；`ch00-experiments.md`，标签 sec-ch0-interact）：
    第 (i, t) 个观测的行是 (z̃_{i,t-1} ⊗ f_t)'，元素按 z̃_j f_k 排列，j = 0, ..., L 在外层、k = 1, ..., K 在内层，
    与 vec(Γ') 的排列顺序一致，所以 `X @ gamma.ravel()` 等于每个观测的 β_{i,t}' f_t。

    参数
    - market：`SimulatedMarket`（或具有同样 `panel`、`factors`、`params.n_chars` 的对象）。

    返回 `(X, Z)`：X 为 (TN, (L+1)K)，Z 为 (TN, L)，行与 `market.panel` 的行一一对应。
    """
    pnl = market.panel
    n_chars = market.params.n_chars
    Z = pnl[[f"z{j}" for j in range(1, n_chars + 1)]].to_numpy()               # (TN, L)
    zz = np.column_stack([np.ones(len(pnl)), Z])                               # (TN, L+1)，每行 z̃'
    ff = market.factors.reindex(pnl.index.get_level_values("period")).to_numpy()   # (TN, K)，每行 f_t'
    X = (zz[:, :, None] * ff[:, None, :]).reshape(len(pnl), -1)                # (TN, L+1, K) -> (TN, (L+1)K)
    return X, Z


class FamaMacBethResult(NamedTuple):
    """Fama–MacBeth 程序的结果，可以直接解包为 `coefs, mean, se`。"""

    coefs: np.ndarray
    """各期横截面回归的系数 (T, p)：第 t 行为 ĝ_t'。"""
    mean: np.ndarray
    """系数的时间平均 ḡ = T⁻¹ Σ_t ĝ_t，(p,)。"""
    se: np.ndarray
    """FM 标准误 (p,)：第 j 个分量为 σ̂(ĝ_{j,t}) / √T，σ̂ 为除以 T − 1 的时间序列标准差。"""


def fama_macbeth(X, y) -> FamaMacBethResult:
    """Fama–MacBeth 程序：逐期横截面回归，再对系数求时间平均与时间序列标准误。

    对应书中 0.2 节（续）“Fama–MacBeth 程序”的三步（`ch00-regressions.md`，标签 sec-ch0-regs）：
    1. 每期把 y_t 对 X_t 做横截面最小二乘回归，得到 ĝ_t；
    2. 点估计 ḡ = T⁻¹ Σ_t ĝ_t；
    3. 标准误 se(ḡ_j) = σ̂(ĝ_{j,t}) / √T，σ̂(ĝ_{j,t}) = [(T − 1)⁻¹ Σ_t (ĝ_{j,t} − ḡ_j)²]^{1/2}。

    参数
    - X：(T, N, p) 解释变量，`X[t]` 为第 t+1 期的 (N, p) 横截面设计矩阵（数组下标从 0 起），
      需要截距时自行加入全 1 列；每期的股票数 N 相同。
    - y：(T, N) 被解释变量，`y[t]` 为第 t+1 期的 N 个观测。

    返回 `FamaMacBethResult(coefs, mean, se)`，形状分别为 (T, p)、(p,)、(p,)。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X 须为 (T, N, p) 的 3 维数组，而不是 {X.ndim} 维")
    if y.shape != X.shape[:2]:
        raise ValueError(f"y 的形状须为 {X.shape[:2]}，而不是 {y.shape}")
    n_periods = X.shape[0]
    if n_periods < 2:
        raise ValueError("至少需要两期才能计算 Fama–MacBeth 标准误")
    coefs = np.array([np.linalg.lstsq(X[t], y[t], rcond=None)[0] for t in range(n_periods)])   # (T, p)
    mean = coefs.mean(axis=0)                                                  # ḡ: (p,)
    se = coefs.std(axis=0, ddof=1) / np.sqrt(n_periods)                        # 逐分量：σ̂(ĝ_{j,t}) / √T
    return FamaMacBethResult(coefs, mean, se)
