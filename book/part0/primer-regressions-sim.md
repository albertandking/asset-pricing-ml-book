---
short_title: 导读（续）：模拟实验
kernelspec:
  name: python3
  display_name: Python 3
---

(sec-ch0-regs-sim)=
# 导读：几种回归（续）——同一份数据上的模拟实验与业界口径

**本页的定位。** 本页接着[导读：几种回归](#sec-ch0-regs)，在同一份模拟数据上跑那里介绍的四种回归，再列出第 0 章中已经用到这些回归的地方，以及它们与业界口径的对应。四种回归的式子、各项的维度与设计矩阵的形状，见上一页对[时间序列回归](#primer-unpack-ts)、[横截面回归](#primer-unpack-cs)、[Fama–MacBeth 程序](#primer-unpack-fm)、[混合回归](#primer-unpack-pool)与[权重矩阵 $\mathbf W_t$](#primer-unpack-w)的拆解，本页不再重复。

**本页准备。** 本书每个页面的代码在独立的内核中运行。下面的单元导入本页用到的模块；本页的模拟数据都在本页生成。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market, stats
```

#### 模拟实验：同一份数据上的四种回归

下面在同一份模拟数据上跑全部四种回归。参数取默认值（$N = 500$、$T = 240$、默认 $\bv\Gamma$、$\bv\theta = \mathbf 0$），种子为 2026。默认 $\bv\Gamma$ 下 $\bv\beta_{i,t} = (1,\ 0.5\, z_{1,i,t-1},\ 0.5\, z_{2,i,t-1})'$，各问题的理论值都可以直接写出。

**(a) 时间序列回归。** 对每只股票把 $r_{i,t}$ 对 $(1, \mathbf f_t')'$ 回归。由[习题 %s](#ex-ch0-ts)，斜率的总体值是 $\E[\bv\beta_{i,t}]$，即 $\bv\Gamma$ 第 0 行的转置 $(1, 0, 0)'$，所以 $\hat{\bv\beta}_i$ 的横截面平均应接近它；而单只股票的 $\hat{\bv\beta}_i$ 应接近这只股票在样本期内的平均载荷 $T^{-1}\sum_t\bv\beta_{i,t}$，误差的量级由回归标准误给出。后一句是给定这只股票特征路径的条件说法，成立的前提是载荷与因子的实现值独立（本模拟器满足；真实数据中，例如条件 CAPM 里载荷随市场状态变化，就会破坏它）；$T \to \infty$ 时特征路径的平均趋于 0，平均载荷趋于 $\E[\bv\beta_{i,t}] = (1, 0, 0)'$，两种说法一致。代码用一次 `lstsq` 解出 $N$ 个回归：右端是 $T\times N$ 的数组时，`lstsq` 对每一列分别求解，系数数组的第 $i$ 列为 $(\hat a_i, \hat{\bv\beta}_i')'$（$N$ 个回归并排的矩阵写法见[拆解](#primer-unpack-ts)）。

(primer-unpack-avgbeta)=
**拆开讲平均载荷 $T^{-1}\sum_t\bv\beta_{i,t}$。** 它是 $K\times 1$ 的列向量，第 $k$ 个分量 $T^{-1}\sum_t\beta_{k,i,t}$ 是股票 $i$ 对第 $k$ 个因子的载荷在样本期 $T$ 个月内的平均。默认 $\bv\Gamma$ 下逐分量写出，

```{math}
:enumerated: false
\frac{1}{T}\sum_{t=1}^{T}\bv\beta_{i,t} = \frac{1}{T}\sum_{t=1}^{T}\begin{pmatrix} 1 \\ 0.5\, z_{1,i,t-1} \\ 0.5\, z_{2,i,t-1} \end{pmatrix} = \begin{pmatrix} 1 \\ 0.5\, \bar z_{1,i} \\ 0.5\, \bar z_{2,i} \end{pmatrix} , \qquad \bar z_{j,i} = \frac{1}{T}\sum_{t=1}^{T} z_{j,i,t-1} ,
```

其中 $\bar z_{j,i}$ 是股票 $i$ 的特征 $j$ 在样本期内的路径均值（没有单位，以横截面标准差计），代码中的 `z1_bar_rg` 就是 $N$ 只股票的 $\bar z_{1,i}$。例如 $z_1$ 路径均值为 0.7 的股票，样本期内对因子 2 的平均载荷为 0.35。**特例**：载荷不随时间变化时，平均载荷就是 $\bv\beta_i$ 本身；$T \to \infty$ 时路径均值趋于 0，平均载荷趋于 $\E[\bv\beta_{i,t}] = (1, 0, 0)'$。**数据矩阵**：$N$ 只股票的平均载荷按行堆叠成 $N\times K$ 矩阵（`beta_avg_rg`，由形状为 `(T, N, K)` 的真实载荷 `B_rg` 沿时间平均得到），与载荷估计值的矩阵 $\hat{\mathbf B}$（`beta_hat_rg`，[拆解](#primer-unpack-cs)）形状相同，下面逐元素比较二者。

```{code-cell} python
# 同一份模拟数据：默认参数（N = 500、T = 240、θ = 0），下面四种回归都用它
rg = simulate_market(MarketParams(), seed=2026)
T_rg, N_rg, K_rg = rg.params.n_periods, rg.params.n_stocks, rg.params.n_factors
mu_rg = np.asarray(rg.params.factor_mean)                                  # μ: (K,)
F_rg = rg.factors.to_numpy()                                               # F: (T, K)，第 t 行为 f_t'
R_rg = rg.panel["exret"].unstack("stock").to_numpy()                       # (T, N)：第 t 行为 r_t'，第 i 列为股票 i 的时间序列
B_rg = rg.panel[["beta1", "beta2", "beta3"]].to_numpy().reshape(T_rg, N_rg, K_rg)   # (TN, K) -> (T, N, K)：真实载荷
Z12_rg = rg.panel[["z1", "z2"]].to_numpy().reshape(T_rg, N_rg, 2)          # (TN, 2) -> (T, N, 2)：z_{1,i,t-1}、z_{2,i,t-1}

# (a) 时间序列回归：N 个回归共用设计矩阵 [1_T, F]，一次解出
D_rg = np.column_stack([np.ones(T_rg), F_rg])                              # (T, 1+K)
coef_ts_rg, *_ = np.linalg.lstsq(D_rg, R_rg, rcond=None)                   # (T, 1+K) 对 (T, N) 逐列求解 -> (1+K, N)
beta_hat_rg = coef_ts_rg[1:].T                                             # (K, N) -> (N, K)：第 i 行为 β̂_i'
resid_ts_rg = R_rg - D_rg @ coef_ts_rg                                     # (T, 1+K) @ (1+K, N) -> (T, N)：u_{i,t} 的估计
s2_ts_rg = (resid_ts_rg ** 2).sum(axis=0) / (T_rg - 1 - K_rg)              # (N,)：各股票的残差方差 σ̂²
DtD_inv_rg = np.linalg.solve(D_rg.T @ D_rg, np.eye(1 + K_rg))              # (1+K, 1+K)：([1_T, F]'[1_T, F])⁻¹，各股票共用
se_beta_rg = np.sqrt(s2_ts_rg[:, None] * np.diag(DtD_inv_rg)[1:])          # 广播 (N, 1) * (K,) -> (N, K)：β̂ 的标准误
beta_avg_rg = B_rg.mean(axis=0)                                            # (T, N, K) -> (N, K)：样本期内的平均载荷
dev_rg = (beta_hat_rg - beta_avg_rg) / se_beta_rg                          # (N, K)：以平均载荷为中心的标准化偏离

print("β̂_i 的横截面平均：", beta_hat_rg.mean(axis=0).round(3), "；E[β] = Γ 第 0 行：", rg.gamma[0])
z1_bar_rg = Z12_rg[:, :, 0].mean(axis=0)                                   # (N,)：各股票 z1 路径的样本均值
i_hi, i_lo = int(np.argmax(z1_bar_rg)), int(np.argmin(z1_bar_rg))
for i in (i_hi, i_lo):
    print(f"股票 {i}：z1 路径均值 {z1_bar_rg[i]:+.3f}")
    print(pd.DataFrame([beta_avg_rg[i], beta_hat_rg[i], se_beta_rg[i]], index=["平均载荷", "估计值 β̂_i", "标准误"],
                       columns=["β1", "β2", "β3"]).round(3).to_string())
order_rg = np.argsort(z1_bar_rg)
groups_rg = {"z1 均值最低的 50 只": order_rg[:50], "z1 均值最高的 50 只": order_rg[-50:]}
for name, idx in groups_rg.items():
    print(f"{name}：β̂2 的组内平均 {beta_hat_rg[idx, 1].mean():+.3f}，平均载荷 β2 的组内平均 {beta_avg_rg[idx, 1].mean():+.3f}")
print(f"全部 N×K 个标准化偏离：标准差 {dev_rg.std():.3f}，绝对值超过 2 的比例 {np.mean(np.abs(dev_rg) > 2):.3f}")

# 横截面平均：单只股票的标准误约 0.13～0.22，500 只平均后约 0.01，再计入各股票共同的特征与冲击，取 0.05 为保守阈值
assert np.allclose(beta_hat_rg.mean(axis=0), rg.gamma[0], atol=0.05)
assert np.isclose(beta_avg_rg[i_hi, 1], 0.5 * z1_bar_rg[i_hi]) and beta_avg_rg[i_hi, 1] > 0.3 > -0.3 > beta_avg_rg[i_lo, 1]
assert np.all(np.abs(dev_rg[[i_hi, i_lo]]) < 3)                            # 两只股票：每个分量都在 3 倍标准误以内
for idx in groups_rg.values():                                             # 每组 50 只：组内平均的标准误约 0.2/√50 ≈ 0.03
    se_grp = np.sqrt(np.sum(se_beta_rg[idx, 1] ** 2)) / len(idx)
    assert abs(beta_hat_rg[idx, 1].mean() - beta_avg_rg[idx, 1].mean()) < 3 * se_grp
    assert abs(beta_hat_rg[idx, 1].mean()) > 0.15                          # 两组的 β̂2 明显偏离 0，且方向与 z1 一致
assert beta_hat_rg[order_rg[-50:], 1].mean() > 0 > beta_hat_rg[order_rg[:50], 1].mean()
assert abs(dev_rg.std() - 1) < 0.1                                         # N×K = 1500 个偏离，标准差的抽样误差约 0.02
```

$\hat{\bv\beta}_i$ 的横截面平均与 $(1, 0, 0)'$ 的差别在 0.05 以内（由 `assert` 保证）。上面挑出 $z_1$ 路径均值最高与最低的两只股票：它们在样本期内对因子 2 的平均载荷分别高于 0.3 与低于 $-0.3$，正好是 $0.5$ 乘以 $z_1$ 的路径均值；估计值 $\hat{\bv\beta}_i$ 的每个分量与平均载荷的偏离都在 3 倍标准误以内（均由 `assert` 保证，具体数值见上方输出）。但单只股票的标准误约为 0.2，与平均载荷在股票之间的差别是同一个量级，所以单只股票的 $\hat{\bv\beta}_i$ 只能粗略地反映它的平均载荷。按 $z_1$ 均值分组后，噪声被平均掉：最高、最低两组 50 只股票的 $\hat\beta_2$ 组内平均明显偏离 0，方向与 $z_1$ 一致，并且与组内的平均载荷相差在 3 倍标准误以内。全部 1500 个标准化偏离的标准差与 1 相差不到 0.1（均由 `assert` 保证）：$\hat{\bv\beta}_i$ 以样本期平均载荷为中心，回归标准误正确地度量了它的误差。

**(b) 横截面回归与 (c) Fama–MacBeth。** 每期把 $\mathbf r_t$ 对 $[\mathbf 1_N, \mathbf z_1, \mathbf z_2]$（$N\times 3$，即 $\tilde{\mathbf Z}_{t-1}$ 的前三列）回归，这与[因子、特征、载荷的关系](#sec-ch0-fcl)中的回归相同，第 $t$ 期的系数应约等于 $(f_{1,t},\ 0.5 f_{2,t},\ 0.5 f_{3,t})$，那里已详细验证，这里只用相关系数简要复核。新的内容是 Fama–MacBeth：$\bar{\mathbf g}$ 应约等于 $\bv\Gamma\bv\mu$ 的前三个分量 $(\mu_1,\ 0.5\mu_2,\ 0.5\mu_3) = (0.006,\ 0.0015,\ 0.002)$。更准确地说，$\hat{\mathbf g}_t$ 等于 $\bv\Gamma\mathbf f_t$ 加上横截面噪声，所以 $\bar{\mathbf g}$ 约等于 $\bv\Gamma\bar{\mathbf f}$（$\bar{\mathbf f}$ 为本样本的因子均值），二者只差平均后的横截面噪声，量级为 $\sigma_\varepsilon/\sqrt{NT}$；$\bar{\mathbf g}$ 与 $\bv\Gamma\bv\mu$ 的差别则主要来自因子样本均值的抽样误差，FM 标准误因而约为 $\hat{\mathbf g}_t$ 中因子部分的标准差除以 $\sqrt T$，即 $\bv\Gamma$ 第 $j$ 行与 $\bv\Sigma_f$ 构成的 $[\bv\Gamma\bv\Sigma_f\bv\Gamma']_{jj}^{1/2}/\sqrt T$。代码同时记下每期回归中每个系数的 OLS $t$ 值，供下文“与业界的对应”使用。Fama–MacBeth 的三步也收在 `apsim.stats.fama_macbeth` 中，代码单元用 `assert` 核对它与这里的内联实现结果相同，后面的页面直接调用它。

(primer-unpack-gsg)=
**拆开讲 $\bv\Gamma\bar{\mathbf f}$ 与 $[\bv\Gamma\bv\Sigma_f\bv\Gamma']_{jj}^{1/2}/\sqrt T$。** $\bar{\mathbf f}$ 是 $K\times 1$ 的因子样本均值（[拆解](#ch0-unpack-bhat)），$\bv\Gamma\bar{\mathbf f}$ 是 $(L+1)\times K$ 乘 $K\times 1$，得 $(L+1)\times 1$，与 $\bv\Gamma\bv\mu$（[拆解](#ch0-unpack-gmu)）的形式相同，只是把总体均值 $\bv\mu$ 换成了本样本的均值。$\hat{\mathbf g}_t$ 的因子部分是 $\bv\Gamma\mathbf f_t$（[拆解](#ch0-unpack-g)），它的协方差矩阵是

```{math}
:enumerated: false
\underbrace{\Var(\bv\Gamma\mathbf f_t)}_{(L+1)\times(L+1)} = \underbrace{\bv\Gamma}_{(L+1)\times K}\,\underbrace{\bv\Sigma_f}_{K\times K}\,\underbrace{\bv\Gamma'}_{K\times(L+1)} , \qquad
[\bv\Gamma\bv\Sigma_f\bv\Gamma']_{jj} = \sum_{k=1}^{K}\sum_{l=1}^{K}\Gamma_{j,k}\Sigma_{kl}\Gamma_{j,l} ,
```

其中 $\Sigma_{kl}$ 是 $\bv\Sigma_f$ 的 $(k, l)$ 元素（$\bv\Sigma_f$ 的结构见[拆解](#ch0-unpack-ts)）。第 $j$ 个对角元是第 $j$ 个系数中因子部分的方差，因子互不相关时化为 $\sum_k\Gamma_{j,k}^2\sigma_k^2$；开平方得标准差（单位与收益相同），再除以 $\sqrt T$，就是 $T$ 期平均的标准误。默认参数下 $\sigma_1 = 0.05$、$\sigma_2 = \sigma_3 = 0.03$，常数项一行为 $1 \times 0.05/\sqrt{240} \approx 0.32\%$，$z_1$、$z_2$ 两行都是 $0.5 \times 0.03/\sqrt{240} \approx 0.097\%$；下面代码中的 `se_factor_rg` 计算的就是这三个数（`G3_rg` 是 $\bv\Gamma$ 的前三行，形状 `(3, K)`）。**特例**：$L = 0$、$K = 1$ 时它一般是 $|\Gamma_{0,1}|\sigma_1/\sqrt T$；再有 $\Gamma_{0,1} = 1$ 时就是 $\sigma_1/\sqrt T$，即因子样本均值的标准误（误区二）。另一个量 $\sigma_\varepsilon/\sqrt{NT}$ 是标量：$\sigma_\varepsilon = 0.10$ 是异质性冲击的月度标准差，除以 $\sqrt N$ 是一期横截面噪声的量级，再除以 $\sqrt T$ 是 $T$ 期平均后的量级，默认参数下约为 $0.10/\sqrt{120000} \approx 0.03\%$。

```{code-cell} python
# 拆开讲 [ΓΣ_fΓ']_jj：默认 Γ 的前三行、对角的 Σ_f 下，三个对角元分别为 σ_1²、(0.5σ_2)²、(0.5σ_3)²
var_fp = np.diag(rg.gamma[:3] @ rg.params.factor_cov @ rg.gamma[:3].T)   # (3, K) (K, K) (K, 3) -> 取对角 (3,)
print("因子部分的标准差除以 √T：", (np.sqrt(var_fp / T_rg) * 100).round(3), "（%）；σ_ε/√(NT) =", f"{0.10 / np.sqrt(N_rg * T_rg):.4%}")
assert np.allclose(var_fp, [0.05 ** 2, (0.5 * 0.03) ** 2, (0.5 * 0.03) ** 2])
assert np.allclose(np.sqrt(var_fp / T_rg), np.array([0.05, 0.015, 0.015]) / np.sqrt(240))
```

```{code-cell} python
# (b) 横截面回归：每期 r_t 对 [1_N, z1, z2]，并记录每个系数在该期回归中的 OLS t 值
X_rg = np.concatenate([np.ones((T_rg, N_rg, 1)), Z12_rg], axis=2)          # (T, N, 3)：第 t 期为 [1_N, z1, z2]
G_rg, tstat_rg = np.empty((T_rg, 3)), np.empty((T_rg, 3))                  # 第 t 行为 ĝ_t' 与对应的 t 值
for t in range(T_rg):
    g_t, *_ = np.linalg.lstsq(X_rg[t], R_rg[t], rcond=None)                # (N, 3) 对 (N,) -> (3,)
    e_t = R_rg[t] - X_rg[t] @ g_t                                          # (N, 3) @ (3,) -> (N,)
    cov_t = e_t @ e_t / (N_rg - 3) * np.linalg.solve(X_rg[t].T @ X_rg[t], np.eye(3))   # (3, 3)：σ̂²(X'X)⁻¹
    G_rg[t], tstat_rg[t] = g_t, g_t / np.sqrt(np.diag(cov_t))
corr_b_rg = np.array([np.corrcoef(G_rg[:, k], F_rg[:, k])[0, 1] for k in range(3)])   # ĝ_{k,t} 与 f_{k+1,t}；乘 0.5 不改变相关系数
print("(b) 各期系数与 (f1, 0.5 f2, 0.5 f3) 的相关系数：", corr_b_rg.round(3))
assert np.all(corr_b_rg > 0.9)                                             # 与 sec-ch0-fcl 中的阈值相同

# (c) Fama–MacBeth：系数的时间平均与时间序列标准误
gbar_rg = G_rg.mean(axis=0)                                                # ḡ: (3,)
se_fm_rg = G_rg.std(axis=0, ddof=1) / np.sqrt(T_rg)                        # 逐分量：σ̂(ĝ_{j,t}) / √T
fm_lib_rg = stats.fama_macbeth(X_rg, R_rg)                                 # 同一程序的 apsim.stats 版本
assert np.allclose(fm_lib_rg.coefs, G_rg) and np.allclose(fm_lib_rg.mean, gbar_rg) and np.allclose(fm_lib_rg.se, se_fm_rg)
G3_rg = rg.gamma[:3]                                                       # Γ 的前三行（常数、z1、z2）：(3, K)
truth_rg = G3_rg @ mu_rg                                                   # Γμ：(3, K) @ (K,) -> (3,)
sample_rg = G3_rg @ F_rg.mean(axis=0)                                      # Γ f̄：(3, K) @ (K,) -> (3,)
se_factor_rg = np.sqrt(np.diag(G3_rg @ rg.params.factor_cov @ G3_rg.T) / T_rg)   # (3, K) (K, K) (K, 3) -> 取对角 (3,)
print(pd.DataFrame({"理论值 Γμ": truth_rg, "本样本 Γf̄": sample_rg, "FM 估计 ḡ": gbar_rg, "FM 标准误": se_fm_rg,
                    "(ḡ − Γμ)/标准误": (gbar_rg - truth_rg) / se_fm_rg, "FM t 值": gbar_rg / se_fm_rg,
                    "标准误 / 因子部分": se_fm_rg / se_factor_rg},
                   index=["常数", "z1", "z2"]).round(5).to_string())
noise_rg = rg.params.idio_vol / np.sqrt(N_rg * T_rg)                       # 横截面噪声平均 T 期后的量级 σ_ε/√(NT)
assert np.all(np.abs(gbar_rg - truth_rg) < 3 * se_fm_rg)                   # 与 Γμ 的偏离在 3 倍 FM 标准误以内
assert np.all(np.abs(gbar_rg - sample_rg) < 4 * noise_rg)                  # ḡ 与 Γf̄ 只差平均后的横截面噪声
# 理论比值：ĝ_t 的方差 = 因子部分 + 横截面噪声 σ_ε²/N（Z̃'Z̃/N ≈ I），比值约为 √(1 + (σ_ε²/N) / V)，V 为因子部分的方差
ratio_theory_rg = np.sqrt(1 + rg.params.idio_vol ** 2 / N_rg / (se_factor_rg ** 2 * T_rg))   # 约 (1.004, 1.043, 1.043)
print("标准误 / 因子部分的理论值：", ratio_theory_rg.round(3))
# T = 240 时，样本标准差的相对抽样误差约 1/√(2T) ≈ 0.046，阈值 0.12 约为 2.6 倍
assert np.all(np.abs(se_fm_rg / se_factor_rg - ratio_theory_rg) < 0.12)

# 每期回归的 t 值：业界“t 值序列”的统计口径
print(pd.DataFrame({"每期 |t| 的均值": np.abs(tstat_rg).mean(axis=0), "t > 2 的比例": (tstat_rg > 2).mean(axis=0),
                    "t < −2 的比例": (tstat_rg < -2).mean(axis=0)}, index=["常数", "z1", "z2"]).round(3).to_string())
assert np.all((tstat_rg[:, 1:] > 2).mean(axis=0) > 0.15) and np.all((tstat_rg[:, 1:] < -2).mean(axis=0) > 0.15)
assert np.all((np.abs(tstat_rg[:, 1:]) > 2).mean(axis=0) > 0.5)               # z1、z2：多数期 |t| > 2
```

三列系数与对应因子的相关系数都在 0.9 以上（由 `assert` 保证），与[因子、特征、载荷的关系](#sec-ch0-fcl)的结论一致。FM 估计 $\bar{\mathbf g}$ 与理论值 $\bv\Gamma\bv\mu$ 的偏离都在 3 倍 FM 标准误以内；它与本样本的 $\bv\Gamma\bar{\mathbf f}$ 更接近，差别不到 $4\sigma_\varepsilon/\sqrt{NT}$；FM 标准误与只由因子波动算出的 $[\bv\Gamma\bv\Sigma_f\bv\Gamma']_{jj}^{1/2}/\sqrt T$ 之比，与计入横截面噪声后的理论值（常数项约 1.004，$z_1$、$z_2$ 约 1.04）相差不到 0.12（均由 `assert` 保证）。后一点的含义是：在这个模拟中，FM 估计的不确定性主要来自因子收益的时间序列波动，横截面噪声 $\sigma_\varepsilon/\sqrt N$ 只占一小部分，增加股票数几乎不能缩小它，增加期数才能。常数项的 FM 估计几乎就是因子 1 的样本均值（只差平均后的横截面噪声），240 个月中它的标准误约为 $0.05/\sqrt{240} \approx 0.32\%$，与误区二算出的数字相同。

**(d) 混合回归。** 把全部 $NT$ 个观测放进一个 OLS，给出点估计与同方差标准误，并与 Fama–MacBeth 比较。设计矩阵 `Xp_rg` 的堆叠方式见上一页的[拆解](#primer-unpack-pool)。代码还核对上一页的恒等式 $\hat{\mathbf g}^{\mathrm P} = \sum_t\mathbf W_t\hat{\mathbf g}_t$（[拆解](#primer-unpack-w)）：`np.einsum("tni,tnj->tij", X, X)` 对每一期 $t$ 把 $n$（股票）求和，得到逐期的 $\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$（这里只取前三列）。最后核对上一页[三条结论](#primer-regs-conclusions)中的“结论 1”：只有一个特征、每期标准化时二者完全相等，有两个特征时则不再完全相等。

```{code-cell} python
# (d) 混合回归：把 (T, N, 3) 的设计数组摊平成 (TN, 3)，一次 OLS
Xp_rg = X_rg.reshape(T_rg * N_rg, 3)                                       # (T, N, 3) -> (TN, 3)，期在外层
yp_rg = R_rg.ravel()                                                       # (T, N) -> (TN,)，顺序与 Xp_rg 的行一致
gp_rg, *_ = np.linalg.lstsq(Xp_rg, yp_rg, rcond=None)                      # ĝ^P: (3,)
ep_rg = yp_rg - Xp_rg @ gp_rg                                              # (TN, 3) @ (3,) -> (TN,)
cov_pool_rg = ep_rg @ ep_rg / (len(yp_rg) - 3) * np.linalg.solve(Xp_rg.T @ Xp_rg, np.eye(3))   # σ̂²(Σ_t Z̃'Z̃)⁻¹：(3, 3)
se_pool_rg = np.sqrt(np.diag(cov_pool_rg))                                 # 同方差标准误：(3,)
ratio_rg = se_fm_rg / se_pool_rg
print(pd.DataFrame({"混合回归估计": gp_rg, "FM 估计": gbar_rg, "差 / FM 标准误": (gp_rg - gbar_rg) / se_fm_rg,
                    "同方差标准误": se_pool_rg, "FM 标准误": se_fm_rg, "FM 标准误 / 同方差标准误": ratio_rg},
                   index=["常数", "z1", "z2"]).round(5).to_string())

# 恒等式 ĝ^P = Σ_t W_t ĝ_t，W_t = (Σ_s Z̃_{s-1}'Z̃_{s-1})⁻¹ Z̃_{t-1}'Z̃_{t-1}
XtX_rg = np.einsum("tni,tnj->tij", X_rg, X_rg)                             # (T, N, 3), (T, N, 3) -> (T, 3, 3)
rhs_rg = np.einsum("tij,tj->i", XtX_rg, G_rg)                              # Σ_t (Z̃'Z̃) ĝ_t：(T, 3, 3), (T, 3) -> (3,)
assert np.allclose(np.linalg.solve(XtX_rg.sum(axis=0), rhs_rg), gp_rg)    # 恒等式精确成立
assert np.all(np.abs(gp_rg - gbar_rg) < 0.5 * se_fm_rg)                    # 本模拟中两者接近：差别不到 FM 标准误的一半
assert np.all(ratio_rg > 2)                                                # 同方差标准误不到 FM 标准误的一半


def pooled_minus_fm(chars):
    """chars 为若干个 (T, N) 的特征数组；返回混合回归与 FM 点估计之差的最大绝对值。"""
    Xs = np.stack([np.ones((T_rg, N_rg)), *chars], axis=2)                 # 形状 (T, N, 1+p)，p 为特征个数
    fm = np.mean([np.linalg.lstsq(Xs[t], R_rg[t], rcond=None)[0] for t in range(T_rg)], axis=0)
    pooled = np.linalg.lstsq(Xs.reshape(T_rg * N_rg, -1), yp_rg, rcond=None)[0]
    return np.abs(pooled - fm).max()


def standardize(a):
    """逐期（逐行）横截面标准化：(T, N) -> (T, N)，每行均值 0、方差 1（除以 N）。"""
    return (a - a.mean(axis=1, keepdims=True)) / a.std(axis=1, keepdims=True)


z1_std, z2_std = standardize(Z12_rg[:, :, 0]), standardize(Z12_rg[:, :, 1])
gap_one, gap_two = pooled_minus_fm([z1_std]), pooled_minus_fm([z1_std, z2_std])
print(f"每期标准化后，混合回归与 FM 之差的最大绝对值：一个特征 {gap_one:.1e}，两个特征 {gap_two:.1e}")
assert gap_one < 1e-12 and gap_two > 1e-8                                  # 前者只是舍入误差，后者不为 0
assert gap_two < 0.5 * se_fm_rg.min()                                      # 但差别远小于 FM 标准误
```

混合回归与 Fama–MacBeth 的点估计相差不到 FM 标准误的一半，而且恒等式 $\hat{\mathbf g}^{\mathrm P} = \sum_t\mathbf W_t\hat{\mathbf g}_t$ 精确成立；只有一个标准化特征时二者之差只是舍入误差，有两个标准化特征时二者不再相等，但差别不到 FM 标准误的一半（均由 `assert` 保证，数值见上方输出）。本模拟器中特征相互独立、每期股票数相同，$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}/N$ 各期都接近 $\mathbf I$，这正是[结论 2](#primer-regs-conclusions) 中“二者接近”的情形；真实数据中特征相关、股票数逐年增加，不能想当然地认为二者接近。

标准误则差别很大：三个系数的 FM 标准误都超过同方差标准误的 2 倍（由 `assert` 保证，比值见上方输出），常数项相差最多，因为它的系数随 $f_{1,t}$ 波动，而 $\sigma_1 = 0.05$ 最大。哪一个标准误是对的？单次模拟回答不了，要靠重复模拟：用 200 个不同的种子各生成一份数据（为节省时间取 $N = 200$、$T = 120$），记下每次的混合回归估计及两种标准误，再把标准误的平均值与估计值在 200 次重复中的实际标准差比较。

```{code-cell} python
# 200 次重复：混合回归估计的实际抽样标准差，与两种标准误的平均值比较
mc_params = MarketParams(n_stocks=200, n_periods=120)
T_mc, N_mc = mc_params.n_periods, mc_params.n_stocks
est_mc, se_homo_mc, se_fm_mc = [], [], []
for r in range(200):
    rep = simulate_market(mc_params, seed=6000 + r)
    R_m = rep.panel["exret"].unstack("stock").to_numpy()                   # (T, N)
    X_m = np.concatenate([np.ones((T_mc, N_mc, 1)),
                          rep.panel[["z1", "z2"]].to_numpy().reshape(T_mc, N_mc, 2)], axis=2)   # (T, N, 3)
    G_m = np.array([np.linalg.lstsq(X_m[t], R_m[t], rcond=None)[0] for t in range(T_mc)])     # (T, 3)
    X_f, y_f = X_m.reshape(-1, 3), R_m.ravel()                             # (TN, 3)、(TN,)
    g_p = np.linalg.lstsq(X_f, y_f, rcond=None)[0]
    e_p = y_f - X_f @ g_p
    est_mc.append(g_p)
    se_homo_mc.append(np.sqrt(np.diag(e_p @ e_p / (len(y_f) - 3) * np.linalg.solve(X_f.T @ X_f, np.eye(3)))))
    se_fm_mc.append(G_m.std(axis=0, ddof=1) / np.sqrt(T_mc))
sd_true_mc = np.std(est_mc, axis=0, ddof=1)                                # 混合回归估计的实际抽样标准差：(3,)
homo_ratio_mc = np.mean(se_homo_mc, axis=0) / sd_true_mc
fm_ratio_mc = np.mean(se_fm_mc, axis=0) / sd_true_mc
print(pd.DataFrame({"实际标准差": sd_true_mc, "同方差标准误 / 实际": homo_ratio_mc, "FM 标准误 / 实际": fm_ratio_mc},
                   index=["常数", "z1", "z2"]).round(5).to_string())
# 200 次重复时，样本标准差的相对抽样误差约 1/√400 = 5%，阈值取约 3 倍
assert np.all(homo_ratio_mc < 0.6) and homo_ratio_mc[0] < 0.3 and np.all(np.abs(fm_ratio_mc - 1) < 0.15)
```

同方差标准误的平均值不到实际抽样标准差的 0.6 倍，常数项更是不到 0.3 倍；FM 标准误的平均值与实际抽样标准差相差在 15% 以内（均由 `assert` 保证，比值见上方输出）。也就是说，同方差标准误严重低估了混合回归估计的不确定性，按它算出的 $t$ 值会严重偏大；FM 标准误则与实际相符。这里用 FM 标准误衡量混合回归估计是合理的，因为本模拟中二者的点估计几乎相同。

**(e) 两步法示意。** 最后看第二步的解释变量换成载荷估计值时会怎样。把 240 期分成前后两半：先用前 120 期做时间序列回归，得到每只股票的 $\hat{\bv\beta}_i$；再在后 120 期逐期把 $\mathbf r_t$ 对 $\hat{\mathbf B}$（$N\times K$）做横截面回归，得到 $\hat{\bv\lambda}_t$，取平均得 $\bar{\bv\lambda}$。如果一切顺利，$\bar{\bv\lambda}$ 应当估计 $\bv\lambda = \bv\mu = (0.006, 0.003, 0.004)'$。第二步不加截距：本章模型中载荷为 $\mathbf 0$ 的资产期望超额收益为 0；而且所有股票对因子 1 的真实载荷都是 1，加上截距后，因子 1 的补偿与截距无法区分。作为对照，再把第二步的解释变量换成每一期的真实载荷 $\mathbf B_t$，它既没有估计误差，也没有前后两段的错位；对照的基准取后半段因子的样本均值 $\bar{\mathbf f}$。

(primer-unpack-twopass)=
**拆开讲两步法中的矩阵。** 第一步与时间序列回归相同（[拆解](#primer-unpack-ts)），只是样本换成前 $h = 120$ 期：$N$ 只股票共用 $h\times(1+K)$ 的设计矩阵 $[\mathbf 1_h, \mathbf F_{1:h}]$（$\mathbf F_{1:h}$ 为 $\mathbf F$ 的前 $h$ 行，代码中的 `D1_rg`），系数矩阵为 $(1+K)\times N$，去掉第一行截距再转置，得到 $N\times K$ 的 $\hat{\mathbf B}$（`B_hat1_rg`），第 $i$ 行为 $\hat{\bv\beta}_i'$。第二步与上一页对载荷估计值的横截面回归相同（[拆解](#primer-unpack-cs)），区别是不加截距：设计矩阵就是 $\hat{\mathbf B}$ 本身，不再是 $N\times(1+K)$ 的 $[\mathbf 1_N, \hat{\mathbf B}]$，于是每期的 $\hat{\bv\lambda}_t = (\hat{\mathbf B}'\hat{\mathbf B})^{-1}\hat{\mathbf B}'\mathbf r_t$ 是 $K\times K$ 乘 $K\times N$ 乘 $N\times 1$，得 $K\times 1$。$\hat{\mathbf B}$ 只估计一次，**后 120 期共用同一个设计矩阵**，每期只换 $\mathbf r_t$，每期各得一个 $\hat{\bv\lambda}_t$；它们按行堆叠成 $(T-h)\times K$ 的矩阵（`lam_t_rg`），各列的均值就是 $\bar{\bv\lambda} = (\bar\lambda_1, \bar\lambda_2, \bar\lambda_3)'$，由全部股票共用，第 $k$ 个分量估计第 $k$ 个因子的风险溢价 $\lambda_k$（各分量的含义与数值 $\bv\lambda = (0.6\%, 0.3\%, 0.4\%)'$ 见[拆解](#ch0-unpack-lambda)）。对照组的设计矩阵则是每期的真实载荷 $\mathbf B_t$（$N\times K$，`B_rg[t]`），每期不同。

```{code-cell} python
# (e) 两步法：前半段估计 β̂_i，后半段逐期把 r_t 对 β̂_i 做横截面回归（不加截距）
h_rg = T_rg // 2                                                           # 前后各 120 期
D1_rg = np.column_stack([np.ones(h_rg), F_rg[:h_rg]])                      # (h, 1+K)
B_hat1_rg = np.linalg.lstsq(D1_rg, R_rg[:h_rg], rcond=None)[0][1:].T      # (1+K, N) 取斜率 (K, N) -> (N, K)：第 i 行为 β̂_i'
lam_t_rg = np.array([np.linalg.lstsq(B_hat1_rg, R_rg[t], rcond=None)[0] for t in range(h_rg, T_rg)])   # (T-h, K)：λ̂_t'
lam_bar_rg = lam_t_rg.mean(axis=0)                                         # λ̄: (K,)
lam_se_rg = lam_t_rg.std(axis=0, ddof=1) / np.sqrt(T_rg - h_rg)

# 对照：第二步改用每期的真实载荷 B_t。θ = 0 时 r_t = B_t f_t + ε_t，所以 λ̂_t = f_t + (B_t'B_t)⁻¹B_t'ε_t，
# 其时间平均应等于后半段的因子样本均值 f̄，只差平均后的横截面噪声
lam_o_rg = np.array([np.linalg.lstsq(B_rg[t], R_rg[t], rcond=None)[0] for t in range(h_rg, T_rg)])   # (N, K) 对 (N,)：(T-h, K)
lam_o_bar = lam_o_rg.mean(axis=0)                                          # (K,)
f_bar2_rg = F_rg[h_rg:].mean(axis=0)                                       # 后半段因子样本均值 f̄：(K,)
BtB_inv_rg = np.mean([np.linalg.solve(B_rg[t].T @ B_rg[t], np.eye(K_rg)) for t in range(h_rg, T_rg)], axis=0)   # (B_t'B_t)⁻¹ 的平均：(K, K)
noise_se_rg = rg.params.idio_vol * np.sqrt(np.diag(BtB_inv_rg) / (T_rg - h_rg))   # 横截面噪声平均后的标准误：(K,)
B_true2_rg = B_rg[h_rg:].mean(axis=0)                                      # 后半段的真实平均载荷：(N, K)，下面用来计算相关系数
print(pd.DataFrame({"真值 λ = μ": mu_rg, "后半段因子均值 f̄": f_bar2_rg,
                    "两步法 λ̄": lam_bar_rg, "其 FM 标准误": lam_se_rg, "(μ − λ̄)/标准误": (mu_rg - lam_bar_rg) / lam_se_rg,
                    "对照 λ̄": lam_o_bar, "对照 − f̄": lam_o_bar - f_bar2_rg, "横截面噪声标准误": noise_se_rg},
                   index=["因子 1", "因子 2", "因子 3"]).round(5).to_string())

# β̂_i 含有多少关于载荷的信息：对因子 2、3，与前半段、后半段真实平均载荷的横截面相关系数
corr_first = np.array([np.corrcoef(B_hat1_rg[:, k], B_rg[:h_rg, :, k].mean(axis=0))[0, 1] for k in (1, 2)])
corr_second = np.array([np.corrcoef(B_hat1_rg[:, k], B_true2_rg[:, k])[0, 1] for k in (1, 2)])
print("β̂ 与前半段平均载荷的相关系数：", corr_first.round(3), "；与后半段平均载荷：", corr_second.round(3))

assert np.all((corr_first > 0.35) & (corr_first < 0.65))                   # 估计误差：β̂ 只部分反映同期的平均载荷
assert np.all(np.abs(corr_second) < 0.15)                                  # 载荷时变：β̂ 几乎不含后半段载荷的信息（N = 500 时相关系数的抽样误差约 0.045）
# 因子 2、3：λ̄ 比真值低 2 倍标准误以上。若两步法无偏，这样的单侧偏离概率约 2.3%，所以它是“显著偏低”的证据
assert np.all(lam_bar_rg[1:] < mu_rg[1:] - 2 * lam_se_rg[1:])
assert abs(lam_bar_rg[0] - mu_rg[0]) < 3 * lam_se_rg[0]                    # 因子 1：在 3 倍标准误以内
# 对照：给定 B_t 与 f_t，噪声均值为 0、标准误为 noise_se_rg；3 倍标准误对应双侧约 0.3% 的误报概率
assert np.all(np.abs(lam_o_bar - f_bar2_rg) < 3 * noise_se_rg)
assert np.all(noise_se_rg < 0.3 * mu_rg)                                   # 噪声标准误不到真值的 0.3 倍，对照才有区分力
```

结果如实报告如下。因子 1 的 $\bar\lambda_1$ 与真值的偏离在 3 倍标准误以内；但因子 2、3 的 $\bar\lambda_2$、$\bar\lambda_3$ 远低于真值，比真值低 2 倍标准误以上（均由 `assert` 保证，数值见上方输出）。两步法在这里基本失败了，原因有两个，最后一行输出把它们分开了：

- **估计误差。** 对因子 2、3，$\hat{\bv\beta}_i$ 与前半段真实平均载荷的相关系数只在 0.35～0.65 之间：单只股票 120 个月的载荷估计，标准误约为 0.3，比平均载荷在股票之间的差别还大，$\hat{\bv\beta}_i$ 中很大一部分是噪声。用带噪声的解释变量做回归，斜率偏向 0，这就是变量误差问题。
- **载荷时变。** $\hat{\bv\beta}_i$ 与后半段真实平均载荷的相关系数绝对值小于 0.15，几乎为 0（均由 `assert` 保证）。$\phi = 0.9$ 时特征的半衰期只有约 6.6 个月（$0.9^{6.6} \approx 0.5$），前 120 个月的平均特征几乎不能预测后 120 个月的平均特征，前半段估计的载荷对后半段也就几乎没有信息。因子 1 基本不受影响，是因为所有股票对它的载荷恒为 1，不随时间变化。

对照组把第二步的解释变量换成每期的真实载荷 $\mathbf B_t$。此时 $\hat{\bv\lambda}_t = \mathbf f_t + (\mathbf B_t'\mathbf B_t)^{-1}\mathbf B_t'\bv\varepsilon_t$（各项的维度见上一页的[拆解](#primer-unpack-cs)），它的时间平均应等于后半段的因子样本均值 $\bar{\mathbf f}$，只差平均后的横截面噪声；这个噪声的标准误不到真值的 0.3 倍，所以对照有足够的区分力。结果三个分量与 $\bar{\mathbf f}$ 之差都在 3 倍噪声标准误以内（均由 `assert` 保证，数值见上方输出）：解释变量换成真实载荷后，第二步准确地恢复了本样本的因子收益。这里的基准是 $\bar{\mathbf f}$ 而不是 $\bv\mu$：二者之差是因子样本均值的抽样误差，任何方法都无法消除。两步法失败，因此不在横截面回归本身，而在于解释变量 $\hat{\bv\beta}_i$。还要说明，用前半段估计一次 $\hat{\bv\beta}_i$、在后 120 期固定不变，是本示意的设计选择；实践中通常用滚动窗口（例如过去 60 个月）逐期重估载荷。在本模拟器中，滚动重估能缓解载荷时变造成的错位，但单只股票的估计误差依然很大，变量误差问题不能因此消除。这个模拟器有意让载荷随特征快速变化，把两步法的两个弱点都放大了；在载荷更稳定的情形下，问题会轻一些，但变量误差始终存在。处理办法，包括用组合代替个股、Shanken 修正以及直接用特征的方法，见第 10 章。

#### 第 0 章中已经用到这些回归的地方

- 0.2 节[“载荷（暴露、β）”](#sec-ch0-loading)：用券商股 6 个月的数据做单因子**时间序列回归**，斜率就是载荷的估计值。
- [因子、特征、载荷的关系](#sec-ch0-fcl)：Barra 式的**横截面回归**，$\mathbf r_t$ 对 $(1, z_1, z_2)$ 逐期回归，系数约为 $(f_{1,t},\ 0.5 f_{2,t},\ 0.5 f_{3,t})$。
- [模拟实验：用交互回归恢复 Γ](#sec-ch0-interact)：交互回归是一个**混合回归**，解释变量是“特征 × 因子”交互项；残差只剩 $\varepsilon_{i,t}$，所以那里的同方差标准误是对的。
- [常见误区](#sec-ch0-pitfalls)中的误区四：$r_{i,t}$ 对 $\tilde{\mathbf z}_{i,t-1}$ 的**混合回归**与**逐期横截面回归**并列，后者的标准误就是 **Fama–MacBeth 标准误**，比同方差标准误大得多。
- [习题 %s](#ex-ch0-ts)：对每只股票的**时间序列回归**，证明斜率是平均载荷、截距是平均定价误差。
- [习题 %s](#ex-ch0-beta)：用**混合回归**估计 $\bv\Gamma\bv\mu + (0, \bv\theta')'$，用 **Fama–MacBeth 标准误**判断偏离。
- [习题 %s](#ex-ch0-theta)：管理组合矩的“按期标准误”与 Fama–MacBeth 是同一个思路——先对每期求横截面平均，再看这些平均值的时间序列波动。

(sec-ch0-regs-industry)=
#### 与业界的对应

国内量化业界天天在用这几种回归，只是名字不同。下面是通行用法，各机构的具体做法不一。

- **Barra 的因子收益就是横截面回归的系数。** 每期把股票收益对行业哑变量与标准化的风格暴露做横截面回归，系数就是当期的“因子收益”，即本节的 $\hat{\mathbf g}_t$（上一页；[因子收益](#glossary-factor-return)、[风险模型](#glossary-risk-model)）。
- **IC 对应横截面相关。** 第 $t$ 期的 IC 是信号与收益的横截面相关系数，与一元横截面回归的斜率只差一个比例 $\hat\sigma_{r,t}/\hat\sigma_{z,t}$；在一定前提下，IC 均值乘以收益的横截面标准差约为 Fama–MacBeth 的点估计，ICIR 乘以 $\sqrt T$ 约为 Fama–MacBeth 的 $t$ 值（精确关系与成立条件见 [IC](#glossary-ic)）。
- **单因子测试的“回归法”本质上是 Fama–MacBeth 的思路。** 卖方金工报告检验一个因子时，常每期把下期收益对因子值做横截面回归，并控制行业与市值（或者先对因子值做行业、市值[中性化](#glossary-neutralization)），得到“因子收益”序列与每期的 $t$ 值序列，再报告三类统计量：因子收益序列的均值及其 $t$ 值，这正是 FM 估计量与 FM $t$ 值；$t$ 值序列绝对值的均值；$|t| > 2$ 的期数比例。后两类统计的是**每期**横截面回归的 $t$ 值，与 FM $t$ 值不是一回事。上面 (c) 的输出给出了一个例子：$z_1$、$z_2$ 的每期 $|t| > 2$ 的期数都超过一半，其中 $t$ 值大于 2 与小于 $-2$ 的期数都超过 15%（均由 `assert` 保证），因为每期系数随 $f_{2,t}$、$f_{3,t}$ 正负波动；它们只说明“这个特征在多数期与收益有很强的关系，方向时正时负”，平均补偿是否为正，要看 FM $t$ 值。有的报告还会用市值平方根加权的 WLS、用排序代替原始因子值，或者对收益做行业调整。

这些口径在学术与业界的差别，整理在术语对照页的[横截面回归与 Fama–MacBeth](#glossary-fm) 条目中。
