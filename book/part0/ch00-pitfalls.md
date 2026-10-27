---
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  equation:
    continue: true
  exercise:
    template: 习题 %s
---

(sec-ch0-pitfalls)=
# 常见误区

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块与函数，并按与 0.4 节、0.6 节完全相同的参数和种子，重建本页用到的两个模拟市场及由它们算出的数组。误区四用到的最小二乘函数 `ols` 已在[“模拟实验”一节](#sec-ch0-interact)内联实现并与 `apsim.stats` 核对，这里直接从 `apsim.stats` 导入。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market, stats
from apsim.stats import ols

# 0.4 节的默认市场（种子 0），误区二用到
params = MarketParams()
mkt = simulate_market(params, seed=0)

# 0.6 节“模拟验证”的长样本（θ1 = 0.01，种子 2026），误区一用到
long_params = MarketParams(n_stocks=20, n_periods=20_000, alpha_coef=(0.01, 0.0, 0.0, 0.0, 0.0))
long_mkt = simulate_market(long_params, seed=2026)
MR = long_mkt.sdf().to_numpy()[:, None] * long_mkt.panel["exret"].unstack("stock").to_numpy()   # (T, N)：m_t r_{i,t}
Z1 = long_mkt.panel["z1"].unstack("stock").to_numpy()                                         # (T, N)
```

**误区一：把同期特征当作期初特征。** 用 $\mathbf z_{i,t}$ 而不是 $\mathbf z_{i,t-1}$ 去解释 $r_{i,t}$，在真实数据中可能造成严重的前视偏差：许多特征（市值、动量、换手率）与当期收益机械相关，例如第 $t$ 期末的市值本身就包含了第 $t$ 期的收益。在本章的模拟器中，特征更新的冲击 $\mathbf e_{i,t}$ 与收益无关，所以错用同期特征不会产生虚假的预测力，只会削弱真实的关系：由式 {eq}`eq-ch0-char` 与 $\mathbf e_{i,t}$ 的独立性，

```{math}
:enumerated: false
\underbrace{\E[m_t r_{i,t} \mathbf z_{i,t}]}_{L\times 1} = \phi\,\E[m_t r_{i,t} \mathbf z_{i,t-1}] + \sqrt{1-\phi^2}\,\E[m_t r_{i,t} \mathbf e_{i,t}] = \phi\,\bv\theta + \mathbf 0 .
```

(ch0-unpack-pit1)=
**拆开讲。** 这是 $L\times 1$ 的向量等式，第 $j$ 个分量是

```{math}
:enumerated: false
\E[m_t r_{i,t}\, z_{j,i,t}] = \phi\,\E[m_t r_{i,t}\, z_{j,i,t-1}] + \sqrt{1-\phi^2}\,\E[m_t r_{i,t}\, e_{j,i,t}] = \phi\,\theta_j ,
```

即把式 {eq}`eq-ch0-char` 的分量形式（[拆解](#ch0-unpack-char)）代入，再利用期望的线性。第一项由[推论 %s](#cor-ch0-managed) 等于 $\phi\theta_j$（各分量的含义见[拆解](#ch0-unpack-managed)）；第二项为 0，因为 $r_{i,t}$ 只依赖 $\mathbf z_{i,t-1}$、$\mathbf f_t$ 与 $\varepsilon_{i,t}$，$e_{j,i,t}$ 与 $(m_t, r_{i,t})$ 独立且均值为 0。默认 $\phi = 0.9$、$\theta_1 = 0.01$ 时，用同期特征算出的管理组合定价误差为 $0.9 \times 0.01 = 0.009$，比真值小一成：错用同期特征在这里只是“削弱”，而不是“虚构”。**特例**：$\phi = 0$ 时同期特征与期初特征无关，管理组合的定价误差完全消失。

这个例子说明了模拟的局限：模拟器不会自动暴露所有错误，只能暴露我们放进模型的那些机制。下面用 $\phi = 0.9$ 的长样本验证。两个样本矩都带有抽样误差，而且误差高度相关；所以除了把它们分别与理论值比较，还检验二者之差是否满足推导给出的关系：用 $\overline{m r z_1}$ 表示 $m_t r_{i,t} z_{1,i,t-1}$ 对全部 $(i,t)$ 的样本均值、$\overline{m r z_1^{+}}$ 表示把 $z_{1,i,t-1}$ 换成同期 $z_{1,i,t}$ 后的样本均值，则 $\overline{m r z_1^{+}} - \phi\, \overline{m r z_1}$ 只剩下 $\sqrt{1-\phi^2}\, \overline{m r e_1}$（$e_{1,i,t}$ 为 $\mathbf e_{i,t}$ 的第一个分量），理论值为 0。

```{code-cell} python
Z1_now = long_mkt.panel["z1"].unstack("stock").shift(-1).to_numpy()   # (T, N)，第 t 行放 z_{1,i,t}
valid = ~np.isnan(Z1_now[:, 0])                                        # 最后一期没有 z_{i,T}
now = (MR[valid] * Z1_now[valid]).mean()
lag = (MR[valid] * Z1[valid]).mean()
phi = long_params.char_persistence
print(f"用期初特征 z_(t-1)：{lag:.5f}（理论值 θ1 = {long_params.alpha_coef[0]}）")
print(f"用同期特征 z_t    ：{now:.5f}（理论值 φθ1 = {phi * 0.01:.4f}）")
resid_term = (MR[valid] * (Z1_now[valid] - phi * Z1[valid])).mean(axis=1)   # 每期横截面平均：(T-1,)
se_resid = resid_term.std() / np.sqrt(resid_term.size)
print(f"now − φ·lag = {now - phi * lag:.5f}，标准误约 {se_resid:.5f}（理论值 0）")
assert abs(now - phi * lag) < 3 * se_resid
```

二者之差与 0 的距离在 3 倍标准误以内（由 `assert` 保证）。这里的标准误同样先对每期求横截面平均再计算。

**误区二：把样本均值当作总体值。** 默认参数下第一个因子的月均值 $\mu_1 = 0.6\%$、波动率 $5\%$，240 个月样本均值的标准误为 $0.05/\sqrt{240} \approx 0.32\%$，样本均值完全可能偏离真值一半以上。更隐蔽的错误是用样本矩构造 SDF，再在同一样本中“验证”它。若 $\hat{\mathbf b} = \hat{\bv\Sigma}_f^{-1}\bar{\mathbf f}$（$\hat{\bv\Sigma}_f$ 为除以 $T$ 的样本协方差矩阵，$\bar{\mathbf f}$ 为样本均值向量），$\hat m_t = 1 - \hat{\mathbf b}'(\mathbf f_t - \bar{\mathbf f})$，则把[定理 %s](#thm-ch0-sdf) 第 2 条证明中的总体矩逐一换成样本矩，可得样本内 $T^{-1}\sum_t \hat m_t \mathbf f_t = \bar{\mathbf f} - \hat{\bv\Sigma}_f \hat{\mathbf b} = \mathbf 0$ **恒成立**，无论模型对错。这样的“检验”什么也没有检验。

(ch0-unpack-bhat)=
**拆开讲。** 带帽子的量都是样本矩：$\bar{\mathbf f} = T^{-1}\sum_t\mathbf f_t$ 是 $K\times 1$，第 $k$ 个分量是第 $k$ 个因子的样本均值；$\hat{\bv\Sigma}_f = T^{-1}\sum_t(\mathbf f_t - \bar{\mathbf f})(\mathbf f_t - \bar{\mathbf f})'$ 是 $K\times K$，$(k, l)$ 元素是第 $k$ 与第 $l$ 个因子的样本协方差（除以 $T$）；$\hat{\mathbf b}$ 是 $K\times 1$，解 $\hat{\bv\Sigma}_f\hat{\mathbf b} = \bar{\mathbf f}$；$\hat m_t$ 是标量。恒等式的推导与定理第 2 条的证明逐步对应（外积的写法见[拆解](#ch0-unpack-outer)），只是把 $\E$ 换成 $T^{-1}\sum_t$、把 $\bv\mu$ 换成 $\bar{\mathbf f}$：

```{math}
:enumerated: false
\frac{1}{T}\sum_{t=1}^{T}\hat m_t\mathbf f_t = \bar{\mathbf f} - \Bigl[\frac{1}{T}\sum_{t=1}^{T}\mathbf f_t(\mathbf f_t - \bar{\mathbf f})'\Bigr]\hat{\mathbf b} = \bar{\mathbf f} - \hat{\bv\Sigma}_f\hat{\mathbf b} = \mathbf 0 .
```

方括号里的 $K\times K$ 矩阵等于 $\hat{\bv\Sigma}_f + \bar{\mathbf f}\,\bigl[T^{-1}\sum_t(\mathbf f_t - \bar{\mathbf f})\bigr]'$，而样本中去均值后的因子之和恰好为 $\mathbf 0$，所以第二项为零。总体证明中对应的一项为零，依赖 $\E[\mathbf f_t] = \bv\mu$ 这一关于总体的性质；样本中它为零，则是 $\bar{\mathbf f}$ 的定义决定的，与模型对错无关。**特例**：$K = 1$ 时 $\hat b = \bar f/\hat\sigma^2$，$T^{-1}\sum_t\hat m_t f_t = \bar f - \hat\sigma^2\hat b = 0$。**数据矩阵**：代码中 `F240` 是 $T\times K$ 的 $\mathbf F$，`(F240 - f_bar) @ b_hat` 一次算出 $T$ 个 $\hat{\mathbf b}'(\mathbf f_t - \bar{\mathbf f})$。

```{code-cell} python
F240 = mkt.factors.to_numpy()                              # (T, K)
f_bar = F240.mean(axis=0)                                  # (K,)
Sigma_hat = np.cov(F240, rowvar=False, ddof=0)             # (K, K)
b_hat = np.linalg.solve(Sigma_hat, f_bar)                  # 解 Σ̂_f b̂ = f̄
m_hat = 1 - (F240 - f_bar) @ b_hat                         # (T, K) @ (K,) -> (T,)
in_sample = (m_hat[:, None] * F240).mean(axis=0)
true_pricing = (mkt.sdf().to_numpy()[:, None] * F240).mean(axis=0)
print("因子样本均值", f_bar.round(4), "真值", params.factor_mean, "标准误", (np.asarray(params.factor_vol) / np.sqrt(len(F240))).round(4))
print("b 的样本估计", b_hat.round(2), "真值", mkt.sdf_weights.round(2))
print("样本内 mean(m̂ f) =", in_sample)
print("真实 m 的 mean(m f) =", true_pricing.round(5))
assert np.max(np.abs(in_sample)) < 1e-12 and np.max(np.abs(true_pricing)) > 1e-4
```

样本内的 $T^{-1}\sum_t \hat m_t \mathbf f_t$ 在机器精度内为 $\mathbf 0$，而真实 SDF 在同一样本中的样本定价误差并不为 0——后者才反映了抽样误差的真实大小。真正的检验需要样本外数据，或者把估计误差计入检验统计量，这是第二部分的主题。

**误区三：忽视随机种子。** 没有固定种子的随机数每次运行都不同，书中的数字因此无法复现。为了避免这种错误，`apsim.simulate_market` 要求必须以关键字传入 `seed`，省略它或写成 `seed=None` 都会报错；确实需要不可复现的随机数时，只能显式传入一个未设种子的 `np.random.default_rng()`。反过来的错误同样常见：在蒙特卡洛循环中每次都用同一个种子，得到的“独立重复”其实完全相同，估计量的抽样方差被算成 0。

```{code-cell} python
small = MarketParams(n_stocks=50, n_periods=60)
try:
    simulate_market(small)
except TypeError as err:
    print("省略 seed：TypeError：", err)

first = simulate_market(small, seed=np.random.default_rng()).panel["exret"].iloc[0]
second = simulate_market(small, seed=np.random.default_rng()).panel["exret"].iloc[0]
print(f"未设种子的 Generator 两次运行的第一个收益：{first:.4f} 与 {second:.4f}")

same = [simulate_market(small, seed=0).factors["f1"].mean() for _ in range(5)]
diff = [simulate_market(small, seed=100 + r).factors["f1"].mean() for r in range(5)]
print("同一种子重复 5 次，f1 样本均值的标准差：", np.std(same))
print("不同种子重复 5 次，f1 样本均值的标准差：", round(float(np.std(diff)), 4))
assert np.std(same) == 0 and np.std(diff) > 0
```

（第二行的两个数字每次重新执行都会变化，这正是问题所在。）本书的约定是：单次演示用固定的整数种子；蒙特卡洛重复用“基础种子 + 重复编号”，保证各次重复互不相同，而整体可以复现。另一种同样可复现的做法，是新建一个带种子的 `rng = np.random.default_rng(2026)`（2026 为基础种子），在循环中反复传入 `seed=rng`：`simulate_market` 每次都从它派生新的子流，各次重复互不相同。

**误区四：把可预测性等同于错误定价。** 即使 $\bv\theta = \mathbf 0$，特征也能预测收益。由式 {eq}`eq-ch0-ret` 与独立性，$\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\mu$，而 $\bv\beta_{i,t}$ 随特征变化。默认参数下 $\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \mu_1 + 0.5\mu_2\, z_{1,i,t-1} + 0.5\mu_3\, z_{2,i,t-1}$：特征 $z_1$、$z_2$ 能预测收益，完全是因为它们决定了风险暴露，这正是“特征即协方差”的含义 [@kelly2019characteristics]。判断一个能预测收益的特征是风险补偿还是错误定价，要看它的预测力能否被载荷与因子风险溢价解释，而不能只看预测力本身（第 8、23 章）。下面用 $\bv\theta = \mathbf 0$ 的长样本做 $r_{i,t}$ 对 $\tilde{\mathbf z}_{i,t-1}$ 的混合回归。由于 $\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \bv\beta_{i,t}'\bv\mu = \tilde{\mathbf z}_{i,t-1}'\bv\Gamma\bv\mu$，系数的理论值为 $\bv\Gamma\bv\mu$（$(L+1)\times K$ 乘 $K\times 1$，为 $L+1$ 维向量）。

(ch0-unpack-gmu)=
**拆开讲 $\bv\Gamma\bv\mu$。** 它的第 $j$ 个分量是 $\bv\Gamma$ 第 $j$ 行与 $\bv\mu$ 的内积 $\sum_k\Gamma_{j,k}\mu_k$，即特征 $j$ 每高 1 个标准差、期望收益高出的部分（第 0 个分量是全部特征处于平均水平的股票的期望收益），单位与收益相同。默认参数下

```{math}
:enumerated: false
\bv\Gamma\bv\mu = \begin{pmatrix} 1\cdot\mu_1 \\ 0.5\,\mu_2 \\ 0.5\,\mu_3 \\ 0 \\ 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 0.6\% \\ 0.15\% \\ 0.2\% \\ 0 \\ 0 \\ 0 \end{pmatrix} ,
```

所以 $z_1$ 每高 1 个标准差，期望收益每月高 0.15%；$z_2$ 每高 1 个标准差，每月高 0.2%，全部来自风险补偿。它是 0.2 节横截面回归系数 $\mathbf g_t$ 的均值（$\bv\theta = \mathbf 0$ 时，[拆解](#ch0-unpack-g)）：每期的 $\mathbf g_t = \bv\Gamma\mathbf f_t$ 围绕 $\bv\Gamma\bv\mu$ 波动。**特例**：$L = 0$ 时 $\bv\Gamma$ 只有一行，$\bv\Gamma\bv\mu = \sum_k\Gamma_{0,k}\mu_k$ 是一个数，即所有股票共同的风险补偿 $\bv\beta'\bv\lambda$（这里 $\bv\lambda = \bv\mu$）。**数据矩阵**：混合回归把 $TN$ 个观测叠在一起，设计矩阵 `Xp` 为 $TN\times(L+1)$，所有观测共用一个系数向量；逐期回归则每期一个 $N\times(L+1)$ 的设计矩阵、一个系数向量。

逐期回归的代码用 `reshape(n_t, n_i, -1)` 把 $TN$ 行的设计矩阵还原为“期 × 股票 × 变量”的三维数组。面板按 `(period, stock)` 排序、期在外层，所以前 $N$ 行是第 1 期，接下来的 $N$ 行是第 2 期，行优先的 `reshape` 恰好按期切开。

```{code-cell} python
pred = simulate_market(MarketParams(n_stocks=100, n_periods=5_000), seed=11)
Xp = np.column_stack([np.ones(len(pred.panel)), pred.panel[[f"z{j}" for j in range(1, 6)]].to_numpy()])  # (TN, L+1)
coef_p, se_p = ols(Xp, pred.panel["exret"].to_numpy())
truth_p = pred.gamma @ np.asarray(pred.params.factor_mean)               # Γμ：(L+1, K) @ (K,) -> (L+1,)

# 逐期横截面回归，用各期系数的时间序列标准差计算标准误（Fama–MacBeth 的做法，第 10 章详述）
n_t, n_i = pred.params.n_periods, pred.params.n_stocks
Xp3 = Xp.reshape(n_t, n_i, -1)                                           # (TN, L+1) -> (T, N, L+1)
yp3 = pred.panel["exret"].to_numpy().reshape(n_t, n_i)                   # (TN,) -> (T, N)
per_period = np.array([np.linalg.lstsq(Xp3[t], yp3[t], rcond=None)[0] for t in range(n_t)])   # (T, L+1)
se_fm = per_period.std(axis=0, ddof=1) / np.sqrt(n_t)
fm_p = stats.fama_macbeth(Xp3, yp3)                                      # 同一程序的 apsim.stats 版本：比较 coefs、mean、se
assert all(np.allclose(a, b) for a, b in [(fm_p.coefs, per_period), (fm_p.mean, per_period.mean(axis=0)), (fm_p.se, se_fm)])
print(pd.DataFrame({"理论值 Γμ": truth_p, "混合回归估计": coef_p, "同方差标准误": se_p, "逐期回归标准误": se_fm},
                   index=["常数", "z1", "z2", "z3", "z4", "z5"]).round(5))
ratio = se_fm / se_p
print("逐期回归标准误 / 同方差标准误：", ratio.round(2))
assert np.all(np.abs(coef_p - truth_p) < 2 * se_fm)                    # 偏离在两倍逐期回归标准误以内
assert np.all(coef_p[1:3] / se_fm[1:3] > 3)                            # z1、z2 的系数显著为正
assert np.all(ratio[:3] > 1.3) and np.all(np.abs(ratio[3:] - 1) < 0.2)
```

系数的估计值与理论值 $\bv\Gamma\bv\mu$ 相符（偏离都在两倍逐期回归标准误以内）：$z_1$、$z_2$ 的系数显著为正（均由 `assert` 保证），而 $\bv\theta = \mathbf 0$，市场中没有任何错误定价。表中还有一个第二部分会反复出现的问题：“同方差标准误”低估了真实的不确定性。同一期的所有股票共同受到 $\mathbf f_t$ 的冲击，残差在横截面上相关，混合回归把 $NT$ 个观测当作相互独立，高估了信息量。按期做横截面回归、再看各期系数的时间序列波动，得到的标准误更接近真实的抽样误差，这就是 Fama–MacBeth 标准误（[导读：几种回归](#sec-ch0-regs)，第 10、12 章）：常数项与 $z_1$、$z_2$ 的逐期回归标准误明显大于同方差标准误（比值见上方输出），因为它们的系数随 $\mathbf f_t$ 逐期波动；$z_3$～$z_5$ 与因子无关，两种标准误相近。估计值与理论值的偏离，应当按这种更大的标准误来判断。

