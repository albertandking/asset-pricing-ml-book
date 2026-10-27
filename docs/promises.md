# 前向引用台账

本文件由 `scripts/promises.py` 生成，请勿手工修改。重新生成：`uv run python scripts/promises.py`。

正文中指向尚未写出的章节或附录的引用，按被引用的章号分组，每条给出文件与行号（如 `book/glossary.md:30`），破折号后是所在句子的上下文。写每一章之前，逐条核对指向本章的承诺：在本章兑现，或改写原处的说法；本章写完后重新生成本文件。

共 121 条，涉及 24 个章节或附录。

## 第 1 章（9 条）

- `book/glossary.md:209` — ……收益与 SDF 的协方差决定，$\E[R^e] = -\Cov(m, R^e)/\E[m]$（随机贴现因子，第 1、3 章）；
- `book/index.md:106` — 建议的阅读顺序是：先读第 0 章，按第 0 章首页的阅读路径，第一次阅读只读必读部分（约 5 万字）即可进入第 1 章；
- `book/notation.md:114` — 第 1、6、7 章的惯例，如 $p_t = \log P_t$
- `book/part0/ch00-concepts.md:24` — 本节只求建立直觉：模型的方程在 0.4 节，定理与证明在 0.5 节，系统的理论在第 1～4 章展开。
- `book/part0/ch00-concepts.md:58` — 但并非所有因子都是资产收益：第 1、4、10 章会遇到消费增长、宏观变量这类不可交易的因子，它们的风险溢价 $\bv\lambda$ 不等于因子的均值。
- `book/part0/ch00-concepts.md:185` — 不可交易的因子（例如消费增长、宏观变量）不是资产收益，不能套用这一推理，$\bv\lambda$ 一般不等于因子的均值，需要另行估计（第 1、4、10 章）。
- `book/part0/ch00-concepts.md:289` — SDF 与 β 定价之间的关系，在模拟市场中由 0.5 节的定理与习题 %s 严格证明，一般理论在第 1～4 章展开。
- `book/part0/ch00-guide.md:82` — 第一次阅读时，只读标为“必读”的部分就可以进入第 1 章；
- `book/part0/ch00-model.md:180` — 需要无风险利率的场合（第 1 章起），再显式引入 $R^f_{t-1}$（面板写法）或 $R^f_t$（理论写法）。

## 第 2 章（4 条）

- `book/glossary.md:486` — 无套利是资产定价的基本假设，由它可以推出存在为正的 SDF（第 2 章）。
- `book/part0/ch00-concepts.md:24` — 本节只求建立直觉：模型的方程在 0.4 节，定理与证明在 0.5 节，系统的理论在第 1～4 章展开。
- `book/part0/ch00-concepts.md:289` — SDF 与 β 定价之间的关系，在模拟市场中由 0.5 节的定理与习题 %s 严格证明，一般理论在第 1～4 章展开。
- `book/part0/ch00-model.md:284` — 这本身并不意味着这些因子之间存在套利：无套利等价于存在某个严格为正的 SDF（第 2 章），而一个特定的线性 $m_t$ 取负值，并不排除另有一个正的 ……

## 第 3 章（7 条）

- `book/glossary.md:30` — 因子相关时二者可以不同（见第 3、4 章）。
- `book/glossary.md:183` — ……绝对值都不超过 $\sigma(m)/\E[m]$（Hansen–Jagannathan 界 [@hansen1991implications]，第 3 章）。
- `book/glossary.md:209` — ……收益与 SDF 的协方差决定，$\E[R^e] = -\Cov(m, R^e)/\E[m]$（随机贴现因子，第 1、3 章）；
- `book/part0/ch00-concepts.md:24` — 本节只求建立直觉：模型的方程在 0.4 节，定理与证明在 0.5 节，系统的理论在第 1～4 章展开。
- `book/part0/ch00-concepts.md:289` — SDF 与 β 定价之间的关系，在模拟市场中由 0.5 节的定理与习题 %s 严格证明，一般理论在第 1～4 章展开。
- `book/part0/ch00-exercises.md:293` — ……DF 波动率与最大夏普比率的关系，即 Hansen–Jagannathan 界，原始文献为 @hansen1991implications，第 3 章详细讨论。
- `book/part0/ch00-model.md:284` — 第 3 章将详细讨论。

## 第 4 章（8 条）

- `book/glossary.md:30` — 因子相关时二者可以不同（见第 3、4 章）。
- `book/glossary.md:209` — 在因子模型中，就是对被定价因子的载荷（第 4 章）。
- `book/glossary.md:486` — Ross（1976）的套利定价理论（APT）用的是近似套利：大量资产的分散化组合几乎消除了异质风险（第 4 章）。
- `book/part0/ch00-concepts.md:24` — 本节只求建立直觉：模型的方程在 0.4 节，定理与证明在 0.5 节，系统的理论在第 1～4 章展开。
- `book/part0/ch00-concepts.md:58` — 但并非所有因子都是资产收益：第 1、4、10 章会遇到消费增长、宏观变量这类不可交易的因子，它们的风险溢价 $\bv\lambda$ 不等于因子的均值。
- `book/part0/ch00-concepts.md:185` — 不可交易的因子（例如消费增长、宏观变量）不是资产收益，不能套用这一推理，$\bv\lambda$ 一般不等于因子的均值，需要另行估计（第 1、4、10 章）。
- `book/part0/ch00-concepts.md:289` — SDF 与 β 定价之间的关系，在模拟市场中由 0.5 节的定理与习题 %s 严格证明，一般理论在第 1～4 章展开。
- `book/part0/ch00-exercises.md:164` — 第 4 章会在更一般的条件下讨论 SDF 表示与 β 表示的等价性。

## 第 5 章（2 条）

- `book/part0/ch00-exercises.md:63` — 这正是第 5 章“条件模型与管理组合”的核心思想：条件矩 $\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = 0$ 等价于对所有……
- `book/part0/ch00-model.md:258` — 这正是第 5 章条件模型要处理的问题。

## 第 6 章（7 条）

- `book/glossary.md:271` — 市场给一只股票高估值，可能是因为预期增长高，也可能是因为折现率低（第 6 章）。
- `book/glossary.md:447` — ……te)）随经济状况变化：坏时候投资者要求更高的补偿，价格低、股息率高，随后的平均收益也高 [@cochrane2011presidential]（第 6 章）。
- `book/glossary.md:473` — 样本外表现普遍不如样本内（第 6、13 章）。
- `book/glossary.md:528` — 资产定价实证多用简单收益（尤其是超额收益），Campbell–Shiller 分解等推导用对数收益（第 6 章）。
- `book/glossary.md:541` — 折现率随时间、也随资产变化 [@cochrane2011presidential]（第 6 章）。
- `book/glossary.md:545` — 对数形式在第 6 章另行定义。
- `book/notation.md:114` — 第 1、6、7 章的惯例，如 $p_t = \log P_t$

## 第 7 章（1 条）

- `book/notation.md:114` — 第 1、6、7 章的惯例，如 $p_t = \log P_t$

## 第 8 章（6 条）

- `book/glossary.md:167` — 学术研究关心的问题是：溢价为什么存在（风险补偿还是错误定价，第 8 章）；
- `book/glossary.md:447` — 可预测性来自风险补偿还是错误定价，是第 8 章的主题。
- `book/glossary.md:486` — ……利限制”（limits to arbitrage）中的套利也取学术含义，讨论的是现实中的套利需要资本、要承担风险与成本，所以错误定价可以持续存在（第 8 章）。
- `book/part0/ch00-concepts.md:240` — 区分这两种解释往往很难，是第 8 章的主题。
- `book/part0/ch00-experiments.md:43` — A 股融券规模小、成本高，许多“空头腿”在现实中难以实现，这既影响因子收益的可实现性（第 26 章），也是错误定价能够持续存在的一个原因（第 8 章）。
- `book/part0/ch00-experiments.md:281` — 判断一个能预测收益的特征是风险补偿还是错误定价，要看它的预测力能否被载荷与因子风险溢价解释，而不能只看预测力本身（第 8、23 章）。

## 第 9 章（6 条）

- `book/glossary.md:554` — 一是两种回归：时间序列回归对每只资产沿时间回归，得到载荷与截距（第 9 章）；
- `book/index.md:106` — 导读：几种回归是第二部分的预习，读到第 9 章之前读即可。
- `book/part0/ch00-exercises.md:109` — 也就是说，即使存在错误定价，每只股票时间序列回归的截距在总体上也是 0，GRS 一类的时间序列检验（第 9 章）检验的只是平均定价误差，看不到这里的条件错误定价……
- `book/part0/ch00-guide.md:95` — 它是第二部分（实证计量）的预习，不属于本章，第一次阅读可以跳过，读到第 9 章之前再回来。
- `book/part0/primer-regressions.md:11` — 本页是第二部分（实证计量）的预习，不属于第 0 章的必读内容，第一次阅读可以跳过，读到第 9 章之前再回来。
- `book/part0/primer-regressions.md:65` — 检验全部 $N$ 个截距同时为 0 的 GRS 检验 [@gibbons1989test]，见第 9 章。

## 第 10 章（9 条）

- `book/glossary.md:391` — 载荷需先由时间序列回归估计（两步法），由此带来的变量误差与 Shanken 修正 [@shanken1992estimation] 见第 10 章。
- `book/glossary.md:408` — ……法是“控制变量”：在 Fama–MacBeth 横截面回归 [@fama1973risk] 中同时放入信号与市值、行业等控制变量，看信号的系数（第 10 章）；
- `book/glossary.md:554` — 横截面回归每期沿资产回归，得到风险溢价或特征的补偿（横截面回归与 Fama–MacBeth，第 10 章）。
- `book/part0/ch00-concepts.md:58` — 但并非所有因子都是资产收益：第 1、4、10 章会遇到消费增长、宏观变量这类不可交易的因子，它们的风险溢价 $\bv\lambda$ 不等于因子的均值。
- `book/part0/ch00-concepts.md:185` — 不可交易的因子（例如消费增长、宏观变量）不是资产收益，不能套用这一推理，$\bv\lambda$ 一般不等于因子的均值，需要另行估计（第 1、4、10 章）。
- `book/part0/ch00-experiments.md:306` — ……接近真实的抽样误差，这就是 Fama–MacBeth 标准误（导读：几种回归，第 10、12 章）：常数项与 $z_1$、$z……
- `book/part0/primer-regressions.md:102` — 这些都在第 10 章展开。
- `book/part0/primer-regressions.md:103` — 估计风险溢价 $\bv\lambda$（第 10 章）；
- `book/part0/primer-regressions.md:367` — 处理办法，包括用组合代替个股、Shanken 修正以及直接用特征的方法，见第 10 章。

## 第 11 章（1 条）

- `book/notation.md:179` — 第 11 章 GMM 的加权矩阵记为 $\mathbf W$（不带下标），二者无关

## 第 12 章（6 条）

- `book/glossary.md:189` — 夏普比率的估计误差很大：样本只有几年时，年化 1 与 0.5 往往难以在统计上区分（标准误见第 12 章）。
- `book/glossary.md:235` — $t$ 值依赖标准误的算法，要考虑自相关、截面相关与估计误差（第 12 章）；
- `book/glossary.md:400` — 按时间聚类的标准误见第 12 章。
- `book/part0/ch00-experiments.md:306` — ……接近真实的抽样误差，这就是 Fama–MacBeth 标准误（导读：几种回归，第 10、12 章）：常数项与 $z_1$、$z……
- `book/part0/primer-regressions.md:104` — ……t$ 在时间上相关时（例如用月度数据计算未来 12 个月的收益，相邻各期的收益期限重叠），标准误会偏小，需要 Newey–West 一类的调整，见第 12 章。
- `book/part0/primer-regressions.md:137` — 允许同一期残差任意相关的“按时间聚类”（clustered by time）标准误，见第 12 章。

## 第 13 章（5 条）

- `book/glossary.md:440` — 样本外检验的计量见第 13、19 章）。
- `book/glossary.md:460` — 若改用历史均值作基准，所有方法的 $R^2$ 约高 3 个百分点（第 13、22 章）。
- `book/glossary.md:473` — 样本外表现普遍不如样本内（第 6、13 章）。
- `book/part0/ch00-exercises.md:230` — 第 13 章讨论这一问题。
- `book/part0/ch00-exercises.md:294` — 持续回归变量的小样本偏误见 @stambaugh1999predictive，第 13 章详细讨论。

## 第 14 章（4 条）

- `book/glossary.md:235` — ……一批数据上试过很多信号时，阈值还要提高，Harvey、Liu 与 Zhu 主张新因子的 $t$ 值应超过 3 [@harvey2016and]（第 14 章）。
- `book/glossary.md:440` — 尝试过的方案越多，回测中最好的结果越可能只是运气（多重检验问题，第 14 章讨论；
- `book/glossary.md:499` — Harvey、Liu 与 Zhu 据此主张提高显著性的门槛 [@harvey2016and]（第 14 章）。
- `book/glossary.md:505` — 一个回测的参数不多，不代表没有过拟合：如果得到它之前试过几十个方案，就要按“试了多少次”校正显著性（第 14 章）。

## 第 15 章（8 条）

- `book/glossary.md:158` — A 股实证中使用的代理变量在第 15 章说明。
- `book/glossary.md:271` — CH3 在中国市场改用盈利市值比（EP），价值因子称为 VMG [@liu2019size]（构造见第 15、16 章）。
- `book/glossary.md:284` — Jegadeesh 与 Titman 发现，这样的组合在随后 3～12 个月有显著的正收益 [@jegadeesh1993returns]（第 15 章）。
- `book/glossary.md:336` — ……Frazzini 与 Pedersen 的 QMJ（quality minus junk）因子 [@asness2019quality]（第 15、16 章）。
- `book/part0/ch00-concepts.md:66` — 因子的具体构造见第 15、16 章。
- `book/part0/ch00-experiments.md:41` — 第 15 章在 `apsim.frictions` 中加入这一机制，第 17 章讨论它对资产定价检验的影响。
- `book/part0/ch00-experiments.md:42` — 月度数据中如何处理停牌股票的收益与特征，会直接影响组合收益，详见第 15 章。
- `book/part0/ch00-experiments.md:45` — 正式研究必须使用带有公告日期的时点数据，第 15 章给出对齐规则。

## 第 16 章（7 条）

- `book/glossary.md:167` — 少数几个因子能否概括大量异象（第 16、18 章）。
- `book/glossary.md:271` — CH3 在中国市场改用盈利市值比（EP），价值因子称为 VMG [@liu2019size]（构造见第 15、16 章）。
- `book/glossary.md:336` — ……Frazzini 与 Pedersen 的 QMJ（quality minus junk）因子 [@asness2019quality]（第 15、16 章）。
- `book/part0/ch00-concepts.md:66` — 因子的具体构造见第 15、16 章。
- `book/part0/ch00-concepts.md:240` — ……行上市管制，是一种与基本面无关的价值成分，既不是普通的风险补偿，也不能简单看作错误定价（见“A 股视角”一节、@liu2019size 与第 16、17 章）。
- `book/part0/ch00-exercises.md:295` — 它们分别是第 22 章与第 16、17 章的主要文献。
- `book/part0/ch00-experiments.md:44` — 第 16、17 章讨论这一处理。

## 第 17 章（5 条）

- `book/glossary.md:323` — CH3 因此剔除市值最小的 30% 的股票后再构造因子 [@liu2019size]（第 17 章）。
- `book/part0/ch00-concepts.md:240` — ……行上市管制，是一种与基本面无关的价值成分，既不是普通的风险补偿，也不能简单看作错误定价（见“A 股视角”一节、@liu2019size 与第 16、17 章）。
- `book/part0/ch00-exercises.md:295` — 它们分别是第 22 章与第 16、17 章的主要文献。
- `book/part0/ch00-experiments.md:41` — 第 15 章在 `apsim.frictions` 中加入这一机制，第 17 章讨论它对资产定价检验的影响。
- `book/part0/ch00-experiments.md:44` — 第 16、17 章讨论这一处理。

## 第 18 章（3 条）

- `book/glossary.md:167` — 少数几个因子能否概括大量异象（第 16、18 章）。
- `book/glossary.md:512` — 发表之后平均下降约 58%，多出的约 32 个百分点归因于投资者从论文中学到并交易了这些信号 [@mclean2016does]（第 18 章）。
- `book/glossary.md:516` — 二者都与策略容量和交易成本有关（第 18、26 章）。

## 第 19 章（2 条）

- `book/glossary.md:440` — 样本外检验的计量见第 13、19 章）。
- `book/glossary.md:499` — 在统计学习中，过拟合指模型过于复杂，样本内误差小而样本外误差大，用正则化与交叉验证来控制（第 19 章）。

## 第 22 章（3 条）

- `book/glossary.md:171` — 业界意义上的多因子选股，本书称为“多信号选股”或“基于特征的收益预测”，它正是第四部分收益预测（第 22 章）与可实施性（第 26 章）研究的对象。
- `book/glossary.md:460` — 若改用历史均值作基准，所有方法的 $R^2$ 约高 3 个百分点（第 13、22 章）。
- `book/part0/ch00-exercises.md:295` — 它们分别是第 22 章与第 16、17 章的主要文献。

## 第 23 章（3 条）

- `book/glossary.md:382` — 统计因子与协方差结构见第 23 章，组合构造中的风险约束见第 26 章。
- `book/part0/ch00-exercises.md:292` — ……“载荷是特征的线性函数”与“特征驱动的截距”的设定，来自 IPCA 模型 @kelly2019characteristics，第 23 章将完整介绍其估计方法。
- `book/part0/ch00-experiments.md:281` — 判断一个能预测收益的特征是风险补偿还是错误定价，要看它的预测力能否被载荷与因子风险溢价解释，而不能只看预测力本身（第 8、23 章）。

## 第 26 章（10 条）

- `book/glossary.md:167` — 扣除交易成本之后是否还在（第 26 章）；
- `book/glossary.md:171` — 业界意义上的多因子选股，本书称为“多信号选股”或“基于特征的收益预测”，它正是第四部分收益预测（第 22 章）与可实施性（第 26 章）研究的对象。
- `book/glossary.md:226` — 讨论可实施性时（第 26 章）报告最大回撤，并注明样本区间。
- `book/glossary.md:241` — 统计显著不等于能赚钱：扣除交易成本、受卖空与容量限制之后，显著的异象可能无利可图（第 26 章）。
- `book/glossary.md:316` — 用换手率等短期量价信号选股时，信号变化快，组合换手率往往很高，扣除交易成本后收益可能大幅下降（第 26 章）。
- `book/glossary.md:327` — 管理规模写作“资产管理规模”，容量写作“策略容量”（第 26 章）。
- `book/glossary.md:382` — 统计因子与协方差结构见第 23 章，组合构造中的风险约束见第 26 章。
- `book/glossary.md:425` — 可实施性与交易成本在第 26 章讨论。
- `book/glossary.md:516` — 二者都与策略容量和交易成本有关（第 18、26 章）。
- `book/part0/ch00-experiments.md:43` — A 股融券规模小、成本高，许多“空头腿”在现实中难以实现，这既影响因子收益的可实现性（第 26 章），也是错误定价能够持续存在的一个原因（第 8 章）。

## 第 27 章（1 条）

- `book/glossary.md:503` — 过拟合既指模型的复杂度，也指研究过程中的选择：选样本、选特征、选参数、选模型的每一次尝试都算在内（第 27 章）。

## 附录 A（4 条）

- `book/glossary.md:414` — ……–Waugh–Lovell 定理，收益对“信号与控制变量”做多元回归时信号的系数，等于收益对“残差化后的信号”做一元回归的系数（附录 A 讨论投影时给出证明）。
- `book/index.md:21` — 附录 A 汇总数学工具
- `book/notation.md:83` — 更多的矩阵代数与矩阵微积分工具见附录 A。
- `book/part0/ch00-exercises.md:295` — vec 算子、Kronecker 积与矩阵求导的系统介绍见附录 A。
