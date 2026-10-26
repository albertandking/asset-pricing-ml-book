---
kernelspec:
  name: python3
  display_name: Python 3
---

# 第 0 章　如何使用本书

```{code-cell} python
from apsim import MarketParams, simulate_market, plotting

plotting.setup()
mkt = simulate_market(MarketParams(n_stocks=100, n_periods=120), seed=0)
mkt.panel.head()
```
