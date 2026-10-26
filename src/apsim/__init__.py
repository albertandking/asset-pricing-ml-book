"""《资产定价与机器学习：从理论到 A 股实证》配套模拟与工具包。"""

from apsim.market import MarketParams, SimulatedMarket, default_gamma, simulate_market

__version__ = "0.1.0"

__all__ = ["MarketParams", "SimulatedMarket", "default_gamma", "simulate_market"]
