import numpy as np


class MarketDynamics:

    def __init__(self, market_df, config):
        self.market_df = market_df
        self.config = config

        self.T = len(market_df)
        self.current_index = 0


    def reset(self):
        if hasattr(self, "fixed_start") and self.fixed_start is not None:
            self.current_index = self.fixed_start
        else:
            self.current_index = np.random.randint(0, self.T)

        return self._get_state(self.current_index)


    def step(self):
        self.current_index = (self.current_index + 1) % self.T

        return self._get_state(self.current_index)

    def _get_state(self, idx):
        row = self.market_df.iloc[idx]

        r_eq = float(row["equity_return"])
        r_debt = float(row["debt_return"])
        r_cash = float(row["cash_return"])

        volatility = float(row["volatility"])
        drawdown = float(row["drawdown"])
        regime = float(row["market_indicator"])

        return {
            # RETURNS
            "r_eq": r_eq,
            "r_debt": r_debt,
            "r_cash": r_cash,

            # MARKET FEATURES
            "volatility": volatility,
            "drawdown": drawdown,
            "regime": regime
        }