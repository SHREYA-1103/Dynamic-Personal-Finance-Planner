import numpy as np


class PortfolioDynamics:

    def __init__(self, config):
        self.config = config

    def reset(self, household_state):

        wealth = household_state["wealth"]
        expense = household_state["expense"]
        buffer_target = household_state["buffer_target"]
        risk_appetite = household_state["risk_appetite"]

        # REQUIRED CASH BUFFER
        cash_required = buffer_target * expense

        if wealth <= 0:
            w_cash = 1.0
            w_eq = 0.0
            w_debt = 0.0

        else:
            w_cash = min(cash_required / wealth, 1.0)

            remaining = 1.0 - w_cash
            w_eq = remaining * risk_appetite
            w_debt = remaining * (1 - risk_appetite)

        # NORMALIZE
        weights = np.array([w_eq, w_debt, w_cash])
        weights = weights / (np.sum(weights) + 1e-8)

        return {
            "wealth": wealth,
            "w_eq": weights[0],
            "w_debt": weights[1],
            "w_cash": weights[2],
            "max_wealth": wealth
        }

    def step(self, prev_state, weights, market_state, household_state):

        # UNPACK STATE
        wealth = prev_state["wealth"]
        max_wealth = prev_state["max_wealth"]

        w_prev = np.array([
            prev_state["w_eq"],
            prev_state["w_debt"],
            prev_state["w_cash"]
        ])

        w_new = np.array(weights)
        w_new = np.clip(w_new, 1e-6, 1.0)
        w_new = w_new / (np.sum(w_new) + 1e-8)

        market_returns = np.array([
            market_state["r_eq"],
            market_state["r_debt"],
            market_state["r_cash"]
        ])

        portfolio_return = 3 * np.dot(w_new, market_returns)

        turnover = np.abs(w_new - w_prev)
        cost = wealth * np.sum(self.config.KAPPA_VECTOR * turnover)

        if w_new[0] < w_prev[0]:
            cost += self.config.FIXED_SELL_FEE

        income = household_state["income"]
        expense = household_state["expense"]

        new_wealth = (
            wealth * (1 + portfolio_return)
            + income
            - expense
            - cost
        )

        new_wealth = max(new_wealth, 1.0)

        new_max_wealth = max(max_wealth, new_wealth)

        # Drawdown
        drawdown = (new_wealth - new_max_wealth) / (new_max_wealth + 1e-8)

        cash = w_new[2] * new_wealth
        buffer_months = cash / (expense + 1e-8)

        return {
            "wealth": new_wealth,
            "w_eq": w_new[0],
            "w_debt": w_new[1],
            "w_cash": w_new[2],

            "portfolio_return": portfolio_return,
            "transaction_cost": cost,
            "emergency_buffer_months": buffer_months,
            "prev_wealth": wealth,

            "max_wealth": new_max_wealth,
            "drawdown": drawdown
        }