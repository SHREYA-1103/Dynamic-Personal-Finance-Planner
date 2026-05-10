import numpy as np


class StateBuilder:

    # BUILD STATE
    def build(self, portfolio_state, market_state, household_state):

        state = {
            # PORTFOLIO
            "wealth": portfolio_state["wealth"],
            "w_eq": portfolio_state["w_eq"],
            "w_debt": portfolio_state["w_debt"],
            "w_cash": portfolio_state["w_cash"],

            # MARKET
            "volatility": market_state["volatility"],
            "drawdown": market_state["drawdown"],
            "market_indicator": market_state["regime"],

            # HOUSEHOLD
            "income": household_state["income"],
            "last_stable_income": household_state["last_stable_income"],
            "income_var": household_state["income_var"],
            "job_loss_prob": household_state["job_loss_prob"],
            "expense": household_state["expense"],
            "base_expense": household_state["base_expense"],
            "expense_adjusted": household_state.get("expense_adjusted", False),

            # TIME
            "age": household_state["age"],
            "goal": household_state["goal"],

            # PREFERENCE
            "risk_appetite": household_state["risk_appetite"],
            "buffer_target": household_state["buffer_target"]
        }

        return state


    # STATE → VECTOR
    def to_vector(self, state):

        # NORMALIZATION
        wealth = state["wealth"] / 1e6
        income = state["income"] / 1e5

        # expense-to-income ratio (stable + meaningful)
        if state["income"] > 0:
            expense = state["expense"] / state["income"]
        else:
            expense = state["expense"] / (state["last_stable_income"] + 1e-6)

        base_expense = state["base_expense"] / 1e5

        age = state["age"] / 100
        goal = state["goal"] / 600

        return np.array([
            # PORTFOLIO
            wealth,
            state["w_eq"],
            state["w_debt"],
            state["w_cash"],

            # MARKET
            state["volatility"],
            state["drawdown"],
            state["market_indicator"],

            # HOUSEHOLD
            income,
            state["income_var"],
            state["job_loss_prob"],
            expense,
            base_expense,
            float(state["expense_adjusted"]),

            # TIME
            age,
            goal,

            # PREFERENCE
            state["risk_appetite"],
            state["buffer_target"]
        ], dtype=np.float32)