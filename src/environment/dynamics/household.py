import numpy as np


class HouseholdDynamics:

    def __init__(self, config):
        self.config = config

    def reset(self, user):

        return {
            "wealth": user["wealth"],
            "income": user["income"],
            "income_var": user["income_var"],
            "job_loss_prob": user["job_loss_prob"],
            "expense": user["expense"],
            "base_expense": user["expense"],
            "expense_adjusted": False,
            "age": user["age"],
            "goal": user["goal"],
            "risk_appetite": user["risk_appetite"],
            "buffer_target": user["emergency_buffer_months"],
            "last_stable_income": user["income"]
        }

    def step(self, prev_state):

        income, last_stable_income = self.simulate_income(prev_state)

        temp_state = prev_state.copy()
        temp_state["income"] = income
        temp_state["last_stable_income"] = last_stable_income

        expense, base_expense, expense_adjusted = self.simulate_expense(temp_state)

        age = prev_state["age"] + 1 / 12

        return {
            "wealth": prev_state["wealth"],
            "income": income,
            "income_var": prev_state["income_var"],
            "job_loss_prob": prev_state["job_loss_prob"],
            "expense": expense,
            "base_expense": base_expense,
            "expense_adjusted": expense_adjusted,
            "age": age,
            "goal": prev_state["goal"] - 1,
            "risk_appetite": prev_state["risk_appetite"],
            "buffer_target": prev_state["buffer_target"],
            "last_stable_income": last_stable_income
        }

    def simulate_income(self, state):

        income_prev = state["income"]
        last_stable_income = state.get("last_stable_income", income_prev)
        age = state["age"]
        p_job = state["job_loss_prob"]
        sigma_income = state["income_var"]

        growth = 0.008      # 10% annual growth
        p_promotion = 0.005

        # Retirement
        if age >= 65:
            income = 0.05 * last_stable_income
            return income, last_stable_income

        # Unemployed / low income
        if income_prev < 0.3 * last_stable_income:

            # recovery
            if np.random.rand() < 0.6:
                income = np.random.uniform(0.7, 0.9) * last_stable_income
                last_stable_income = income
                return income, last_stable_income

            # fallback income
            if np.random.rand() < 0.3:
                income = np.random.uniform(0.3, 0.5) * last_stable_income
                return income, last_stable_income

            return 0.0, last_stable_income

        # Job loss
        if np.random.rand() < p_job:
            income = np.random.uniform(0.0, 0.3) * income_prev
            return income, last_stable_income

        # Normal growth
        shock = np.clip(np.random.normal(0, sigma_income), -0.15, 0.15)
        income = income_prev * (1 + growth + shock)

        # Promotion
        if np.random.rand() < p_promotion:
            income *= np.random.uniform(1.1, 1.25)

        last_stable_income = income

        return max(income, 0.0), last_stable_income

    def simulate_expense(self, state):

        base_expense_prev = state.get("base_expense", state["expense"])
        income = state["income"]
        last_stable_income = state.get("last_stable_income", income)
        wealth = state["wealth"]
        age = state["age"]

        expense_adjusted = state.get("expense_adjusted", False)

        inflation = 0.0056

        base_expense = base_expense_prev * (1 + inflation)

        min_expense = 0.25 * last_stable_income

        if income < 0.3 * last_stable_income and not expense_adjusted:
            base_expense *= 0.7
            expense_adjusted = True

        if income > 0:
            shock = income * np.random.uniform(0.02, 0.05)
        else:
            shock = 0.4 * last_stable_income * np.random.uniform(0.02, 0.04)

        leisure = 0.0
        ref_income = income if income > 0 else last_stable_income

        if ref_income < 50000:
            leisure = 0.02 * ref_income
        elif ref_income < 150000:
            leisure = 0.05 * ref_income
        else:
            leisure = np.random.uniform(0.06, 0.08) * ref_income

        expense = base_expense + shock + leisure

        # Cannot exceed affordability
        expense = min(expense, 0.6 * income + 0.02 * wealth)

        expense = max(expense, min_expense)

        return expense, base_expense, expense_adjusted