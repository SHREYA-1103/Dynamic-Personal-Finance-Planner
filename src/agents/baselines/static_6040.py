import numpy as np


class Static6040:
    """
    Baseline:
    - Maintain required emergency buffer in cash
    - Remaining wealth allocated 60% equity, 40% debt
    """

    def predict(self, state):
        wealth = state["wealth"]
        expense = state["expense"]
        buffer_target = state["buffer_target"]

        # --- Required cash ---
        required_cash = buffer_target * expense

        # Avoid division issues
        if wealth <= 1e-8:
            return np.array([0.0, 0.0, 1.0])

        # --- Cash weight ---
        w_cash = required_cash / wealth

        # Cap between 0 and 1
        w_cash = np.clip(w_cash, 0.0, 1.0)

        # --- Remaining allocation ---
        remaining = 1.0 - w_cash

        w_eq = 0.6 * remaining
        w_debt = 0.4 * remaining

        weights = np.array([w_eq, w_debt, w_cash])

        # Final normalization (safety)
        return weights / (np.sum(weights) + 1e-8)