import numpy as np

class GlidePath:

    def __init__(self, max_age=60):
        self.max_age = max_age

    def predict(self, state):
        age = state["age"]

        # Equity decreases with age
        equity = max(0.2, 1 - age / self.max_age)

        # Keep minimum cash buffer
        cash = 0.1

        # Remaining goes to debt
        debt = max(0.0, 1 - equity - cash)

        weights = np.array([equity, debt, cash])
        return weights / (np.sum(weights) + 1e-8)