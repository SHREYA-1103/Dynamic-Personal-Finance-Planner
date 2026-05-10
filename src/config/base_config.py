import numpy as np


class BaseConfig:
    """
    Final configuration for RL financial planning.

    Design:
    - Balanced growth vs risk
    - Strong but learnable ruin penalty
    - No unnecessary signals
    """

    # Environment
    DEBUG = False

    STATE_DIM = 17
    ACTION_DIM = 3


    # Transaction Cost
    KAPPA_VECTOR = np.array([
        0.0013,   # equity
        0.0002,   # debt
        0.0000    # cash
    ])

    FIXED_SELL_FEE = 20.0


    # Reward weights
    GROWTH_WEIGHT   = 1.0
    DRAWDOWN_WEIGHT = 1.5
    BUFFER_WEIGHT   = 0.8
    RUIN_PENALTY    = 5.0

    # Target emergency buffer (months)
    BUFFER_TARGET = 6

    # Survival
    RUIN_THRESHOLD = 1.0       # wealth < 1 → ruin


    # Numerical Stability
    EPS = 1e-8