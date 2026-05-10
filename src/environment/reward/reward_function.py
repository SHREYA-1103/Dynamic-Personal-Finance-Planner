import numpy as np

def compute_reward(
    prev_wealth,
    curr_wealth,
    drawdown,
    buffer_months,
    config,
    return_components=False
):
    eps = config.EPS

    # Growth (normalized)
    growth = np.log((curr_wealth + eps) / (prev_wealth + eps))
    growth = np.clip(growth, -0.1, 0.1)
    growth_norm = growth / 0.1   # [-1, 1]
    growth_term = config.GROWTH_WEIGHT * growth_norm

    # Drawdown (fixed)
    drawdown = abs(drawdown)     # [0, 1]
    drawdown_term = -config.DRAWDOWN_WEIGHT * drawdown

    # Buffer (normalized)
    buffer_gap = max(0.0, config.BUFFER_TARGET - buffer_months)
    buffer_norm = min(buffer_gap / config.BUFFER_TARGET, 1.0)
    buffer_term = -config.BUFFER_WEIGHT * buffer_norm

    # Ruin
    ruin_term = 0.0
    if curr_wealth <= config.RUIN_THRESHOLD:
        ruin_term = -config.RUIN_PENALTY

    reward = growth_term + drawdown_term + buffer_term + ruin_term

    if return_components:
        return reward, {
            "growth": growth_term,
            "drawdown": drawdown_term,
            "buffer": buffer_term,
            "ruin": ruin_term
        }

    return float(reward)