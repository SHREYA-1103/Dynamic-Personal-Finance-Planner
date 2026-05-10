# import os
# import numpy as np
# import matplotlib.pyplot as plt

# from environment.financial_env import FinancialEnv
# from environment.reward.reward_function import compute_reward 


# def compute_sharpe(returns):
#     returns = np.array(returns)
#     if len(returns) < 2:
#         return 0.0
#     return np.mean(returns) / (np.std(returns) + 1e-8)


# def compute_max_drawdown(wealth_series):
#     wealth = np.array(wealth_series)
#     peak = np.maximum.accumulate(wealth)
#     drawdown = (wealth - peak) / peak
#     return np.min(drawdown)


# def evaluate(
#     model,
#     market_df,
#     user_df,
#     config,
#     algo_name="ppo",
#     num_episodes=20,
#     fixed_user_idx=None,
#     fixed_market_start=None
# ):

#     base_dir = f"outputs/logs/{algo_name}"
#     os.makedirs(base_dir, exist_ok=True)

#     final_wealths = []
#     sharpe_ratios = []
#     max_drawdowns = []
#     rewards_all = []
#     ruin_count = 0

#     for episode in range(num_episodes):

#         env = FinancialEnv(market_df, user_df, config)

#         if fixed_user_idx is not None:
#             env.fixed_user_idx = fixed_user_idx

#         if fixed_market_start is not None:
#             env.market.fixed_start = fixed_market_start

#         obs, _ = env.reset()

#         user_idx = env.current_user_idx
#         user_profile = user_df.iloc[user_idx].to_dict()

#         done = False
#         total_reward = 0

#         wealth_series = []
#         income_series = []
#         expense_series = []
#         reward_series = []

#         growth_series = []
#         drawdown_series = []
#         buffer_series = []
#         ruin_series = []

#         eq_series = []
#         debt_series = []
#         cash_series = []

#         return_series = []

#         prev_wealth = env.state["wealth"]

#         while not done:

#             if hasattr(model, "policy"):
#                 action, _ = model.predict(obs, deterministic=True)
#             else:
#                 action = model.predict(env.state)

#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             state = env.state

#             wealth = state["wealth"]
#             income = state["income"]
#             expense = state["expense"]

#             # Store values
#             wealth_series.append(wealth)
#             income_series.append(income)
#             expense_series.append(expense)
#             reward_series.append(reward)

#             eq_series.append(state["w_eq"])
#             debt_series.append(state["w_debt"])
#             cash_series.append(state["w_cash"])

#             # Returns
#             ret = (wealth - prev_wealth) / (prev_wealth + 1e-8)
#             return_series.append(ret)

#             # Reward decomposition
#             _, components = compute_reward(
#                 prev_wealth=prev_wealth,
#                 curr_wealth=wealth,
#                 drawdown=state["drawdown"],
#                 buffer_months=state["buffer_target"],
#                 config=config,
#                 return_components=True
#             )

#             growth_series.append(components["growth"])
#             drawdown_series.append(components["drawdown"])
#             buffer_series.append(components["buffer"])
#             ruin_series.append(components["ruin"])

#             prev_wealth = wealth
#             total_reward += reward

#         # Episode metrics
#         final_wealth = wealth_series[-1]
#         sharpe = compute_sharpe(return_series)
#         max_dd = compute_max_drawdown(wealth_series)

#         final_wealths.append(final_wealth)
#         sharpe_ratios.append(sharpe)
#         max_drawdowns.append(max_dd)
#         rewards_all.append(total_reward)

#         if final_wealth <= state["buffer_target"] * state["expense"]:
#             ruin_count += 1

#         # 🔹 User profile (top)
#         desc = (
#             f"User ID: {user_idx} | Age: {user_profile['age']} | Horizon: {user_profile['goal']}\n"
#             f"Wealth₀: {user_profile['wealth']:.2f} | Income: {user_profile['income']:.2f} "
#             f"(σ={user_profile['income_var']:.2f}) | Expense: {user_profile['expense']:.2f}\n"
#             f"Job Loss Prob: {user_profile['job_loss_prob']:.2f} | "
#             f"Risk Appetite: {user_profile.get('risk_appetite', 'N/A')} | "
#             f"Buffer Target: {state['buffer_target']:.0f} months"
#         )

#         # 🔹 Summary (bottom)
#         summary = (
#             f"Final Wealth: {final_wealth:.2f} | "
#             f"Sharpe: {sharpe:.3f} | "
#             f"Max Drawdown: {max_dd:.2%} | "
#             f"Total Reward: {total_reward:.2f}"
#         )

#         # 🔥 Plot
#         fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

#         # Title (no overlap)
#         fig.suptitle("Financial Trajectory Analysis", fontsize=16, fontweight='bold', y=0.995)

#         # User profile
#         fig.text(
#             0.5, 0.94,
#             desc,
#             ha='center',
#             fontsize=10,
#             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5)
#         )

#         # 1. Wealth
#         axs[0].plot(wealth_series)
#         axs[0].axhline(
#             y=state["buffer_target"] * state["expense"],
#             linestyle='--',
#             label="Ruin Threshold"
#         )
#         axs[0].set_title("Wealth Evolution")
#         axs[0].legend()

#         # 2. Income vs Expense
#         axs[1].plot(income_series, label="Income")
#         axs[1].plot(expense_series, label="Expense")
#         axs[1].set_title("Income vs Expenses")
#         axs[1].legend()

#         # 3. Allocation
#         axs[2].plot(eq_series, label="Equity")
#         axs[2].plot(debt_series, label="Debt")
#         axs[2].plot(cash_series, label="Cash")
#         axs[2].set_title("Portfolio Allocation")
#         axs[2].legend()

#         # 4. Reward breakdown
#         axs[3].plot(reward_series, label="Total Reward", linewidth=2)
#         axs[3].plot(growth_series, label="Growth")
#         axs[3].plot(drawdown_series, label="Drawdown Penalty")
#         axs[3].plot(buffer_series, label="Buffer Penalty")
#         axs[3].plot(ruin_series, label="Ruin Penalty")

#         axs[3].set_title("Reward Breakdown")
#         axs[3].legend()
#         axs[3].set_xlabel("Time Steps")

#         # Bottom summary
#         fig.text(
#             0.5, 0.02,
#             summary,
#             ha='center',
#             fontsize=11,
#             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5)
#         )

#         plt.tight_layout(rect=[0, 0.05, 1, 0.93])

#         file_path = os.path.join(base_dir, f"user_{user_idx}_ep_{episode}.png")
#         plt.savefig(file_path)
#         plt.close()

#         print(f"Saved plot for user {user_idx}, episode {episode}")

#     results = {
#         "avg_final_wealth": float(np.mean(final_wealths)),
#         "median_final_wealth": float(np.median(final_wealths)),
#         "wealth_std": float(np.std(final_wealths)),
#         "avg_sharpe": float(np.mean(sharpe_ratios)),
#         "avg_max_drawdown": float(np.mean(max_drawdowns)),
#         "ruin_probability": ruin_count / num_episodes,
#         "avg_reward": float(np.mean(rewards_all)),
#     }

#     return results



import os
import numpy as np
import matplotlib.pyplot as plt

from environment.financial_env import FinancialEnv
from environment.reward.reward_function import compute_reward 


# =========================
# METRICS
# =========================
def compute_sharpe(returns):
    returns = np.array(returns)
    if len(returns) < 2:
        return 0.0
    return np.mean(returns) / (np.std(returns) + 1e-8)


def compute_max_drawdown(wealth_series):
    wealth = np.array(wealth_series)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    return np.min(drawdown)


def compute_cvar(final_wealths, alpha=0.1):
    final_wealths = np.array(final_wealths)
    var = np.percentile(final_wealths, alpha * 100)
    cvar = final_wealths[final_wealths <= var].mean()
    return cvar


def compute_liquidity_shortfall(cash_series, expense_series):
    cash_series = np.array(cash_series)
    expense_series = np.array(expense_series)
    return np.mean(cash_series < expense_series)


def compute_recovery_time(wealth_series):
    peak = wealth_series[0]
    recovery_times = []

    for i in range(1, len(wealth_series)):
        if wealth_series[i] > peak:
            peak = wealth_series[i]
        else:
            for j in range(i + 1, len(wealth_series)):
                if wealth_series[j] >= peak:
                    recovery_times.append(j - i)
                    break

    return np.mean(recovery_times) if recovery_times else 0


# =========================
# MAIN EVALUATION
# =========================
def evaluate(
    model,
    market_df,
    user_df,
    config,
    algo_name="ppo",
    num_episodes=20,
    fixed_user_idx=None,
    fixed_market_start=None
):

    base_dir = f"outputs/logs/{algo_name}"
    os.makedirs(base_dir, exist_ok=True)

    final_wealths = []
    sharpe_ratios = []
    max_drawdowns = []
    rewards_all = []
    ruin_count = 0

    liquidity_shortfalls = []
    recovery_times = []

    for episode in range(num_episodes):

        env = FinancialEnv(market_df, user_df, config)

        if fixed_user_idx is not None:
            env.fixed_user_idx = fixed_user_idx

        if fixed_market_start is not None:
            env.market.fixed_start = fixed_market_start

        obs, _ = env.reset()

        user_idx = env.current_user_idx
        user_profile = user_df.iloc[user_idx].to_dict()

        done = False
        total_reward = 0

        wealth_series = []
        income_series = []
        expense_series = []
        reward_series = []

        growth_series = []
        drawdown_series = []
        buffer_series = []
        ruin_series = []

        eq_series = []
        debt_series = []
        cash_series = []

        return_series = []

        prev_wealth = env.state["wealth"]

        while not done:

            if hasattr(model, "policy"):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = model.predict(env.state)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = env.state

            wealth = state["wealth"]
            income = state["income"]
            expense = state["expense"]

            # Store values
            wealth_series.append(wealth)
            income_series.append(income)
            expense_series.append(expense)
            reward_series.append(reward)

            eq_series.append(state["w_eq"])
            debt_series.append(state["w_debt"])
            cash_series.append(state["w_cash"] * wealth)  # convert weight → actual cash

            # Returns
            ret = (wealth - prev_wealth) / (prev_wealth + 1e-8)
            return_series.append(ret)

            # Reward decomposition
            _, components = compute_reward(
                prev_wealth=prev_wealth,
                curr_wealth=wealth,
                drawdown=state["drawdown"],
                buffer_months=state["buffer_target"],
                config=config,
                return_components=True
            )

            growth_series.append(components["growth"])
            drawdown_series.append(components["drawdown"])
            buffer_series.append(components["buffer"])
            ruin_series.append(components["ruin"])

            prev_wealth = wealth
            total_reward += reward

        # =========================
        # METRICS PER EPISODE
        # =========================
        final_wealth = wealth_series[-1]
        sharpe = compute_sharpe(return_series)
        max_dd = compute_max_drawdown(wealth_series)

        liquidity_prob = compute_liquidity_shortfall(cash_series, expense_series)
        rec_time = compute_recovery_time(wealth_series)

        final_wealths.append(final_wealth)
        sharpe_ratios.append(sharpe)
        max_drawdowns.append(max_dd)
        rewards_all.append(total_reward)
        liquidity_shortfalls.append(liquidity_prob)
        recovery_times.append(rec_time)

        if final_wealth <= state["buffer_target"] * state["expense"]:
            ruin_count += 1

        print(f"Saved plot for user {user_idx}, episode {episode}")

    # =========================
    # FINAL METRICS
    # =========================
    cvar = compute_cvar(final_wealths, alpha=0.1)

    results = {
        "avg_final_wealth": float(np.mean(final_wealths)),
        "median_final_wealth": float(np.median(final_wealths)),
        "wealth_std": float(np.std(final_wealths)),
        "avg_sharpe": float(np.mean(sharpe_ratios)),
        "avg_max_drawdown": float(np.mean(max_drawdowns)),
        "ruin_probability": ruin_count / num_episodes,
        "avg_reward": float(np.mean(rewards_all)),

        "final_wealth": float(final_wealths[0]),
        "cvar_0.1": float(cvar),
        "liquidity_shortfall_prob": float(np.mean(liquidity_shortfalls)),
        "avg_recovery_time": float(np.mean(recovery_times)),
    }

    return results